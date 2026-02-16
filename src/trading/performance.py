# -*- coding: utf-8 -*-
"""
绩效分析模块

从自动交易日志和每日快照中计算:
    - 基础指标: 总收益率/胜率/盈亏比/最大回撤/Sharpe
    - 进阶分析: 按评分区间/板块/持有天数的胜率分布
    - 提供给 StrategyLearner 做策略优化的原始数据
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


class PerformanceAnalyzer:
    """绩效分析器"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), '..', '..', config.DB_PATH)
        self.db_path = db_path

    def _query_df(self, sql: str) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df

    # ================================================================
    # 交易记录
    # ================================================================

    def get_completed_trades(self) -> pd.DataFrame:
        """
        获取所有已完成的买卖配对 (从 auto_trade_log 中匹配买入和卖出)

        返回 DataFrame 列:
            stock_code, stock_name, buy_date, buy_price, sell_date, sell_price,
            shares, pnl, pnl_pct, hold_days, ai_score, reason
        """
        buys = self._query_df(
            "SELECT * FROM auto_trade_log WHERE action='买入' ORDER BY created_at"
        )
        sells = self._query_df(
            "SELECT * FROM auto_trade_log WHERE action='卖出' ORDER BY created_at"
        )

        if buys.empty or sells.empty:
            return pd.DataFrame()

        trades = []
        sell_used = set()

        for _, buy_row in buys.iterrows():
            code = buy_row['stock_code']
            buy_date = buy_row['trade_date']

            # 找到该股票第一个尚未配对的卖出
            matched_sells = sells[
                (sells['stock_code'] == code) &
                (sells['trade_date'] >= buy_date) &
                (~sells.index.isin(sell_used))
            ]

            if matched_sells.empty:
                continue

            sell_row = matched_sells.iloc[0]
            sell_used.add(sell_row.name)

            buy_dt = pd.to_datetime(buy_row['trade_date'])
            sell_dt = pd.to_datetime(sell_row['trade_date'])
            hold_days = max((sell_dt - buy_dt).days, 1)

            trades.append({
                'stock_code': code,
                'stock_name': buy_row.get('stock_name', ''),
                'buy_date': buy_row['trade_date'],
                'buy_price': buy_row['price'],
                'sell_date': sell_row['trade_date'],
                'sell_price': sell_row['price'],
                'shares': buy_row['shares'],
                'pnl': sell_row.get('pnl', 0),
                'pnl_pct': sell_row.get('pnl_pct', 0),
                'hold_days': hold_days,
                'ai_score': buy_row.get('ai_score', 0),
                'sell_reason': sell_row.get('reason', ''),
                'stop_price': buy_row.get('stop_price', 0),
                'target_price': buy_row.get('target_price', 0),
            })

        return pd.DataFrame(trades) if trades else pd.DataFrame()

    def get_open_positions_from_log(self) -> pd.DataFrame:
        """获取当前仍在持仓的买入记录 (有买入无卖出)"""
        buys = self._query_df(
            "SELECT * FROM auto_trade_log WHERE action='买入' ORDER BY created_at"
        )
        sells = self._query_df(
            "SELECT * FROM auto_trade_log WHERE action='卖出' ORDER BY created_at"
        )

        if buys.empty:
            return pd.DataFrame()

        sell_counts = sells.groupby('stock_code').size().to_dict() if not sells.empty else {}
        buy_counts = buys.groupby('stock_code').size().to_dict()

        open_codes = set()
        for code, n_buy in buy_counts.items():
            n_sell = sell_counts.get(code, 0)
            if n_buy > n_sell:
                open_codes.add(code)

        if not open_codes:
            return pd.DataFrame()

        return buys[buys['stock_code'].isin(open_codes)].drop_duplicates(
            subset='stock_code', keep='last'
        )

    # ================================================================
    # 基础绩效指标
    # ================================================================

    def compute_basic_metrics(self) -> dict:
        """
        计算核心绩效指标

        返回:
            {
                total_trades, win_count, loss_count, win_rate,
                total_pnl, avg_pnl_pct, avg_win_pct, avg_loss_pct,
                profit_factor, max_single_win, max_single_loss,
                avg_hold_days, total_return_pct, max_drawdown_pct,
                sharpe_ratio,
            }
        """
        trades = self.get_completed_trades()
        snapshots = self._query_df(
            "SELECT * FROM daily_snapshot ORDER BY date"
        )

        result = {
            'total_trades': 0,
            'win_count': 0,
            'loss_count': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_pnl_pct': 0.0,
            'avg_win_pct': 0.0,
            'avg_loss_pct': 0.0,
            'profit_factor': 0.0,
            'max_single_win': 0.0,
            'max_single_loss': 0.0,
            'avg_hold_days': 0.0,
            'total_return_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
        }

        if trades.empty:
            # 即使无交易，也从快照计算整体收益
            if not snapshots.empty:
                result['total_return_pct'] = self._calc_total_return(snapshots)
                result['max_drawdown_pct'] = self._calc_max_drawdown(snapshots)
            return result

        n = len(trades)
        wins = trades[trades['pnl_pct'] > 0]
        losses = trades[trades['pnl_pct'] <= 0]

        result['total_trades'] = n
        result['win_count'] = len(wins)
        result['loss_count'] = len(losses)
        result['win_rate'] = round(len(wins) / n * 100, 1) if n > 0 else 0
        result['total_pnl'] = round(trades['pnl'].sum(), 2)
        result['avg_pnl_pct'] = round(trades['pnl_pct'].mean(), 2)
        result['avg_win_pct'] = round(wins['pnl_pct'].mean(), 2) if not wins.empty else 0
        result['avg_loss_pct'] = round(losses['pnl_pct'].mean(), 2) if not losses.empty else 0
        result['max_single_win'] = round(trades['pnl_pct'].max(), 2)
        result['max_single_loss'] = round(trades['pnl_pct'].min(), 2)
        result['avg_hold_days'] = round(trades['hold_days'].mean(), 1)

        total_wins = wins['pnl'].sum() if not wins.empty else 0
        total_losses = abs(losses['pnl'].sum()) if not losses.empty else 0
        result['profit_factor'] = round(total_wins / total_losses, 2) if total_losses > 0 else float('inf')

        if not snapshots.empty:
            result['total_return_pct'] = self._calc_total_return(snapshots)
            result['max_drawdown_pct'] = self._calc_max_drawdown(snapshots)
            result['sharpe_ratio'] = self._calc_sharpe(snapshots)

        return result

    def _calc_total_return(self, snapshots: pd.DataFrame) -> float:
        if snapshots.empty:
            return 0.0
        initial = config.INITIAL_CAPITAL
        latest = snapshots.iloc[-1]['total_equity']
        return round((latest - initial) / initial * 100, 2)

    def _calc_max_drawdown(self, snapshots: pd.DataFrame) -> float:
        if snapshots.empty or len(snapshots) < 2:
            return 0.0
        equity = snapshots['total_equity'].values.astype(float)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        return round(float(np.min(drawdown)), 2)

    def _calc_sharpe(self, snapshots: pd.DataFrame, risk_free: float = 0.02) -> float:
        if snapshots.empty or len(snapshots) < 5:
            return 0.0
        equity = snapshots['total_equity'].values.astype(float)
        returns = np.diff(equity) / equity[:-1]
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        daily_rf = risk_free / 252
        sharpe = (np.mean(returns) - daily_rf) / np.std(returns) * np.sqrt(252)
        return round(float(sharpe), 2)

    # ================================================================
    # 进阶分析 (按维度拆解)
    # ================================================================

    def analyze_by_score_range(self) -> list:
        """按AI评分区间分析胜率和收益"""
        trades = self.get_completed_trades()
        if trades.empty or 'ai_score' not in trades.columns:
            return []

        ranges = [(80, 85), (85, 90), (90, 95), (95, 100)]
        results = []
        for low, high in ranges:
            subset = trades[(trades['ai_score'] >= low) & (trades['ai_score'] < high)]
            if subset.empty:
                continue
            n = len(subset)
            wins = len(subset[subset['pnl_pct'] > 0])
            results.append({
                'range': f'{low}-{high}',
                'count': n,
                'win_rate': round(wins / n * 100, 1),
                'avg_pnl_pct': round(subset['pnl_pct'].mean(), 2),
                'total_pnl': round(subset['pnl'].sum(), 2),
            })
        return results

    def analyze_by_hold_days(self) -> list:
        """按持有天数分析收益"""
        trades = self.get_completed_trades()
        if trades.empty:
            return []

        ranges = [(1, 3), (3, 7), (7, 14), (14, 30)]
        results = []
        for low, high in ranges:
            subset = trades[(trades['hold_days'] >= low) & (trades['hold_days'] < high)]
            if subset.empty:
                continue
            n = len(subset)
            wins = len(subset[subset['pnl_pct'] > 0])
            results.append({
                'range': f'{low}-{high}天',
                'count': n,
                'win_rate': round(wins / n * 100, 1),
                'avg_pnl_pct': round(subset['pnl_pct'].mean(), 2),
            })
        return results

    def analyze_exit_reasons(self) -> list:
        """分析各种退出原因的分布和收益"""
        trades = self.get_completed_trades()
        if trades.empty:
            return []

        reason_map = {
            '止损': [],
            '止盈': [],
            '追踪止损': [],
            '超期': [],
            '其他': [],
        }

        for _, t in trades.iterrows():
            reason = t.get('sell_reason', '')
            categorized = False
            for key in ['止损', '止盈', '追踪止损', '超期', '有效期']:
                if key in reason:
                    mapped_key = '超期' if key == '有效期' else key
                    reason_map[mapped_key].append(t['pnl_pct'])
                    categorized = True
                    break
            if not categorized:
                reason_map['其他'].append(t['pnl_pct'])

        results = []
        for reason, pnls in reason_map.items():
            if not pnls:
                continue
            arr = np.array(pnls)
            results.append({
                'reason': reason,
                'count': len(arr),
                'avg_pnl_pct': round(float(arr.mean()), 2),
                'win_rate': round(float(np.sum(arr > 0) / len(arr) * 100), 1),
            })
        return results

    def get_equity_curve(self) -> pd.DataFrame:
        """获取资产曲线数据"""
        return self._query_df("SELECT date, total_equity FROM daily_snapshot ORDER BY date")

    def get_full_report(self) -> dict:
        """获取完整绩效报告"""
        return {
            'basic': self.compute_basic_metrics(),
            'by_score': self.analyze_by_score_range(),
            'by_hold_days': self.analyze_by_hold_days(),
            'by_exit_reason': self.analyze_exit_reasons(),
            'completed_trades': self.get_completed_trades().to_dict('records'),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
