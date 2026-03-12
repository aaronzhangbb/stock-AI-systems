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
import logging
import config
from src.utils.db import connect_db

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """绩效分析器"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = config.DB_PATH
        self.db_path = os.path.abspath(db_path)

    def _query_df(self, sql: str) -> pd.DataFrame:
        conn = connect_db(self.db_path)
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
        logs = self._query_df(
            "SELECT * FROM auto_trade_log ORDER BY created_at, id"
        )

        if logs.empty:
            return pd.DataFrame()

        trades = []
        queues: dict[str, list[dict]] = {}

        for _, row in logs.iterrows():
            code = row['stock_code']
            action = row['action']
            if action == '买入':
                queues.setdefault(code, []).append({
                    'stock_name': row.get('stock_name', ''),
                    'buy_date': row['trade_date'],
                    'buy_created_at': row.get('created_at', row['trade_date']),
                    'buy_price': float(row['price']),
                    'remaining_shares': int(row['shares']),
                    'ai_score': row.get('ai_score', 0),
                    'stop_price': row.get('stop_price', 0),
                    'target_price': row.get('target_price', 0),
                })
                continue

            if action != '卖出' or code not in queues:
                continue

            remaining_sell = int(row['shares'])
            while remaining_sell > 0 and queues.get(code):
                buy_lot = queues[code][0]
                matched_shares = min(remaining_sell, buy_lot['remaining_shares'])
                buy_dt = pd.to_datetime(buy_lot['buy_created_at'])
                sell_dt = pd.to_datetime(row.get('created_at', row['trade_date']))
                hold_days = max((sell_dt - buy_dt).days, 1)
                sell_price = float(row['price'])
                buy_price = float(buy_lot['buy_price'])
                pnl = (sell_price - buy_price) * matched_shares
                pnl_pct = (sell_price - buy_price) / buy_price * 100 if buy_price > 0 else 0

                hold_trading_days = max(int(np.busday_count(
                    buy_dt.date(), sell_dt.date())), 1)
                trade_amount = round(buy_price * matched_shares, 2)

                trades.append({
                    'stock_code': code,
                    'stock_name': buy_lot.get('stock_name', ''),
                    'buy_date': buy_lot['buy_date'],
                    'buy_price': buy_price,
                    'sell_date': row['trade_date'],
                    'sell_price': sell_price,
                    'shares': matched_shares,
                    'pnl': round(pnl, 2),
                    'pnl_pct': round(pnl_pct, 2),
                    'hold_days': hold_days,
                    'hold_trading_days': hold_trading_days,
                    'trade_amount': trade_amount,
                    'ai_score': buy_lot.get('ai_score', 0),
                    'sell_reason': row.get('reason', ''),
                    'stop_price': buy_lot.get('stop_price', 0),
                    'target_price': buy_lot.get('target_price', 0),
                })

                remaining_sell -= matched_shares
                buy_lot['remaining_shares'] -= matched_shares
                if buy_lot['remaining_shares'] <= 0:
                    queues[code].pop(0)

        return pd.DataFrame(trades) if trades else pd.DataFrame()

    def get_open_positions_from_log(self) -> pd.DataFrame:
        """获取当前仍在持仓的买入记录 (有买入无卖出)"""
        logs = self._query_df(
            "SELECT * FROM auto_trade_log ORDER BY created_at, id"
        )

        if logs.empty:
            return pd.DataFrame()

        queues: dict[str, list[dict]] = {}
        for _, row in logs.iterrows():
            code = row['stock_code']
            action = row['action']
            if action == '买入':
                queues.setdefault(code, []).append(row.to_dict())
                queues[code][-1]['remaining_shares'] = int(row['shares'])
            elif action == '卖出' and code in queues:
                remaining_sell = int(row['shares'])
                while remaining_sell > 0 and queues.get(code):
                    buy_lot = queues[code][0]
                    matched = min(remaining_sell, buy_lot['remaining_shares'])
                    remaining_sell -= matched
                    buy_lot['remaining_shares'] -= matched
                    if buy_lot['remaining_shares'] <= 0:
                        queues[code].pop(0)

        open_rows = []
        for code, lots in queues.items():
            for lot in lots:
                if lot.get('remaining_shares', 0) > 0:
                    open_rows.append(lot)
        return pd.DataFrame(open_rows) if open_rows else pd.DataFrame()

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

    def analyze_post_sell_performance(self) -> pd.DataFrame:
        """
        分析每笔已卖出交易的卖后行情，判断卖出时机是否合理。

        返回 DataFrame 列:
            stock_code, stock_name, sell_date, sell_price, sell_reason,
            pnl_pct, hold_days,
            post_5d_max_pct, post_10d_max_pct, post_20d_max_pct,
            post_10d_close_pct, label
        """
        trades = self.get_completed_trades()
        if trades.empty:
            return pd.DataFrame()

        from src.data.data_fetcher import get_history_data

        results = []
        _kline_cache: dict[str, pd.DataFrame] = {}
        for _, t in trades.iterrows():
            code = t['stock_code']
            sell_date = str(t['sell_date'])[:10]
            sell_price = t['sell_price']
            if sell_price <= 0:
                continue

            try:
                if code not in _kline_cache:
                    _kline_cache[code] = get_history_data(code, days=400, use_cache=True)
                df = _kline_cache[code]
                if df.empty:
                    continue

                df['date_str'] = df['date'].astype(str).str[:10]
                sell_idx_mask = df['date_str'] >= sell_date
                if not sell_idx_mask.any():
                    continue

                sell_iloc = df.index[sell_idx_mask][0]
                pos_in_df = df.index.get_loc(sell_iloc)
                post_df = df.iloc[pos_in_df + 1:]

                if post_df.empty:
                    continue

                post_3 = post_df.head(3)
                post_5 = post_df.head(5)
                post_10 = post_df.head(10)
                post_20 = post_df.head(20)

                post_3d_max = float(post_3['high'].max()) if not post_3.empty else sell_price
                post_5d_max = float(post_5['high'].max()) if not post_5.empty else sell_price
                post_10d_max = float(post_10['high'].max()) if not post_10.empty else sell_price
                post_20d_max = float(post_20['high'].max()) if not post_20.empty else sell_price
                post_10d_close = float(post_10['close'].iloc[-1]) if len(post_10) >= 10 else (
                    float(post_df['close'].iloc[-1]) if not post_df.empty else sell_price
                )

                post_3d_max_pct = round((post_3d_max - sell_price) / sell_price * 100, 2)
                post_5d_max_pct = round((post_5d_max - sell_price) / sell_price * 100, 2)
                post_10d_max_pct = round((post_10d_max - sell_price) / sell_price * 100, 2)
                post_20d_max_pct = round((post_20d_max - sell_price) / sell_price * 100, 2)
                post_10d_close_pct = round((post_10d_close - sell_price) / sell_price * 100, 2)

                sell_reason = t.get('sell_reason', '')
                pnl_pct = t.get('pnl_pct', 0)

                if '止损' in str(sell_reason) or '追踪止损' in str(sell_reason):
                    if post_10d_close_pct < -1:
                        label = '卖对了'
                    elif post_5d_max_pct >= 5:
                        label = '卖早了'
                    elif pnl_pct < -8:
                        label = '卖晚了'
                    else:
                        label = '待观察'
                elif '止盈' in str(sell_reason):
                    if post_10d_close_pct < 0:
                        label = '卖对了'
                    elif post_10d_max_pct >= 5:
                        label = '卖早了'
                    else:
                        label = '待观察'
                else:
                    if post_10d_max_pct >= 5:
                        label = '卖早了'
                    elif pnl_pct < 0 and post_10d_close_pct < -2:
                        label = '卖晚了'
                    elif post_10d_max_pct < 3:
                        label = '卖对了'
                    else:
                        label = '待观察'

                results.append({
                    'stock_code': code,
                    'stock_name': t.get('stock_name', ''),
                    'sell_date': sell_date,
                    'sell_price': sell_price,
                    'sell_reason': sell_reason,
                    'pnl_pct': pnl_pct,
                    'hold_days': t.get('hold_days', 0),
                    'post_3d_max_pct': post_3d_max_pct,
                    'post_5d_max_pct': post_5d_max_pct,
                    'post_10d_max_pct': post_10d_max_pct,
                    'post_20d_max_pct': post_20d_max_pct,
                    'post_10d_close_pct': post_10d_close_pct,
                    'label': label,
                })
            except Exception as exc:
                logger.warning("卖后行情分析失败 %s: %s", code, exc)
                continue

        return pd.DataFrame(results) if results else pd.DataFrame()

    def analyze_post_sell_by_reason(self, post_df: pd.DataFrame = None) -> list:
        """按卖出原因分组统计卖后表现 (可传入已有 post_df 避免重复计算)"""
        if post_df is None:
            post_df = self.analyze_post_sell_performance()
        if post_df.empty:
            return []

        reason_keywords = {
            '止损': '止损',
            '止盈': '止盈',
            '追踪止损': '追踪止损',
            '策略卖出': '策略卖出',
            'AI评分衰减': 'AI评分衰减',
            '共振': '共振',
        }

        def categorize(reason_str):
            reason_str = str(reason_str)
            if '追踪止损' in reason_str:
                return '追踪止损'
            if '止损' in reason_str:
                return '止损'
            if '止盈' in reason_str:
                return '止盈'
            if 'AI评分' in reason_str or '共振' in reason_str:
                return '策略+AI确认'
            if '策略卖出' in reason_str:
                return '策略信号'
            return '其他'

        post_df['reason_cat'] = post_df['sell_reason'].apply(categorize)

        results = []
        for cat, group in post_df.groupby('reason_cat'):
            n = len(group)
            n_right = len(group[group['label'] == '卖对了'])
            n_early = len(group[group['label'] == '卖早了'])
            n_late = len(group[group['label'] == '卖晚了'])
            results.append({
                'reason': cat,
                'count': n,
                'right_pct': round(n_right / n * 100, 1) if n > 0 else 0,
                'early_pct': round(n_early / n * 100, 1) if n > 0 else 0,
                'late_pct': round(n_late / n * 100, 1) if n > 0 else 0,
                'avg_post_10d_max': round(group['post_10d_max_pct'].mean(), 2),
                'avg_post_10d_close': round(group['post_10d_close_pct'].mean(), 2),
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
