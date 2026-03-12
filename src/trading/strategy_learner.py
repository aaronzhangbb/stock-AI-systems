# -*- coding: utf-8 -*-
"""
策略学习模块 V2

分析已完成交易的数据, 提炼规律, 给出可操作的策略优化建议。
需要至少 10 笔完成交易才能开始学习, 20 笔以上结果更可靠。

学习维度:
    1. 最优评分阈值 (动态数据驱动)
    2. 止损精度 (含全局盈亏比修正)
    3. 止盈精度 (含全局盈亏比修正)
    4. 持有时间效率 (自适应分桶 + 日均收益率)
    5. 仓位合理性 (资金加权分析)
    6. 卖出时机
    7. 整体评价

闭环: 学习结果写入 strategy_insights.json -> AI扫描/自动交易读取并应用
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from src.trading.performance import PerformanceAnalyzer
from src.utils.state_store import load_json_safe, write_json_atomic
import logging

DATA_DIR = config.DATA_ROOT
logger = logging.getLogger(__name__)


class StrategyLearner:
    """策略学习器 V2 — 闭环自适应优化"""

    SCHEMA_VERSION = 2
    MIN_TRADES_FOR_LEARNING = 10
    RELIABLE_TRADES = 20

    def __init__(self, db_path: str = None):
        self.analyzer = PerformanceAnalyzer(db_path)

    # ================================================================
    # 公开接口
    # ================================================================

    def learn(self) -> dict:
        """
        执行全量策略学习, 返回 schema V2 结构化报告

        返回 dict 包含:
            schema_version, status, trade_count, reliability,
            learning_window, metrics_snapshot,
            insights (带 metadata), optimal_params, generated_at
        """
        trades = self.analyzer.get_completed_trades()
        n = len(trades)

        if n < self.MIN_TRADES_FOR_LEARNING:
            return {
                'schema_version': self.SCHEMA_VERSION,
                'status': 'insufficient_data',
                'trade_count': n,
                'reliability': 'none',
                'insights': [{
                    'category': '数据量',
                    'finding': f'当前仅有{n}笔完成交易, 需要至少{self.MIN_TRADES_FOR_LEARNING}笔才能开始学习',
                    'suggestion': '继续执行自动交易积累数据',
                    'confidence': 0,
                    'metadata': {'current_count': n, 'required_count': self.MIN_TRADES_FOR_LEARNING},
                }],
                'optimal_params': {},
                'metrics_snapshot': {},
                'learning_window': {},
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }

        reliability = 'high' if n >= self.RELIABLE_TRADES else 'medium'
        trades_w = self._apply_recency_weight(trades)

        global_metrics = self._compute_global_metrics(trades_w)

        insights = []
        insights.extend(self._analyze_score_threshold(trades_w))
        insights.extend(self._analyze_stop_loss(trades_w, global_metrics))
        insights.extend(self._analyze_take_profit(trades_w, global_metrics))
        insights.extend(self._analyze_hold_duration(trades_w))
        insights.extend(self._analyze_position_sizing(trades_w))
        insights.extend(self._analyze_sell_timing())

        optimal = self._derive_optimal_params(trades_w, insights)
        learning_window = self._build_learning_window(trades)

        auto_evolved = None
        validation_result = {}
        if getattr(config, 'AUTO_EVOLVE_ENABLED', True) and n >= getattr(config, 'AUTO_EVOLVE_MIN_TRADES', 10):
            candidate = self._generate_candidate_params(insights)
            validated, validation_result = self._validate_candidate(candidate, trades_w)
            if validated:
                self._activate_params(candidate, validation_result)
                auto_evolved = candidate
            else:
                auto_evolved = None

        insights.extend(self._analyze_overall_patterns(
            trades_w, global_metrics, insights, auto_evolved, validation_result
        ))

        report = {
            'schema_version': self.SCHEMA_VERSION,
            'status': 'ok',
            'trade_count': n,
            'reliability': reliability,
            'learning_window': learning_window,
            'metrics_snapshot': global_metrics,
            'insights': insights,
            'optimal_params': optimal,
            'auto_evolved': auto_evolved,
            'validation': validation_result,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        self._save_report(report)
        return report

    def load_latest_report(self) -> dict:
        """加载最近一次的学习报告"""
        report_path = os.path.join(DATA_DIR, 'strategy_insights.json')
        return load_json_safe(report_path, default={}, log_prefix='策略学习')

    def load_previous_report(self) -> dict:
        """加载上一次归档的学习报告 (用于对比)"""
        archive_dir = os.path.join(DATA_DIR, 'insights_archive')
        if not os.path.isdir(archive_dir):
            return {}
        files = sorted(
            [f for f in os.listdir(archive_dir) if f.endswith('.json')],
            reverse=True
        )
        if len(files) < 2:
            return {}
        return load_json_safe(
            os.path.join(archive_dir, files[1]),
            default={}, log_prefix='策略学习归档',
        )

    # ================================================================
    # 工具方法
    # ================================================================

    @staticmethod
    def _apply_recency_weight(trades: pd.DataFrame) -> pd.DataFrame:
        """给交易加时效性权重: 近30天=1.0, 30~90天=0.6, 90+天=0.3"""
        if trades.empty:
            return trades

        weights = getattr(config, 'LEARNER_RECENCY_WEIGHTS', [1.0, 0.6, 0.3])
        df = trades.copy()
        now = datetime.now()

        def _weight(sell_date_str):
            try:
                dt = pd.to_datetime(sell_date_str)
                days_ago = (now - dt).days
                if days_ago <= 30:
                    return weights[0]
                elif days_ago <= 90:
                    return weights[1]
                else:
                    return weights[2]
            except Exception:
                return weights[2]

        df['weight'] = df['sell_date'].apply(_weight)
        return df

    @staticmethod
    def _compute_global_metrics(trades: pd.DataFrame) -> dict:
        """计算全局核心指标快照 (供各分析函数引用)"""
        if trades.empty:
            return {}

        win_rate = float((trades['pnl_pct'] > 0).mean() * 100)
        wins = trades[trades['pnl_pct'] > 0]
        losses = trades[trades['pnl_pct'] <= 0]

        avg_win = float(wins['pnl_pct'].mean()) if not wins.empty else 0
        avg_loss = float(abs(losses['pnl_pct'].mean())) if not losses.empty else 0
        rr_ratio = round(avg_win / avg_loss, 2) if avg_loss > 0 else 99.0

        total_pnl = float(trades['pnl'].sum()) if 'pnl' in trades.columns else 0
        profit_factor = 0.0
        if not wins.empty and not losses.empty:
            tw = wins['pnl'].sum()
            tl = abs(losses['pnl'].sum())
            profit_factor = round(tw / tl, 2) if tl > 0 else 99.0

        return {
            'win_rate': round(win_rate, 1),
            'rr_ratio': rr_ratio,
            'avg_pnl_pct': round(float(trades['pnl_pct'].mean()), 2),
            'avg_win_pct': round(avg_win, 2),
            'avg_loss_pct': round(-avg_loss, 2),
            'profit_factor': profit_factor,
            'total_pnl': round(total_pnl, 2),
            'total_trades': len(trades),
        }

    @staticmethod
    def _weighted_mean(values, weights):
        """加权平均, 处理空序列"""
        w_sum = weights.sum()
        if w_sum == 0 or len(values) == 0:
            return 0.0
        return float((values * weights).sum() / w_sum)

    @staticmethod
    def _weighted_win_rate(pnl_pct: pd.Series, weights: pd.Series) -> float:
        """加权胜率"""
        w_sum = weights.sum()
        if w_sum == 0:
            return 0.0
        return float(((pnl_pct > 0).astype(float) * weights).sum() / w_sum * 100)

    @staticmethod
    def _build_learning_window(trades: pd.DataFrame) -> dict:
        """构建学习窗口元数据"""
        if trades.empty:
            return {}
        now = datetime.now()
        try:
            dates = pd.to_datetime(trades['sell_date'])
            return {
                'start_date': str(dates.min().date()),
                'end_date': str(dates.max().date()),
                'total_calendar_days': (dates.max() - dates.min()).days,
                'recent_30d_trades': int((dates >= (now - timedelta(days=30))).sum()),
                'recent_90d_trades': int((dates >= (now - timedelta(days=90))).sum()),
            }
        except Exception:
            return {}

    # ================================================================
    # 1. 动态评分阈值分析
    # ================================================================

    def _analyze_score_threshold(self, trades: pd.DataFrame) -> list:
        """数据驱动搜索最优 AI 评分买入阈值"""
        insights = []
        if 'ai_score' not in trades.columns or trades.empty:
            return insights

        min_bucket = getattr(config, 'LEARNER_MIN_BUCKET_SIZE', 5)
        current_th = config.AUTO_SCORE_THRESHOLD

        profitable = trades[trades['pnl_pct'] > 0]['ai_score']
        losing = trades[trades['pnl_pct'] <= 0]['ai_score']

        if len(profitable) < 3:
            insights.append({
                'category': '评分阈值',
                'finding': f'盈利交易不足3笔, 暂无法分析评分阈值',
                'suggestion': '继续积累交易数据',
                'confidence': 0,
                'metadata': {'current_threshold': current_th, 'sample_count': len(trades)},
            })
            return insights

        lower = max(float(profitable.quantile(0.3)), 60)
        upper = min(float(losing.median()) + 10 if len(losing) >= 3 else 95, 95)
        lower = min(lower, upper - 4)

        best_threshold = current_th
        best_metric = -999
        best_wr = 0
        best_count = 0

        w = trades['weight'] if 'weight' in trades.columns else pd.Series(1.0, index=trades.index)

        for th in np.arange(lower, upper + 1, 2):
            subset = trades[trades['ai_score'] >= th]
            if len(subset) < min_bucket:
                continue
            sw = w.loc[subset.index]
            wr = self._weighted_win_rate(subset['pnl_pct'], sw)
            avg_pnl = self._weighted_mean(subset['pnl_pct'], sw)
            penalty = 1.0 if len(subset) >= 10 else len(subset) / 10
            metric = wr * avg_pnl / 100 * penalty
            if metric > best_metric:
                best_metric = metric
                best_threshold = int(th)
                best_wr = wr
                best_count = len(subset)

        above_current = trades[trades['ai_score'] >= current_th]
        wr_current = self._weighted_win_rate(
            above_current['pnl_pct'],
            w.loc[above_current.index]
        ) if len(above_current) > 0 else 0

        metadata = {
            'recommended_threshold': best_threshold,
            'current_threshold': current_th,
            'sample_count': best_count,
            'expected_win_rate': round(best_wr, 1),
            'search_range': [int(lower), int(upper)],
        }

        if best_threshold != current_th:
            _dir = 'up' if best_threshold > current_th else 'down'
            insights.append({
                'category': '评分阈值',
                'finding': (
                    f'当前阈值{current_th}分(胜率{wr_current:.0f}%, {len(above_current)}笔), '
                    f'建议阈值{best_threshold}分(胜率{best_wr:.0f}%, {best_count}笔)'
                ),
                'suggestion': f'建议将评分阈值从{current_th}调整为{best_threshold}',
                'confidence': min(best_count / 10, 1.0),
                'status': _dir,
                'key_value': f'{current_th} → {best_threshold}',
                'metadata': metadata,
            })
        else:
            insights.append({
                'category': '评分阈值',
                'finding': f'当前阈值{current_th}分表现良好, 胜率{wr_current:.0f}%',
                'suggestion': '维持当前设置',
                'confidence': min(len(above_current) / 10, 1.0),
                'status': 'keep',
                'key_value': f'{current_th}分',
                'metadata': metadata,
            })

        return insights

    # ================================================================
    # 2. 止损精度分析 (含全局盈亏比修正)
    # ================================================================

    def _analyze_stop_loss(self, trades: pd.DataFrame, global_metrics: dict) -> list:
        """分析止损是否过紧或过松, 综合全局盈亏比判断"""
        insights = []
        if 'sell_reason' not in trades.columns:
            return insights

        stop_trades = trades[trades['sell_reason'].str.contains('止损', na=False)]
        if stop_trades.empty:
            return insights

        n_stop = len(stop_trades)
        n_total = len(trades)
        stop_rate = n_stop / n_total * 100
        avg_loss = float(stop_trades['pnl_pct'].mean())
        rr_ratio = global_metrics.get('rr_ratio', 0)
        overall_profitable = global_metrics.get('avg_pnl_pct', 0) > 0

        if stop_rate > 50 and not (rr_ratio >= 1.5 and overall_profitable):
            action = 'loosen'
            finding = f'止损触发率偏高: {stop_rate:.0f}% ({n_stop}/{n_total}笔), 平均亏损{avg_loss:.1f}%'
            suggestion = '止损可能设置过紧, 建议适当放宽ATR倍数或检查买入时机'
        elif stop_rate > 50 and rr_ratio >= 1.5 and overall_profitable:
            action = 'maintain'
            finding = (f'止损触发率{stop_rate:.0f}%, 但盈亏比{rr_ratio:.1f}:1且整体盈利, '
                       f'高止损率被高盈利弥补')
            suggestion = '虽然止损频繁但策略整体健康, 暂不需调整'
        elif stop_rate > 30:
            action = 'maintain'
            finding = f'止损触发率正常: {stop_rate:.0f}% ({n_stop}/{n_total}笔), 平均亏损{avg_loss:.1f}%'
            suggestion = '止损参数合理, 维持当前设置'
        else:
            action = 'tighten'
            finding = f'止损触发率较低: {stop_rate:.0f}%, 可能设置偏松'
            suggestion = '可以适当收紧止损以保护资金'

        _status_map = {'loosen': 'watch', 'maintain': 'keep', 'tighten': 'watch'}
        _kv_map = {
            'loosen': f'触发率{stop_rate:.0f}% 偏高',
            'maintain': f'触发率{stop_rate:.0f}% 正常',
            'tighten': f'触发率{stop_rate:.0f}% 偏低',
        }
        insights.append({
            'category': '止损精度',
            'finding': finding,
            'suggestion': suggestion,
            'confidence': min(n_stop / 8, 1.0),
            'status': _status_map.get(action, 'keep'),
            'key_value': _kv_map.get(action, f'触发率{stop_rate:.0f}%'),
            'metadata': {
                'stop_rate': round(stop_rate, 1),
                'avg_stop_loss_pct': round(avg_loss, 2),
                'global_rr_ratio': rr_ratio,
                'recommended_action': action,
            },
        })

        return insights

    # ================================================================
    # 3. 止盈精度分析 (含全局盈亏比修正)
    # ================================================================

    def _analyze_take_profit(self, trades: pd.DataFrame, global_metrics: dict) -> list:
        """分析止盈是否过早, 综合止损亏损比较"""
        insights = []
        if 'sell_reason' not in trades.columns:
            return insights

        tp_trades = trades[trades['sell_reason'].str.contains('止盈', na=False)]
        if tp_trades.empty:
            return insights

        n_tp = len(tp_trades)
        avg_gain = float(tp_trades['pnl_pct'].mean())
        avg_sl_loss = abs(global_metrics.get('avg_loss_pct', 0))
        effective_rr = round(avg_gain / avg_sl_loss, 2) if avg_sl_loss > 0 else 99.0

        if avg_gain < 3 and effective_rr < 1.2:
            action = 'raise_target'
            finding = (f'止盈触发{n_tp}次, 平均收益仅{avg_gain:.1f}%, '
                       f'实际盈亏比{effective_rr:.1f}:1偏低')
            suggestion = '建议提高ATR止盈倍数, 让利润多跑一段'
        elif avg_gain < 3 and effective_rr >= 1.2:
            action = 'maintain'
            finding = (f'止盈触发{n_tp}次, 平均收益{avg_gain:.1f}%, '
                       f'但对比止损亏损的实际盈亏比{effective_rr:.1f}:1仍健康')
            suggestion = '止盈目标虽不高但盈亏比合理, 暂可维持'
        else:
            action = 'maintain'
            finding = f'止盈触发{n_tp}次, 平均收益{avg_gain:.1f}%'
            suggestion = '止盈目标设置合理'

        _tp_status = 'watch' if action == 'raise_target' else 'keep'
        _tp_kv = f'均收{avg_gain:.1f}% 偏低' if action == 'raise_target' else f'均收{avg_gain:.1f}%'
        insights.append({
            'category': '止盈精度',
            'finding': finding,
            'suggestion': suggestion,
            'confidence': min(n_tp / 5, 1.0),
            'status': _tp_status,
            'key_value': _tp_kv,
            'metadata': {
                'avg_tp_gain': round(avg_gain, 2),
                'avg_sl_loss': round(avg_sl_loss, 2),
                'effective_rr': effective_rr,
                'tp_count': n_tp,
                'recommended_action': action,
            },
        })

        return insights

    # ================================================================
    # 4. 持有时间效率分析 (自适应分桶 + 日均收益率)
    # ================================================================

    def _analyze_hold_duration(self, trades: pd.DataFrame) -> list:
        """分析持有时间效率, 核心指标: 日均收益率 = pnl_pct / hold_trading_days"""
        insights = []

        hold_col = 'hold_trading_days' if 'hold_trading_days' in trades.columns else 'hold_days'
        if hold_col not in trades.columns or trades.empty:
            return insights

        min_bucket = getattr(config, 'LEARNER_MIN_BUCKET_SIZE', 5)
        w = trades['weight'] if 'weight' in trades.columns else pd.Series(1.0, index=trades.index)

        hold_days = trades[hold_col]
        trades = trades.copy()
        trades['daily_return'] = trades['pnl_pct'] / trades[hold_col].clip(lower=1)

        q33 = hold_days.quantile(0.33)
        q66 = hold_days.quantile(0.66)
        q33 = max(q33, 2)
        q66 = max(q66, q33 + 1)

        buckets = [
            (f'1-{int(q33)}天', trades[hold_days <= q33]),
            (f'{int(q33)+1}-{int(q66)}天', trades[(hold_days > q33) & (hold_days <= q66)]),
            (f'{int(q66)+1}天+', trades[hold_days > q66]),
        ]

        bucket_data = []
        best_bucket = None
        best_daily = -999

        for name, group in buckets:
            if len(group) < min_bucket:
                continue
            gw = w.loc[group.index]
            daily_ret = self._weighted_mean(group['daily_return'], gw)
            wr = self._weighted_win_rate(group['pnl_pct'], gw)
            bucket_data.append({
                'range': name, 'count': len(group),
                'daily_return': round(daily_ret, 3), 'win_rate': round(wr, 1),
            })
            if daily_ret > best_daily:
                best_daily = daily_ret
                best_bucket = name

        if bucket_data:
            parts = [f"{b['range']}: 日均{b['daily_return']:.2f}%/胜率{b['win_rate']:.0f}%({b['count']}笔)"
                     for b in bucket_data]
            avg_hd = round(float(trades[hold_col].mean()), 1)

            insights.append({
                'category': '持有时间',
                'finding': ' | '.join(parts),
                'suggestion': f'时间效率最优区间为{best_bucket}, 可据此微调预测有效期' if best_bucket else '数据不足',
                'confidence': min(len(trades) / 15, 1.0),
                'status': 'good' if best_bucket else 'watch',
                'key_value': best_bucket if best_bucket else '数据不足',
                'metadata': {
                    'best_bucket': best_bucket,
                    'best_daily_return': round(best_daily, 3),
                    'best_hold_range': best_bucket,
                    'avg_hold_days': avg_hd,
                    'buckets': bucket_data,
                },
            })

        return insights

    # ================================================================
    # 5. 仓位合理性分析 (新增)
    # ================================================================

    def _analyze_position_sizing(self, trades: pd.DataFrame) -> list:
        """按AI评分档位分析仓位与收益关系"""
        insights = []
        if trades.empty or 'ai_score' not in trades.columns:
            return insights

        has_amount = 'trade_amount' in trades.columns
        if not has_amount:
            return insights

        min_bucket = getattr(config, 'LEARNER_MIN_BUCKET_SIZE', 5)
        w = trades['weight'] if 'weight' in trades.columns else pd.Series(1.0, index=trades.index)

        score_bins = [(90, 100, '90+'), (85, 90, '85-90'), (80, 85, '80-85'), (75, 80, '75-80')]
        position_map = {}
        bin_findings = []

        total_amount = trades['trade_amount'].sum()
        if total_amount <= 0:
            return insights

        for low, high, label in score_bins:
            subset = trades[(trades['ai_score'] >= low) & (trades['ai_score'] < high)] if high < 100 else \
                     trades[trades['ai_score'] >= low]
            if len(subset) < min_bucket:
                continue

            sw = w.loc[subset.index]
            wr = self._weighted_win_rate(subset['pnl_pct'], sw)
            avg_pnl = self._weighted_mean(subset['pnl_pct'], sw)
            amount_weighted_return = float(
                (subset['pnl_pct'] * subset['trade_amount']).sum() / subset['trade_amount'].sum()
            ) if subset['trade_amount'].sum() > 0 else 0

            position_map[label] = round(
                min(max(avg_pnl / 5, 0.05), config.MAX_SINGLE_POSITION), 2
            ) if avg_pnl > 0 else config.MIN_SINGLE_POSITION

            bin_findings.append(f'{label}分: 胜率{wr:.0f}%, 资金加权收益{amount_weighted_return:.1f}%')

        if not bin_findings:
            return insights

        median_amount = trades['trade_amount'].median()
        large = trades[trades['trade_amount'] > median_amount]
        small = trades[trades['trade_amount'] <= median_amount]
        large_wr = float((large['pnl_pct'] > 0).mean() * 100) if not large.empty else 0
        small_wr = float((small['pnl_pct'] > 0).mean() * 100) if not small.empty else 0

        high_score_better = position_map.get('90+', 0) > position_map.get('75-80', 0)

        consecutive_losses = 0
        max_consec = 0
        for pnl in trades.sort_values('sell_date')['pnl_pct']:
            if pnl <= 0:
                consecutive_losses += 1
                max_consec = max(max_consec, consecutive_losses)
            else:
                consecutive_losses = 0

        concentration = 'low' if max_consec <= 3 else ('medium' if max_consec <= 5 else 'high')

        insights.append({
            'category': '仓位合理性',
            'finding': ' | '.join(bin_findings) + f' | 大仓胜率{large_wr:.0f}% vs 小仓胜率{small_wr:.0f}%',
            'suggestion': (
                '高评分标的可适度加仓' if high_score_better
                else '各评分区间表现接近, 建议均衡配置'
            ),
            'confidence': min(len(trades) / 15, 1.0),
            'status': 'up' if high_score_better else 'keep',
            'key_value': '高分加仓' if high_score_better else '均衡配置',
            'metadata': {
                'high_score_should_overweight': high_score_better,
                'recommended_position_map': position_map,
                'large_position_win_rate': round(large_wr, 1),
                'small_position_win_rate': round(small_wr, 1),
                'concentration_risk': concentration,
                'max_consecutive_losses': max_consec,
            },
        })

        return insights

    # ================================================================
    # 6. 卖出时机分析
    # ================================================================

    def _analyze_sell_timing(self) -> list:
        """分析卖出时机: 先输出总览, 再按卖出原因逐条输出子结论"""
        insights = []
        try:
            post_df = self.analyzer.analyze_post_sell_performance()
            if post_df.empty:
                return insights

            by_reason = self.analyzer.analyze_post_sell_by_reason(post_df=post_df)

            n_total = len(post_df)
            n_right = len(post_df[post_df['label'] == '卖对了'])
            n_early = len(post_df[post_df['label'] == '卖早了'])
            n_late = len(post_df[post_df['label'] == '卖晚了'])
            right_pct = round(n_right / n_total * 100, 1) if n_total > 0 else 0
            early_pct = round(n_early / n_total * 100, 1) if n_total > 0 else 0
            late_pct = round(n_late / n_total * 100, 1) if n_total > 0 else 0

            if early_pct > 40:
                _sell_status, _sell_kv = 'watch', f'卖早{early_pct:.0f}%'
            elif right_pct >= 50:
                _sell_status, _sell_kv = 'good', f'卖对{right_pct:.0f}%'
            else:
                _sell_status, _sell_kv = 'bad', '需关注'

            insights.append({
                'category': '卖出时机',
                'finding': (
                    f'共{n_total}笔: 卖对{n_right}笔({right_pct}%) / '
                    f'卖早{n_early}笔({early_pct}%) / 卖晚{n_late}笔({late_pct}%)'
                ),
                'suggestion': '卖早比例偏高，可适当放宽卖出条件' if early_pct > 40 else (
                    '卖出时机整体合理' if right_pct >= 50 else '需关注止损触发精度'
                ),
                'confidence': min(n_total / 10, 1.0),
                'status': _sell_status,
                'key_value': _sell_kv,
                'metadata': {
                    'total_analyzed': n_total,
                    'right_pct': right_pct,
                    'early_pct': early_pct,
                    'late_pct': late_pct,
                },
            })

            for rc in by_reason:
                if rc['count'] < 2:
                    continue
                r_early = rc['early_pct']
                r_right = rc['right_pct']
                reason_name = rc['reason']

                if r_early > 40:
                    r_status, r_kv = 'watch', f'卖早{r_early:.0f}%'
                    r_suggestion = f'「{reason_name}」规则偏敏感, 建议放宽或加AI确认'
                elif r_right >= 50:
                    r_status, r_kv = 'good', f'卖对{r_right:.0f}%'
                    r_suggestion = f'「{reason_name}」规则表现良好, 维持'
                else:
                    r_status, r_kv = 'keep', f'{rc["count"]}笔'
                    r_suggestion = f'「{reason_name}」规则需更多数据验证'

                insights.append({
                    'category': '卖出时机',
                    'finding': (
                        f'「{reason_name}」{rc["count"]}笔: '
                        f'卖对{r_right:.0f}% / 卖早{r_early:.0f}% / 卖晚{rc["late_pct"]:.0f}%'
                        f'（卖后10天最高涨{rc["avg_post_10d_max"]:+.1f}%）'
                    ),
                    'suggestion': r_suggestion,
                    'confidence': min(rc['count'] / 5, 1.0),
                    'status': r_status,
                    'key_value': f'{reason_name}: {r_kv}',
                    'metadata': {
                        'reason': reason_name,
                        'count': rc['count'],
                        'right_pct': r_right,
                        'early_pct': r_early,
                        'late_pct': rc['late_pct'],
                        'avg_post_10d_max': rc['avg_post_10d_max'],
                        'avg_post_10d_close': rc['avg_post_10d_close'],
                    },
                })
        except Exception as exc:
            logger.warning("卖出时机分析失败: %s", exc)
        return insights

    # ================================================================
    # 7. 整体评价
    # ================================================================

    def _analyze_overall_patterns(self, trades: pd.DataFrame, global_metrics: dict,
                                   all_insights: list = None,
                                   auto_evolved: dict = None,
                                   validation_result: dict = None) -> list:
        """三段式整体评价: 已自动执行 / 学习中 / 待积累数据"""
        insights = []
        win_rate = global_metrics.get('win_rate', 0)
        rr_ratio = global_metrics.get('rr_ratio', 0)
        all_insights = all_insights or []

        auto_applied = []
        learning = []
        need_data = []

        for ins in all_insights:
            cat = ins.get('category', '')
            st = ins.get('status', 'keep')
            meta = ins.get('metadata', {})
            kv = ins.get('key_value', '')

            if cat == '评分阈值':
                if st in ('up', 'down'):
                    auto_applied.append(f'评分阈值: {kv}')
                else:
                    auto_applied.append(f'评分阈值: 维持{kv}')

            elif cat == '仓位合理性':
                auto_applied.append(f'仓位策略: {kv}')

            elif cat == '止损精度':
                action = meta.get('recommended_action', 'maintain')
                if action == 'maintain':
                    auto_applied.append(f'止损策略: 维持')
                else:
                    action_label = {'loosen': '放宽', 'tighten': '收紧'}.get(action, action)
                    if auto_evolved:
                        auto_applied.append(f'止损策略: {action_label} (已验证生效)')
                    else:
                        learning.append(f'止损策略: 建议{action_label}, 验证未通过')

            elif cat == '止盈精度':
                action = meta.get('recommended_action', 'maintain')
                if action == 'maintain':
                    auto_applied.append(f'止盈策略: 维持')
                else:
                    if auto_evolved and auto_evolved.get('take_profit', {}).get('target_multi_factor', 1.0) > 1.0:
                        auto_applied.append(f'止盈目标: 放大{int((auto_evolved["take_profit"]["target_multi_factor"]-1)*100)}%')
                    else:
                        learning.append(f'止盈目标: 建议提高, 验证未通过')

            elif cat == '持有时间':
                if meta.get('best_bucket'):
                    if auto_evolved and auto_evolved.get('hold', {}).get('est_hold_days_override'):
                        auto_applied.append(f'持有期: 调整为{auto_evolved["hold"]["est_hold_days_override"]}天')
                    else:
                        learning.append(f'持有期: 候选{meta.get("avg_hold_days")}天, 待验证')
                else:
                    need_data.append(f'持有期: 数据不足')

            elif cat == '卖出时机' and meta.get('total_analyzed'):
                early = meta.get('early_pct', 0)
                if early > 50:
                    if auto_evolved and auto_evolved.get('sell_rule', {}).get('ai_sell_score_drop', 15) > 15:
                        auto_applied.append(f'卖出规则: 已收紧AI确认 (卖早{early:.0f}%)')
                    else:
                        learning.append(f'卖出规则: 卖早{early:.0f}%, 正在优化')
                elif early > 30:
                    learning.append(f'卖出规则: 卖早{early:.0f}%, 持续监测中')

        min_evolve_trades = getattr(config, 'AUTO_EVOLVE_MIN_TRADES', 10)
        if len(trades) < min_evolve_trades:
            need_data.append(f'自动进化需{min_evolve_trades}笔交易, 当前{len(trades)}笔')

        n_auto = len(auto_applied)
        n_learn = len(learning)
        n_need = len(need_data)

        parts = []
        if auto_applied:
            parts.append('已自动执行:\n' + '\n'.join(f'  - {x}' for x in auto_applied))
        if learning:
            parts.append('学习中:\n' + '\n'.join(f'  - {x}' for x in learning))
        if need_data:
            parts.append('待积累数据:\n' + '\n'.join(f'  - {x}' for x in need_data))

        finding_detail = '\n'.join(parts) if parts else '暂无进化动作'

        if n_auto > 0 and n_learn == 0:
            overall_status = 'good'
            kv = f'自动进化{n_auto}项'
            suggestion = f'本轮已自动优化{n_auto}项策略参数, 系统运行正常'
        elif n_learn > 0:
            overall_status = 'keep'
            kv = f'学习中{n_learn}项'
            suggestion = f'已自动执行{n_auto}项, 另有{n_learn}项正在学习验证中'
        else:
            overall_status = 'watch'
            kv = f'待积累{n_need}项'
            suggestion = '交易数据不足, 继续积累交易以启动自动进化'

        v_info = ''
        if validation_result:
            v_info = f' | 验证: 候选{validation_result.get("candidate_avg_pnl", 0):.2f}% vs 当前{validation_result.get("current_avg_pnl", 0):.2f}%'

        insights.append({
            'category': '整体评价',
            'finding': (
                f'胜率{win_rate:.0f}%, 盈亏比{rr_ratio:.1f}:1{v_info}\n\n{finding_detail}'
            ),
            'suggestion': suggestion,
            'confidence': min(len(trades) / 15, 1.0),
            'status': overall_status,
            'key_value': kv,
            'metadata': {
                'win_rate': round(win_rate, 1),
                'rr_ratio': rr_ratio,
                'total_trades': len(trades),
                'auto_applied_count': n_auto,
                'learning_count': n_learn,
                'need_data_count': n_need,
                'auto_applied': auto_applied,
                'learning': learning,
                'need_data': need_data,
            },
        })

        return insights

    # ================================================================
    # 参数推导 (从 metadata 结构化取值, 不再解析字符串)
    # ================================================================

    def _derive_optimal_params(self, trades: pd.DataFrame, insights: list) -> dict:
        """从分析洞察的 metadata 中直接提取最优参数"""
        hold_col = 'hold_trading_days' if 'hold_trading_days' in trades.columns else 'hold_days'
        optimal = {
            'score_threshold': config.AUTO_SCORE_THRESHOLD,
            'avg_hold_days': round(float(trades[hold_col].mean()), 1) if hold_col in trades.columns else 7,
            'recommended_position_map': {},
            'recommended_stop_multi': None,
            'recommended_target_multi': None,
        }

        for ins in insights:
            meta = ins.get('metadata', {})
            cat = ins.get('category', '')

            if cat == '评分阈值' and meta.get('recommended_threshold'):
                optimal['score_threshold'] = meta['recommended_threshold']

            elif cat == '仓位合理性' and meta.get('recommended_position_map'):
                optimal['recommended_position_map'] = meta['recommended_position_map']

        if 'sell_reason' in trades.columns:
            stop_trades = trades[trades['sell_reason'].str.contains('止损', na=False)]
            tp_trades = trades[trades['sell_reason'].str.contains('止盈', na=False)]
            optimal['stop_trigger_rate'] = round(len(stop_trades) / len(trades) * 100, 1) if len(trades) > 0 else 0
            optimal['tp_trigger_rate'] = round(len(tp_trades) / len(trades) * 100, 1) if len(trades) > 0 else 0
        else:
            optimal['stop_trigger_rate'] = 0
            optimal['tp_trigger_rate'] = 0

        return optimal

    # ================================================================
    # 自动进化: 候选参数生成 + 反事实验证 + 激活
    # ================================================================

    ACTIVE_PARAMS_PATH = os.path.join(DATA_DIR, 'active_strategy_params.json')

    DEFAULT_ACTIVE_PARAMS = {
        'version': 3,
        'sell_rule': {
            'ai_sell_score_drop': 15,
            'min_sell_resonance': 2,
            'require_ai_confirm': True,
            'time_decay_tighten_factor': 1.0,
        },
        'take_profit': {
            'target_multi_factor': 1.0,
        },
        'hold': {
            'phase2_start': 0.7,
            'phase3_start': 1.0,
            'phase4_start': 1.5,
            'est_hold_days_override': None,
        },
    }

    def _load_current_active_params(self) -> dict:
        """加载当前已激活的策略参数"""
        import copy
        base = copy.deepcopy(self.DEFAULT_ACTIVE_PARAMS)
        loaded = load_json_safe(self.ACTIVE_PARAMS_PATH, default={}, log_prefix='激活参数')
        if loaded:
            for section in ('sell_rule', 'take_profit', 'hold'):
                if section in loaded:
                    base.setdefault(section, {}).update(loaded[section])
            base['version'] = loaded.get('version', 3)
            base['updated_at'] = loaded.get('updated_at', '')
        return base

    def _generate_candidate_params(self, insights: list) -> dict:
        """从 insights 的 metadata 中生成候选策略参数"""
        import copy
        current = self._load_current_active_params()
        candidate = copy.deepcopy(current)

        for ins in insights:
            meta = ins.get('metadata', {})
            cat = ins.get('category', '')

            if cat == '止损精度':
                action = meta.get('recommended_action', 'maintain')
                factor_map = {'loosen': 0.85, 'tighten': 1.15, 'maintain': 1.0}
                candidate['sell_rule']['time_decay_tighten_factor'] = factor_map.get(action, 1.0)

            elif cat == '止盈精度':
                action = meta.get('recommended_action', 'maintain')
                if action == 'raise_target':
                    candidate['take_profit']['target_multi_factor'] = 1.2
                else:
                    candidate['take_profit']['target_multi_factor'] = 1.0

            elif cat == '持有时间':
                avg_hd = meta.get('avg_hold_days')
                best_bucket = meta.get('best_bucket')
                best_daily = meta.get('best_daily_return', 0)
                if best_bucket and best_daily > 0 and avg_hd:
                    candidate['hold']['est_hold_days_override'] = round(avg_hd, 1)
                else:
                    candidate['hold']['est_hold_days_override'] = None

            elif cat == '卖出时机':
                early_pct = meta.get('early_pct', 0)
                if early_pct > 50:
                    candidate['sell_rule']['require_ai_confirm'] = True
                    candidate['sell_rule']['min_sell_resonance'] = 2
                    cur_drop = current['sell_rule'].get('ai_sell_score_drop', 15)
                    candidate['sell_rule']['ai_sell_score_drop'] = min(cur_drop + 3, 30)

        return candidate

    def _validate_candidate(self, candidate: dict, trades: pd.DataFrame) -> tuple:
        """
        用历史交易做反事实验证: 模拟候选参数下的收益表现。
        返回 (是否通过, 验证详情 dict)。
        """
        min_trades = getattr(config, 'AUTO_EVOLVE_MIN_TRADES', 10)
        min_improvement = getattr(config, 'AUTO_EVOLVE_MIN_IMPROVEMENT', 0.05)

        if trades.empty or len(trades) < min_trades:
            return False, {
                'passed': False,
                'reason': f'交易数不足 ({len(trades)}/{min_trades})',
                'sample_count': len(trades),
            }

        current_avg_pnl = float(trades['pnl_pct'].mean())

        tp_factor = candidate.get('take_profit', {}).get('target_multi_factor', 1.0)
        tighten_factor = candidate.get('sell_rule', {}).get('time_decay_tighten_factor', 1.0)

        adjusted_pnls = []
        for _, t in trades.iterrows():
            pnl = t['pnl_pct']
            reason = str(t.get('sell_reason', ''))

            if '止盈' in reason and tp_factor > 1.0:
                pnl = pnl * min(tp_factor, 1.5)
            elif '止损' in reason and tighten_factor < 1.0:
                post_max = t.get('post_5d_max_pct', t.get('pnl_pct', 0))
                if hasattr(t, 'post_5d_max_pct') and 'post_5d_max_pct' in t.index:
                    if t['post_5d_max_pct'] > 3:
                        pnl = pnl * 0.5
            adjusted_pnls.append(pnl)

        candidate_avg_pnl = float(np.mean(adjusted_pnls)) if adjusted_pnls else current_avg_pnl

        improvement = (candidate_avg_pnl - current_avg_pnl) / abs(current_avg_pnl) if current_avg_pnl != 0 else 0

        result = {
            'passed': improvement >= min_improvement,
            'current_avg_pnl': round(current_avg_pnl, 3),
            'candidate_avg_pnl': round(candidate_avg_pnl, 3),
            'improvement_pct': round(improvement * 100, 1),
            'threshold_pct': round(min_improvement * 100, 1),
            'sample_count': len(trades),
        }

        if result['passed']:
            logger.info("[自动进化] 候选参数验证通过: 提升%.1f%% (阈值%.1f%%)",
                        result['improvement_pct'], result['threshold_pct'])
        else:
            logger.info("[自动进化] 候选参数未通过验证: 提升%.1f%% < 阈值%.1f%%",
                        result['improvement_pct'], result['threshold_pct'])

        return result['passed'], result

    def _activate_params(self, candidate: dict, validation: dict):
        """将验证通过的候选参数写入激活文件"""
        candidate['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        candidate['validation'] = validation
        try:
            write_json_atomic(self.ACTIVE_PARAMS_PATH, candidate)
            logger.info("[自动进化] 新参数已激活: %s", self.ACTIVE_PARAMS_PATH)
        except Exception as exc:
            logger.warning("[自动进化] 激活参数写入失败: %s", exc)

    # ================================================================
    # 保存与归档
    # ================================================================

    def _save_report(self, report: dict):
        """保存学习报告到文件 + 归档历史版本"""
        try:
            out_path = os.path.join(DATA_DIR, 'strategy_insights.json')
            write_json_atomic(out_path, report)
        except Exception as exc:
            logger.warning("保存策略学习报告失败: %s", exc)
            return

        try:
            archive_dir = os.path.join(DATA_DIR, 'insights_archive')
            os.makedirs(archive_dir, exist_ok=True)
            date_str = datetime.now().strftime('%Y-%m-%d')
            archive_path = os.path.join(archive_dir, f'{date_str}.json')
            write_json_atomic(archive_path, report)
        except Exception as exc:
            logger.warning("归档策略学习报告失败: %s", exc)
