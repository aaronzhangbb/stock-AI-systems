# -*- coding: utf-8 -*-
"""
策略学习模块

分析已完成交易的数据, 提炼规律, 给出可操作的策略优化建议。
需要至少 10 笔完成交易才能开始学习, 20 笔以上结果更可靠。

学习维度:
    1. 最优评分阈值
    2. 止损精度 (过紧/过松)
    3. 止盈精度 (过早/过晚)
    4. 持有时间效率
    5. 仓位合理性
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from src.trading.performance import PerformanceAnalyzer

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


class StrategyLearner:
    """策略学习器"""

    MIN_TRADES_FOR_LEARNING = 10
    RELIABLE_TRADES = 20

    def __init__(self, db_path: str = None):
        self.analyzer = PerformanceAnalyzer(db_path)

    def learn(self) -> dict:
        """
        执行全量策略学习, 返回结构化的分析报告和调参建议

        返回:
            {
                'status': 'ok' | 'insufficient_data',
                'trade_count': int,
                'reliability': 'low' | 'medium' | 'high',
                'insights': [
                    {'category': '...', 'finding': '...', 'suggestion': '...', 'confidence': float},
                    ...
                ],
                'optimal_params': {
                    'score_threshold': float,
                    'avg_hold_days': float,
                    'stop_accuracy': float,
                    'target_accuracy': float,
                },
                'generated_at': str,
            }
        """
        trades = self.analyzer.get_completed_trades()
        n = len(trades)

        if n < self.MIN_TRADES_FOR_LEARNING:
            return {
                'status': 'insufficient_data',
                'trade_count': n,
                'reliability': 'none',
                'insights': [{
                    'category': '数据量',
                    'finding': f'当前仅有{n}笔完成交易, 需要至少{self.MIN_TRADES_FOR_LEARNING}笔才能开始学习',
                    'suggestion': '继续执行自动交易积累数据',
                    'confidence': 0,
                }],
                'optimal_params': {},
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }

        reliability = 'high' if n >= self.RELIABLE_TRADES else 'medium'
        insights = []

        # 1. 评分阈值分析
        insights.extend(self._analyze_score_threshold(trades))

        # 2. 止损精度分析
        insights.extend(self._analyze_stop_loss(trades))

        # 3. 止盈精度分析
        insights.extend(self._analyze_take_profit(trades))

        # 4. 持有时间分析
        insights.extend(self._analyze_hold_duration(trades))

        # 5. 整体模式
        insights.extend(self._analyze_overall_patterns(trades))

        # 推导最优参数
        optimal = self._derive_optimal_params(trades, insights)

        report = {
            'status': 'ok',
            'trade_count': n,
            'reliability': reliability,
            'insights': insights,
            'optimal_params': optimal,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # 保存到文件
        self._save_report(report)

        return report

    def _analyze_score_threshold(self, trades: pd.DataFrame) -> list:
        """分析不同评分区间的表现, 找最优买入门槛"""
        insights = []
        if 'ai_score' not in trades.columns:
            return insights

        thresholds = [75, 80, 85, 90]
        best_threshold = config.AUTO_SCORE_THRESHOLD
        best_metric = -999

        for th in thresholds:
            subset = trades[trades['ai_score'] >= th]
            if len(subset) < 3:
                continue
            win_rate = (subset['pnl_pct'] > 0).mean() * 100
            avg_pnl = subset['pnl_pct'].mean()
            # 综合指标: 胜率 * 平均收益 (同时考虑频率和质量)
            metric = win_rate * avg_pnl / 100  # 标准化
            if metric > best_metric:
                best_metric = metric
                best_threshold = th

        current_th = config.AUTO_SCORE_THRESHOLD
        if best_threshold != current_th:
            above_best = trades[trades['ai_score'] >= best_threshold]
            above_current = trades[trades['ai_score'] >= current_th]
            wr_best = (above_best['pnl_pct'] > 0).mean() * 100 if len(above_best) > 0 else 0
            wr_current = (above_current['pnl_pct'] > 0).mean() * 100 if len(above_current) > 0 else 0

            insights.append({
                'category': '评分阈值',
                'finding': (
                    f'当前阈值{current_th}分(胜率{wr_current:.0f}%, {len(above_current)}笔), '
                    f'若提高到{best_threshold}分(胜率{wr_best:.0f}%, {len(above_best)}笔)'
                ),
                'suggestion': f'建议将AUTO_SCORE_THRESHOLD从{current_th}调整为{best_threshold}',
                'confidence': min(len(above_best) / 10, 1.0),
            })
        else:
            above = trades[trades['ai_score'] >= current_th]
            wr = (above['pnl_pct'] > 0).mean() * 100 if len(above) > 0 else 0
            insights.append({
                'category': '评分阈值',
                'finding': f'当前阈值{current_th}分表现良好, 胜率{wr:.0f}%',
                'suggestion': '维持当前设置',
                'confidence': min(len(above) / 10, 1.0),
            })

        return insights

    def _analyze_stop_loss(self, trades: pd.DataFrame) -> list:
        """分析止损是否过紧或过松"""
        insights = []

        stop_trades = trades[trades['sell_reason'].str.contains('止损', na=False)]
        if stop_trades.empty:
            return insights

        n_stop = len(stop_trades)
        n_total = len(trades)
        stop_rate = n_stop / n_total * 100

        avg_loss = stop_trades['pnl_pct'].mean()

        if stop_rate > 50:
            insights.append({
                'category': '止损精度',
                'finding': f'止损触发率偏高: {stop_rate:.0f}% ({n_stop}/{n_total}笔), 平均亏损{avg_loss:.1f}%',
                'suggestion': '止损可能设置过紧, 建议适当放宽ATR倍数或检查买入时机',
                'confidence': min(n_stop / 8, 1.0),
            })
        elif stop_rate > 30:
            insights.append({
                'category': '止损精度',
                'finding': f'止损触发率正常: {stop_rate:.0f}% ({n_stop}/{n_total}笔), 平均亏损{avg_loss:.1f}%',
                'suggestion': '止损参数合理, 维持当前设置',
                'confidence': min(n_stop / 8, 1.0),
            })
        else:
            insights.append({
                'category': '止损精度',
                'finding': f'止损触发率较低: {stop_rate:.0f}%, 可能设置偏松',
                'suggestion': '可以适当收紧止损以保护资金',
                'confidence': min(n_stop / 5, 1.0),
            })

        return insights

    def _analyze_take_profit(self, trades: pd.DataFrame) -> list:
        """分析止盈是否过早"""
        insights = []

        tp_trades = trades[trades['sell_reason'].str.contains('止盈', na=False)]
        if tp_trades.empty:
            return insights

        n_tp = len(tp_trades)
        avg_gain = tp_trades['pnl_pct'].mean()

        # 如果止盈平均收益偏低, 说明目标可能过近
        if avg_gain < 3:
            insights.append({
                'category': '止盈精度',
                'finding': f'止盈触发{n_tp}次, 平均收益仅{avg_gain:.1f}%, 目标可能过近',
                'suggestion': '建议提高ATR止盈倍数, 让利润多跑一段',
                'confidence': min(n_tp / 5, 1.0),
            })
        else:
            insights.append({
                'category': '止盈精度',
                'finding': f'止盈触发{n_tp}次, 平均收益{avg_gain:.1f}%',
                'suggestion': '止盈目标设置合理',
                'confidence': min(n_tp / 5, 1.0),
            })

        return insights

    def _analyze_hold_duration(self, trades: pd.DataFrame) -> list:
        """分析持有时间与收益的关系"""
        insights = []

        if 'hold_days' not in trades.columns or trades.empty:
            return insights

        avg_days = trades['hold_days'].mean()

        # 短持有 vs 长持有
        short = trades[trades['hold_days'] <= 5]
        medium = trades[(trades['hold_days'] > 5) & (trades['hold_days'] <= 12)]
        long_ = trades[trades['hold_days'] > 12]

        parts = []
        best_group = None
        best_avg = -999

        for name, group in [('1-5天', short), ('6-12天', medium), ('12天+', long_)]:
            if group.empty:
                continue
            avg = group['pnl_pct'].mean()
            wr = (group['pnl_pct'] > 0).mean() * 100
            parts.append(f'{name}: 胜率{wr:.0f}%/均收{avg:.1f}%({len(group)}笔)')
            if avg > best_avg:
                best_avg = avg
                best_group = name

        if parts:
            insights.append({
                'category': '持有时间',
                'finding': ' | '.join(parts),
                'suggestion': f'收益最优区间为{best_group}, 可据此微调预测有效期' if best_group else '数据不足',
                'confidence': min(len(trades) / 15, 1.0),
            })

        return insights

    def _analyze_overall_patterns(self, trades: pd.DataFrame) -> list:
        """分析整体交易模式"""
        insights = []

        win_rate = (trades['pnl_pct'] > 0).mean() * 100
        avg_win = trades[trades['pnl_pct'] > 0]['pnl_pct'].mean() if (trades['pnl_pct'] > 0).any() else 0
        avg_loss = abs(trades[trades['pnl_pct'] <= 0]['pnl_pct'].mean()) if (trades['pnl_pct'] <= 0).any() else 0

        if avg_loss > 0:
            rr_ratio = avg_win / avg_loss
        else:
            rr_ratio = float('inf')

        if win_rate >= 55 and rr_ratio >= 1.5:
            quality = '优秀'
            suggestion = '策略表现良好, 保持当前参数, 可考虑适度加仓'
        elif win_rate >= 45 and rr_ratio >= 1.0:
            quality = '合格'
            suggestion = '策略可行但有优化空间, 建议关注上述各维度建议'
        else:
            quality = '需优化'
            suggestion = '策略表现不佳, 建议检查评分阈值和止损参数'

        insights.append({
            'category': '整体评价',
            'finding': f'胜率{win_rate:.0f}%, 盈亏比{rr_ratio:.1f}:1, 评级: {quality}',
            'suggestion': suggestion,
            'confidence': min(len(trades) / 15, 1.0),
        })

        return insights

    def _derive_optimal_params(self, trades: pd.DataFrame, insights: list) -> dict:
        """从分析中推导最优参数建议"""
        optimal = {
            'score_threshold': config.AUTO_SCORE_THRESHOLD,
            'avg_hold_days': round(trades['hold_days'].mean(), 1) if 'hold_days' in trades.columns else 7,
        }

        # 从 insights 中提取评分建议
        for ins in insights:
            if ins['category'] == '评分阈值' and '调整为' in ins.get('suggestion', ''):
                try:
                    val = int(ins['suggestion'].split('调整为')[1].strip().split()[0])
                    optimal['score_threshold'] = val
                except (ValueError, IndexError):
                    pass

        # 止损/止盈触发率
        stop_trades = trades[trades['sell_reason'].str.contains('止损', na=False)] if 'sell_reason' in trades.columns else pd.DataFrame()
        tp_trades = trades[trades['sell_reason'].str.contains('止盈', na=False)] if 'sell_reason' in trades.columns else pd.DataFrame()

        optimal['stop_trigger_rate'] = round(len(stop_trades) / len(trades) * 100, 1) if len(trades) > 0 else 0
        optimal['tp_trigger_rate'] = round(len(tp_trades) / len(trades) * 100, 1) if len(trades) > 0 else 0

        return optimal

    def _save_report(self, report: dict):
        """保存学习报告到文件"""
        try:
            out_path = os.path.join(DATA_DIR, 'strategy_insights.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass

    def load_latest_report(self) -> dict:
        """加载最近一次的学习报告"""
        report_path = os.path.join(DATA_DIR, 'strategy_insights.json')
        if not os.path.exists(report_path):
            return {}
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
