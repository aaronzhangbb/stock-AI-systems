# -*- coding: utf-8 -*-
"""
策略验证引擎
- 对每套策略在历史数据上做滚动回测
- 计算胜率、盈亏比、最大回撤、夏普比率
- 用 ML 对多策略信号做组合加权
- 输出可信度等级(A/B/C) + 风险等级(低/中/高)
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from src.strategy.strategies import STRATEGY_REGISTRY


def validate_single_strategy(df: pd.DataFrame, strategy_func, hold_days: int = 5) -> dict:
    """
    对单个策略在历史数据上做逐日回测

    参数:
        df: 完整历史 OHLCV DataFrame（需至少 120 行）
        strategy_func: 策略函数
        hold_days: 买入后持有天数（用于计算收益）

    返回:
        dict: 胜率、盈亏比、最大回撤、夏普、交易次数、平均收益等
    """
    if df.empty or len(df) < 120:
        return _empty_result()

    data = df.sort_values('date').reset_index(drop=True)
    trades = []

    # 从第 60 行开始滚动回测（保证策略有足够历史数据）
    for i in range(60, len(data) - hold_days):
        window = data.iloc[:i + 1].copy()
        try:
            result = strategy_func(window)
        except Exception:
            continue

        if result is None:
            continue

        signal = result.get('signal')
        if signal != 'buy':
            continue

        buy_price = float(data.iloc[i]['close'])
        sell_price = float(data.iloc[i + hold_days]['close'])
        ret = (sell_price - buy_price) / buy_price

        trades.append({
            'date': data.iloc[i]['date'],
            'buy_price': buy_price,
            'sell_price': sell_price,
            'return': ret,
            'strength': result.get('strength', 0),
        })

    if not trades:
        return _empty_result()

    returns = [t['return'] for t in trades]
    returns_arr = np.array(returns)

    win_trades = [r for r in returns if r > 0]
    lose_trades = [r for r in returns if r <= 0]

    win_rate = len(win_trades) / len(returns) * 100
    avg_win = np.mean(win_trades) * 100 if win_trades else 0
    avg_lose = abs(np.mean(lose_trades)) * 100 if lose_trades else 0.01
    profit_loss_ratio = avg_win / avg_lose if avg_lose > 0 else 999

    # 计算最大回撤（基于累计收益）
    equity = (1 + returns_arr).cumprod()
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown = float(np.min(drawdown)) * 100

    # 夏普比率（年化）
    if returns_arr.std() != 0:
        sharpe = float(returns_arr.mean() / returns_arr.std() * np.sqrt(252 / hold_days))
    else:
        sharpe = 0.0

    avg_return = float(returns_arr.mean()) * 100
    total_return = float((equity[-1] - 1) * 100) if len(equity) > 0 else 0

    return {
        'total_trades': len(trades),
        'win_rate': round(win_rate, 1),
        'avg_return': round(avg_return, 2),
        'total_return': round(total_return, 2),
        'profit_loss_ratio': round(profit_loss_ratio, 2),
        'max_drawdown': round(max_drawdown, 2),
        'sharpe': round(sharpe, 2),
        'avg_win': round(avg_win, 2),
        'avg_lose': round(avg_lose, 2),
    }


def _empty_result() -> dict:
    return {
        'total_trades': 0,
        'win_rate': 0.0,
        'avg_return': 0.0,
        'total_return': 0.0,
        'profit_loss_ratio': 0.0,
        'max_drawdown': 0.0,
        'sharpe': 0.0,
        'avg_win': 0.0,
        'avg_lose': 0.0,
    }


def grade_strategy(metrics: dict) -> tuple:
    """
    根据策略回测指标输出可信度等级和风险等级

    返回:
        (confidence_grade, risk_grade)
        confidence_grade: 'A' / 'B' / 'C'
        risk_grade: '低' / '中' / '高'
    """
    if metrics['total_trades'] < 5:
        return 'C', '高'

    score = 0

    # 胜率评分 (0~30)
    wr = metrics['win_rate']
    if wr >= 55:
        score += 30
    elif wr >= 45:
        score += 20
    elif wr >= 35:
        score += 10

    # 盈亏比评分 (0~25)
    plr = metrics['profit_loss_ratio']
    if plr >= 2.0:
        score += 25
    elif plr >= 1.5:
        score += 18
    elif plr >= 1.0:
        score += 10

    # 夏普比率评分 (0~25)
    sr = metrics['sharpe']
    if sr >= 1.5:
        score += 25
    elif sr >= 0.8:
        score += 18
    elif sr >= 0.3:
        score += 10

    # 最大回撤评分 (0~20)
    mdd = abs(metrics['max_drawdown'])
    if mdd <= 10:
        score += 20
    elif mdd <= 20:
        score += 12
    elif mdd <= 30:
        score += 5

    # 可信度等级
    if score >= 65:
        confidence = 'A'
    elif score >= 40:
        confidence = 'B'
    else:
        confidence = 'C'

    # 风险等级
    risk_score = 0
    if mdd > 25:
        risk_score += 3
    elif mdd > 15:
        risk_score += 2
    elif mdd > 8:
        risk_score += 1

    vol = abs(metrics.get('avg_lose', 0))
    if vol > 5:
        risk_score += 2
    elif vol > 3:
        risk_score += 1

    if risk_score >= 4:
        risk = '高'
    elif risk_score >= 2:
        risk = '中'
    else:
        risk = '低'

    return confidence, risk


def validate_all_strategies(df: pd.DataFrame, hold_days: int = 5) -> dict:
    """
    对所有启用的策略进行验证

    参数:
        df: 单只股票的完整历史数据

    返回:
        dict: {strategy_id: {metrics, confidence_grade, risk_grade}}
    """
    results = {}
    for sid, info in STRATEGY_REGISTRY.items():
        if not info.get('enabled', True):
            continue
        metrics = validate_single_strategy(df, info['func'], hold_days)
        confidence, risk = grade_strategy(metrics)
        results[sid] = {
            'name': info['name'],
            'metrics': metrics,
            'confidence_grade': confidence,
            'risk_grade': risk,
        }
    return results


def compute_composite_score(signals: list, validations: dict) -> float:
    """
    根据策略验证结果对信号做加权组合评分

    参数:
        signals: run_all_strategies 输出的信号列表
        validations: validate_all_strategies 输出的验证结果

    返回:
        float: 0~100 的综合评分
    """
    if not signals:
        return 0.0

    grade_weight = {'A': 1.0, 'B': 0.6, 'C': 0.3}
    total_weight = 0
    weighted_score = 0

    for sig in signals:
        sid = sig.get('strategy_id', '')
        strength = sig.get('strength', 0)
        v = validations.get(sid, {})
        grade = v.get('confidence_grade', 'C')
        wr = v.get('metrics', {}).get('win_rate', 0)
        w = grade_weight.get(grade, 0.3)
        # 综合 = 信号强度 * 可信度权重 * (胜率/100)
        score = strength * w * max(wr / 100, 0.3)
        weighted_score += score
        total_weight += w

    if total_weight == 0:
        return 0.0

    composite = weighted_score / total_weight
    return round(min(composite, 100), 1)
