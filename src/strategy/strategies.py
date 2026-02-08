# -*- coding: utf-8 -*-
"""
多策略组合框架
每个策略函数接收 DataFrame（含 OHLCV），返回信号字典
全市场扫描引擎会对每只股票调用所有启用的策略
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


# ==================== 指标计算工具 ====================

def _ma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    dif = ema_fast - ema_slow
    dea = _ema(dif, signal)
    macd_bar = 2 * (dif - dea)
    return dif, dea, macd_bar


def _kdj(df: pd.DataFrame, n=9, m1=3, m2=3):
    low_n = df['low'].rolling(window=n, min_periods=n).min()
    high_n = df['high'].rolling(window=n, min_periods=n).max()
    rsv = (df['close'] - low_n) / (high_n - low_n) * 100
    k = rsv.ewm(com=m1-1, adjust=False).mean()
    d = k.ewm(com=m2-1, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def _bollinger(series: pd.Series, period=20, std_dev=2):
    mid = _ma(series, period)
    std = series.rolling(window=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


# ==================== 策略定义 ====================
# 每个策略返回:
#   signal: 'buy' / 'sell' / None
#   strength: 0~100 信号强度
#   reason: 信号原因描述


def strategy_ma_cross(df: pd.DataFrame) -> dict:
    """
    策略1: 双均线金叉/死叉
    - 买入: MA5 上穿 MA20，且 RSI < 80
    - 卖出: MA5 下穿 MA20，且 RSI > 20
    """
    if len(df) < config.MA_LONG + 2:
        return None

    ma_short = _ma(df['close'], config.MA_SHORT)
    ma_long = _ma(df['close'], config.MA_LONG)
    rsi = _rsi(df['close'], config.RSI_PERIOD)

    curr_short = ma_short.iloc[-1]
    curr_long = ma_long.iloc[-1]
    prev_short = ma_short.iloc[-2]
    prev_long = ma_long.iloc[-2]
    curr_rsi = rsi.iloc[-1]

    if pd.isna(curr_short) or pd.isna(curr_long) or pd.isna(prev_short) or pd.isna(curr_rsi):
        return None

    # 金叉
    if prev_short <= prev_long and curr_short > curr_long and curr_rsi < config.RSI_OVERBOUGHT:
        strength = min(80, 50 + abs(curr_short - curr_long) / curr_long * 1000)
        return {
            'signal': 'buy',
            'strategy': '均线金叉',
            'strength': round(strength),
            'reason': f'MA{config.MA_SHORT}上穿MA{config.MA_LONG}，RSI={curr_rsi:.1f}',
            'indicators': {'ma_short': curr_short, 'ma_long': curr_long, 'rsi': curr_rsi},
        }

    # 死叉
    if prev_short >= prev_long and curr_short < curr_long and curr_rsi > config.RSI_OVERSOLD:
        strength = min(80, 50 + abs(curr_long - curr_short) / curr_long * 1000)
        return {
            'signal': 'sell',
            'strategy': '均线死叉',
            'strength': round(strength),
            'reason': f'MA{config.MA_SHORT}下穿MA{config.MA_LONG}，RSI={curr_rsi:.1f}',
            'indicators': {'ma_short': curr_short, 'ma_long': curr_long, 'rsi': curr_rsi},
        }

    return None


def strategy_rsi_extreme(df: pd.DataFrame) -> dict:
    """
    策略2: RSI 超卖反弹 / 超买回落
    - 买入: RSI 从超卖区（<20）回升到 20 以上
    - 卖出: RSI 从超买区（>80）回落到 80 以下
    """
    if len(df) < 20:
        return None

    rsi = _rsi(df['close'], config.RSI_PERIOD)
    curr_rsi = rsi.iloc[-1]
    prev_rsi = rsi.iloc[-2]

    if pd.isna(curr_rsi) or pd.isna(prev_rsi):
        return None

    # RSI 从超卖回升
    if prev_rsi < config.RSI_OVERSOLD and curr_rsi >= config.RSI_OVERSOLD:
        strength = min(90, 60 + (config.RSI_OVERSOLD - prev_rsi) * 2)
        return {
            'signal': 'buy',
            'strategy': 'RSI超卖反弹',
            'strength': round(strength),
            'reason': f'RSI从{prev_rsi:.1f}回升到{curr_rsi:.1f}，离开超卖区',
            'indicators': {'rsi': curr_rsi, 'prev_rsi': prev_rsi},
        }

    # RSI 从超买回落
    if prev_rsi > config.RSI_OVERBOUGHT and curr_rsi <= config.RSI_OVERBOUGHT:
        strength = min(90, 60 + (prev_rsi - config.RSI_OVERBOUGHT) * 2)
        return {
            'signal': 'sell',
            'strategy': 'RSI超买回落',
            'strength': round(strength),
            'reason': f'RSI从{prev_rsi:.1f}回落到{curr_rsi:.1f}，离开超买区',
            'indicators': {'rsi': curr_rsi, 'prev_rsi': prev_rsi},
        }

    return None


def strategy_volume_breakout(df: pd.DataFrame) -> dict:
    """
    策略3: 放量突破
    - 买入: 当日成交量 > 前5日均量的2倍，且收盘价创20日新高
    - 卖出: 当日成交量 > 前5日均量的2倍，且收盘价创20日新低
    """
    if len(df) < 25:
        return None

    close = df['close'].iloc[-1]
    volume = df['volume'].iloc[-1]
    avg_vol_5 = df['volume'].iloc[-6:-1].mean()

    if pd.isna(volume) or pd.isna(avg_vol_5) or avg_vol_5 == 0:
        return None

    vol_ratio = volume / avg_vol_5
    high_20 = df['high'].iloc[-21:-1].max()
    low_20 = df['low'].iloc[-21:-1].min()

    if pd.isna(high_20) or pd.isna(low_20):
        return None

    # 放量突破新高
    if vol_ratio >= 2.0 and close > high_20:
        strength = min(95, 60 + vol_ratio * 5)
        return {
            'signal': 'buy',
            'strategy': '放量突破',
            'strength': round(strength),
            'reason': f'量比={vol_ratio:.1f}x，突破20日新高{high_20:.2f}',
            'indicators': {'vol_ratio': vol_ratio, 'high_20': high_20, 'close': close},
        }

    # 放量跌破新低
    if vol_ratio >= 2.0 and close < low_20:
        strength = min(95, 60 + vol_ratio * 5)
        return {
            'signal': 'sell',
            'strategy': '放量破位',
            'strength': round(strength),
            'reason': f'量比={vol_ratio:.1f}x，跌破20日新低{low_20:.2f}',
            'indicators': {'vol_ratio': vol_ratio, 'low_20': low_20, 'close': close},
        }

    return None


def strategy_macd_cross(df: pd.DataFrame) -> dict:
    """
    策略4: MACD 金叉/死叉
    - 买入: DIF 上穿 DEA（MACD 金叉），且在零轴附近或以下
    - 卖出: DIF 下穿 DEA（MACD 死叉），且在零轴附近或以上
    """
    if len(df) < 35:
        return None

    dif, dea, macd_bar = _macd(df['close'])

    curr_dif = dif.iloc[-1]
    curr_dea = dea.iloc[-1]
    prev_dif = dif.iloc[-2]
    prev_dea = dea.iloc[-2]

    if pd.isna(curr_dif) or pd.isna(curr_dea) or pd.isna(prev_dif) or pd.isna(prev_dea):
        return None

    # MACD 金叉（零轴下方金叉更有效）
    if prev_dif <= prev_dea and curr_dif > curr_dea:
        below_zero = curr_dif < 0
        strength = 70 if below_zero else 55
        label = '（零轴下方，强信号）' if below_zero else '（零轴上方）'
        return {
            'signal': 'buy',
            'strategy': 'MACD金叉',
            'strength': strength,
            'reason': f'DIF上穿DEA{label}，DIF={curr_dif:.3f}',
            'indicators': {'dif': curr_dif, 'dea': curr_dea},
        }

    # MACD 死叉
    if prev_dif >= prev_dea and curr_dif < curr_dea:
        above_zero = curr_dif > 0
        strength = 70 if above_zero else 55
        label = '（零轴上方，强信号）' if above_zero else '（零轴下方）'
        return {
            'signal': 'sell',
            'strategy': 'MACD死叉',
            'strength': strength,
            'reason': f'DIF下穿DEA{label}，DIF={curr_dif:.3f}',
            'indicators': {'dif': curr_dif, 'dea': curr_dea},
        }

    return None


def strategy_bollinger_band(df: pd.DataFrame) -> dict:
    """
    策略5: 布林带突破
    - 买入: 价格触及下轨后回升（从下轨外侧回到轨道内）
    - 卖出: 价格触及上轨后回落（从上轨外侧回到轨道内）
    """
    if len(df) < 25:
        return None

    upper, mid, lower = _bollinger(df['close'])

    curr_close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    curr_lower = lower.iloc[-1]
    curr_upper = upper.iloc[-1]
    prev_lower = lower.iloc[-2]
    prev_upper = upper.iloc[-2]

    if pd.isna(curr_lower) or pd.isna(curr_upper):
        return None

    # 触及下轨后反弹
    if prev_close <= prev_lower and curr_close > curr_lower:
        strength = 65
        return {
            'signal': 'buy',
            'strategy': '布林带下轨反弹',
            'strength': strength,
            'reason': f'价格从下轨{curr_lower:.2f}反弹至{curr_close:.2f}',
            'indicators': {'close': curr_close, 'lower': curr_lower, 'upper': curr_upper},
        }

    # 触及上轨后回落
    if prev_close >= prev_upper and curr_close < curr_upper:
        strength = 65
        return {
            'signal': 'sell',
            'strategy': '布林带上轨回落',
            'strength': strength,
            'reason': f'价格从上轨{curr_upper:.2f}回落至{curr_close:.2f}',
            'indicators': {'close': curr_close, 'lower': curr_lower, 'upper': curr_upper},
        }

    return None


def strategy_kdj_cross(df: pd.DataFrame) -> dict:
    """
    策略6: KDJ 金叉/死叉
    - 买入: K 上穿 D，且 J < 20（超卖区金叉更可靠）
    - 卖出: K 下穿 D，且 J > 80（超买区死叉更可靠）
    """
    if len(df) < 15:
        return None

    k, d, j = _kdj(df)
    curr_k = k.iloc[-1]
    curr_d = d.iloc[-1]
    curr_j = j.iloc[-1]
    prev_k = k.iloc[-2]
    prev_d = d.iloc[-2]

    if pd.isna(curr_k) or pd.isna(curr_d) or pd.isna(prev_k):
        return None

    # KDJ 金叉
    if prev_k <= prev_d and curr_k > curr_d:
        in_oversold = curr_j < 20
        strength = 75 if in_oversold else 55
        label = '（超卖区，强信号）' if in_oversold else ''
        return {
            'signal': 'buy',
            'strategy': 'KDJ金叉',
            'strength': strength,
            'reason': f'K上穿D{label}，J={curr_j:.1f}',
            'indicators': {'k': curr_k, 'd': curr_d, 'j': curr_j},
        }

    # KDJ 死叉
    if prev_k >= prev_d and curr_k < curr_d:
        in_overbought = curr_j > 80
        strength = 75 if in_overbought else 55
        label = '（超买区，强信号）' if in_overbought else ''
        return {
            'signal': 'sell',
            'strategy': 'KDJ死叉',
            'strength': strength,
            'reason': f'K下穿D{label}，J={curr_j:.1f}',
            'indicators': {'k': curr_k, 'd': curr_d, 'j': curr_j},
        }

    return None


# ==================== 策略注册表 ====================
# key: 策略ID
# value: {name, func, enabled_default, description}

STRATEGY_REGISTRY = {
    'ma_cross': {
        'name': '均线交叉',
        'func': strategy_ma_cross,
        'enabled': True,
        'description': f'MA{config.MA_SHORT}/MA{config.MA_LONG}金叉买入、死叉卖出，RSI过滤',
    },
    'rsi_extreme': {
        'name': 'RSI极端反转',
        'func': strategy_rsi_extreme,
        'enabled': True,
        'description': f'RSI跌破{config.RSI_OVERSOLD}后回升买入，突破{config.RSI_OVERBOUGHT}后回落卖出',
    },
    'volume_breakout': {
        'name': '放量突破',
        'func': strategy_volume_breakout,
        'enabled': True,
        'description': '成交量放大2倍以上+创20日新高/新低',
    },
    'macd_cross': {
        'name': 'MACD交叉',
        'func': strategy_macd_cross,
        'enabled': True,
        'description': 'MACD DIF上穿/下穿DEA，零轴下方金叉更强',
    },
    'bollinger': {
        'name': '布林带突破',
        'func': strategy_bollinger_band,
        'enabled': True,
        'description': '价格触及布林带上轨/下轨后反转',
    },
    'kdj_cross': {
        'name': 'KDJ交叉',
        'func': strategy_kdj_cross,
        'enabled': True,
        'description': 'KDJ金叉/死叉，超卖/超买区信号更强',
    },
}


def get_enabled_strategies() -> list:
    """获取所有启用的策略列表"""
    return [
        {'id': k, **v}
        for k, v in STRATEGY_REGISTRY.items()
        if v.get('enabled', True)
    ]


def run_all_strategies(df: pd.DataFrame, strategy_ids: list = None) -> list:
    """
    对一只股票运行所有启用的策略

    参数:
        df: OHLCV DataFrame
        strategy_ids: 指定运行哪些策略（None=全部启用的）

    返回:
        list[dict]: 触发的信号列表，每个信号包含 signal/strategy/strength/reason
    """
    signals = []

    for sid, info in STRATEGY_REGISTRY.items():
        if strategy_ids and sid not in strategy_ids:
            continue
        if not strategy_ids and not info.get('enabled', True):
            continue

        try:
            result = info['func'](df)
            if result:
                result['strategy_id'] = sid
                signals.append(result)
        except Exception:
            continue

    # 按信号强度降序排列
    signals.sort(key=lambda x: x.get('strength', 0), reverse=True)
    return signals
