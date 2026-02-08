# -*- coding: utf-8 -*-
"""
AI量化评分模块
- 使用技术指标构建特征
- 训练轻量分类模型预测未来涨跌概率
- 输出评分与风险指标
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


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


def _bollinger(series: pd.Series, period=20, std_dev=2):
    mid = _ma(series, period)
    std = series.rolling(window=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.sort_values('date').reset_index(drop=True)

    data['ret1'] = data['close'].pct_change()
    data['ret5'] = data['close'].pct_change(5)
    data['vol_20'] = data['ret1'].rolling(20).std() * np.sqrt(252)

    data['ma5'] = _ma(data['close'], 5)
    data['ma20'] = _ma(data['close'], 20)
    data['ma_diff'] = (data['ma5'] - data['ma20']) / data['ma20']

    data['rsi14'] = _rsi(data['close'], 14)

    dif, dea, macd_bar = _macd(data['close'])
    data['macd_dif'] = dif
    data['macd_dea'] = dea
    data['macd_bar'] = macd_bar

    upper, mid, lower = _bollinger(data['close'])
    data['bb_upper'] = upper
    data['bb_mid'] = mid
    data['bb_lower'] = lower
    data['bb_pos'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])

    data['momentum_10'] = data['close'] / data['close'].shift(10) - 1
    data['vol_ratio'] = data['volume'] / data['volume'].rolling(5).mean()

    return data


def compute_risk_metrics(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 30:
        return {
            'volatility': 0.0,
            'max_drawdown': 0.0,
            'sharpe': 0.0,
            'win_rate': 0.0,
        }

    ret = df['close'].pct_change().dropna()
    if ret.empty:
        return {
            'volatility': 0.0,
            'max_drawdown': 0.0,
            'sharpe': 0.0,
            'win_rate': 0.0,
        }

    equity = (1 + ret).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    volatility = ret.std() * np.sqrt(252)
    sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() != 0 else 0.0
    win_rate = (ret > 0).mean() * 100

    return {
        'volatility': float(volatility) * 100,
        'max_drawdown': float(drawdown.min()) * 100,
        'sharpe': float(sharpe),
        'win_rate': float(win_rate),
    }


def _risk_score(metrics: dict) -> tuple:
    vol = abs(metrics.get('volatility', 0))
    mdd = abs(metrics.get('max_drawdown', 0))

    vol_score = min(100, vol * 1.5)
    mdd_score = min(100, mdd * 1.2)

    risk = round((vol_score * 0.5 + mdd_score * 0.5), 1)
    if risk >= 70:
        level = '高'
    elif risk >= 40:
        level = '中'
    else:
        level = '低'
    return risk, level


def score_stock(df: pd.DataFrame) -> dict:
    """
    对单只股票进行AI评分
    返回: score(0~100), risk_score(0~100), risk_level, metrics
    """
    if df.empty or len(df) < 60:
        return {
            'score': 0,
            'risk_score': 0,
            'risk_level': '未知',
            'confidence': 0,
            'metrics': {},
        }

    data = build_features(df)
    data['future_ret'] = data['close'].shift(-config.AI_PREDICT_HORIZON) / data['close'] - 1
    data['label'] = (data['future_ret'] > 0).astype(int)

    feature_cols = [
        'ret1', 'ret5', 'vol_20', 'ma_diff', 'rsi14',
        'macd_dif', 'macd_dea', 'macd_bar', 'bb_pos',
        'momentum_10', 'vol_ratio'
    ]
    data = data.dropna(subset=feature_cols + ['label']).copy()

    if len(data) < config.AI_MIN_TRAIN_SAMPLES:
        metrics = compute_risk_metrics(df)
        risk_score, risk_level = _risk_score(metrics)
        return {
            'score': 50,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'confidence': 0,
            'metrics': metrics,
        }

    X = data[feature_cols].values
    y = data['label'].values

    x_last = X[-1].reshape(1, -1)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        X_train, X_test, y_train, y_test = train_test_split(
            X[:-1], y[:-1], test_size=0.2, random_state=42, shuffle=True
        )
        model = LogisticRegression(max_iter=300)
        model.fit(X_train, y_train)

        prob = model.predict_proba(x_last)[0][1]
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        score = round(prob * 100, 1)
        confidence = round(acc * 100, 1)

    except Exception:
        recent = y[-60:]
        score = round(recent.mean() * 100, 1)
        confidence = 0

    metrics = compute_risk_metrics(df)
    risk_score, risk_level = _risk_score(metrics)

    return {
        'score': score,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'confidence': confidence,
        'metrics': metrics,
    }


def compute_price_targets(df: pd.DataFrame, buy_price: float = None) -> dict:
    """
    计算建议买入价、目标价、止损价

    基于技术支撑/阻力位:
    - 买入价: 最近一个支撑位（20日低点附近），或当前收盘价
    - 目标价: 基于止盈比例 + 近期阻力位
    - 止损价: 基于止损比例 + 近期支撑位

    返回: {buy_price, target_price, stop_price}
    """
    if df.empty or len(df) < 20:
        return {'buy_price': 0, 'target_price': 0, 'stop_price': 0}

    close = float(df['close'].iloc[-1])
    if buy_price is None or buy_price <= 0:
        buy_price = close

    # 近20日支撑位（最低价）
    low_20 = float(df['low'].tail(20).min())
    # 近20日阻力位（最高价）
    high_20 = float(df['high'].tail(20).max())

    # 建议买入价：收盘价与20日支撑位的加权平均（偏向支撑位）
    suggested_buy = round(close * 0.6 + low_20 * 0.4, 2)
    # 不超过当前收盘价的 102%（避免追高）
    suggested_buy = min(suggested_buy, round(close * 1.02, 2))

    # 目标价：以买入价为基础，取 止盈位 与 阻力位 中较低者
    tp_price = round(buy_price * (1 + config.TAKE_PROFIT_PCT), 2)
    target_price = round(min(tp_price, high_20 * 1.05), 2)
    # 至少比买入价高 5%
    if target_price < buy_price * 1.05:
        target_price = round(buy_price * 1.05, 2)

    # 止损价
    stop_price = round(buy_price * (1 - config.STOP_LOSS_PCT), 2)

    return {
        'buy_price': suggested_buy,
        'target_price': target_price,
        'stop_price': stop_price,
    }
