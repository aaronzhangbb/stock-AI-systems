# -*- coding: utf-8 -*-
"""
AI 策略挖掘与全市场回测分析
从全市场股票的3年历史数据中，用机器学习自动发现投资策略
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import sys
import json
import warnings
import time
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'stock_cache.db')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'ai_strategy_report.json')

INITIAL_CAPITAL = 100000.0
COMMISSION_RATE = 0.0003
STAMP_TAX_RATE = 0.001
HOLD_DAYS_LIST = [3, 5, 10]
MAX_STOCKS = 500
MIN_DATA_ROWS = 400
TEST_RATIO = 0.3


def log(msg):
    print(msg, flush=True)


# ============================================================
# Feature Engineering
# ============================================================
def build_rich_features(df):
    data = df.copy()
    data = data.sort_values('date').reset_index(drop=True)
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    open_ = data['open']

    # Trend
    for p in [5, 10, 20, 30, 60]:
        ma = close.rolling(p).mean()
        data[f'ma{p}'] = ma
        data[f'ma{p}_diff'] = (close - ma) / ma
        if p >= 10:
            data[f'ma{p}_slope'] = ma.pct_change(5)

    data['trend_score'] = (
        (data.get('ma5', close) > data.get('ma10', close)).astype(int) +
        (data.get('ma10', close) > data.get('ma20', close)).astype(int) +
        (data.get('ma20', close) > data.get('ma60', close)).astype(int)
    ) / 3.0

    for p in [12, 26]:
        data[f'ema{p}'] = close.ewm(span=p, adjust=False).mean()

    dif = data['ema12'] - data['ema26']
    dea = dif.ewm(span=9, adjust=False).mean()
    data['macd_dif'] = dif
    data['macd_dea'] = dea
    data['macd_bar'] = 2 * (dif - dea)
    data['macd_cross'] = np.sign(dif - dea)

    # Momentum
    for p in [6, 14, 24]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss_ = (-delta).where(delta < 0, 0.0)
        ag = gain.rolling(p, min_periods=p).mean()
        al = loss_.rolling(p, min_periods=p).mean()
        rs = ag / al
        data[f'rsi{p}'] = 100 - (100 / (1 + rs))

    for p in [3, 5, 10, 20]:
        data[f'mom{p}'] = close / close.shift(p) - 1

    data['roc5'] = close.pct_change(5)
    data['roc10'] = close.pct_change(10)
    data['roc20'] = close.pct_change(20)

    hh14 = high.rolling(14).max()
    ll14 = low.rolling(14).min()
    data['williams_r14'] = (hh14 - close) / (hh14 - ll14) * -100

    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9) * 100
    data['kdj_k'] = rsv.ewm(com=2, adjust=False).mean()
    data['kdj_d'] = data['kdj_k'].ewm(com=2, adjust=False).mean()
    data['kdj_j'] = 3 * data['kdj_k'] - 2 * data['kdj_d']

    # Volatility
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift(1)),
        'lc': abs(low - close.shift(1))
    }).max(axis=1)
    data['atr14'] = tr.rolling(14).mean()
    data['atr14_pct'] = data['atr14'] / close

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    data['bb_pos'] = (close - bb_lower) / (bb_upper - bb_lower)
    data['bb_width'] = (bb_upper - bb_lower) / bb_mid

    ret1 = close.pct_change()
    data['vol_5'] = ret1.rolling(5).std() * np.sqrt(252)
    data['vol_20'] = ret1.rolling(20).std() * np.sqrt(252)
    data['vol_60'] = ret1.rolling(60).std() * np.sqrt(252)
    data['vol_ratio_5_20'] = data['vol_5'] / data['vol_20']

    # Volume
    vol_ma5 = volume.rolling(5).mean()
    vol_ma20 = volume.rolling(20).mean()
    data['vol_ratio'] = volume / vol_ma5
    data['vol_ratio_20'] = volume / vol_ma20
    data['vol_trend'] = vol_ma5 / vol_ma20

    obv = (np.sign(close.diff()) * volume).cumsum()
    data['obv_slope5'] = obv.pct_change(5)
    data['obv_slope20'] = obv.pct_change(20)
    data['vol_price_corr'] = close.pct_change(5).rolling(20).corr(volume.pct_change(5))

    # Pattern
    body = abs(close - open_)
    wick = high - low
    data['body_ratio'] = body / (wick + 1e-10)
    data['upper_shadow'] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / (wick + 1e-10)
    data['lower_shadow'] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / (wick + 1e-10)

    up = (close > close.shift(1)).astype(int)
    data['consec_up'] = up.groupby((up != up.shift()).cumsum()).cumsum()
    down = (close < close.shift(1)).astype(int)
    data['consec_down'] = down.groupby((down != down.shift()).cumsum()).cumsum()

    data['dist_high20'] = close / high.rolling(20).max() - 1
    data['dist_low20'] = close / low.rolling(20).min() - 1
    data['dist_high60'] = close / high.rolling(60).max() - 1
    data['dist_low60'] = close / low.rolling(60).min() - 1

    if 'turnover' in df.columns:
        data['turnover_feat'] = pd.to_numeric(df['turnover'], errors='coerce')
        data['turnover_ma5'] = data['turnover_feat'].rolling(5).mean()

    return data


# ============================================================
# Feature name mapping
# ============================================================
FEATURE_CN = {
    'ma5_diff': '价格偏离MA5', 'ma10_diff': '价格偏离MA10', 'ma20_diff': '价格偏离MA20',
    'ma30_diff': '价格偏离MA30', 'ma60_diff': '价格偏离MA60',
    'ma10_slope': 'MA10斜率', 'ma20_slope': 'MA20斜率', 'ma30_slope': 'MA30斜率',
    'ma60_slope': 'MA60斜率', 'trend_score': '均线多头排列度',
    'macd_dif': 'MACD-DIF', 'macd_dea': 'MACD-DEA', 'macd_bar': 'MACD柱', 'macd_cross': 'MACD方向',
    'rsi6': 'RSI(6)', 'rsi14': 'RSI(14)', 'rsi24': 'RSI(24)',
    'mom3': '3日动量', 'mom5': '5日动量', 'mom10': '10日动量', 'mom20': '20日动量',
    'roc5': '5日ROC', 'roc10': '10日ROC', 'roc20': '20日ROC',
    'williams_r14': '威廉指标', 'kdj_k': 'KDJ-K', 'kdj_d': 'KDJ-D', 'kdj_j': 'KDJ-J',
    'atr14_pct': 'ATR占比', 'bb_pos': '布林带位置', 'bb_width': '布林带宽',
    'vol_5': '5日波动率', 'vol_20': '20日波动率', 'vol_60': '60日波动率',
    'vol_ratio_5_20': '短/长波动比', 'vol_ratio': '量比(5日)', 'vol_ratio_20': '量比(20日)',
    'vol_trend': '量能趋势', 'obv_slope5': 'OBV-5日斜率', 'obv_slope20': 'OBV-20日斜率',
    'vol_price_corr': '量价相关性', 'body_ratio': 'K线实体比', 'upper_shadow': '上影线比',
    'lower_shadow': '下影线比', 'consec_up': '连涨天数', 'consec_down': '连跌天数',
    'dist_high20': '距20日新高', 'dist_low20': '距20日新低',
    'dist_high60': '距60日新高', 'dist_low60': '距60日新低',
}


def fcn(name):
    return FEATURE_CN.get(name, name)


# ============================================================
# Load data
# ============================================================
def load_all_stock_data(max_stocks=MAX_STOCKS, min_rows=MIN_DATA_ROWS):
    if not os.path.exists(DB_PATH):
        log("[ERR] DB not found: " + DB_PATH)
        return {}

    conn = sqlite3.connect(DB_PATH)
    meta = pd.read_sql_query(f"""
        SELECT m.stock_code, n.stock_name, COUNT(k.date) as cnt
        FROM cache_meta m
        JOIN daily_kline k ON m.stock_code = k.stock_code
        LEFT JOIN stock_names n ON m.stock_code = n.stock_code
        GROUP BY m.stock_code
        HAVING cnt >= {min_rows}
        ORDER BY RANDOM()
        LIMIT {max_stocks}
    """, conn)

    log(f"[DATA] Found {len(meta)} stocks with >= {min_rows} rows")

    stock_data = {}
    loaded = 0
    for _, row in meta.iterrows():
        code = row['stock_code']
        name = row['stock_name'] if pd.notna(row.get('stock_name')) else code

        df = pd.read_sql_query(
            'SELECT date, open, high, low, close, volume, amount, pctChg, turnover '
            'FROM daily_kline WHERE stock_code = ? ORDER BY date',
            conn, params=[code]
        )
        if df.empty or len(df) < min_rows:
            continue

        df['date'] = pd.to_datetime(df['date'])
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg', 'turnover']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df[df['close'] > 0].reset_index(drop=True)
        if len(df) < min_rows:
            continue

        stock_data[code] = {'name': name, 'df': df}
        loaded += 1
        if loaded % 100 == 0:
            log(f"  ... loaded {loaded} stocks")

    conn.close()
    log(f"[DATA] Loaded {len(stock_data)} stocks total")
    return stock_data


# ============================================================
# Build dataset
# ============================================================
def build_dataset(stock_data, horizon=5, threshold=0.02):
    all_frames = []
    processed = 0

    for code, info in stock_data.items():
        df = info['df']
        try:
            data = build_rich_features(df)
        except Exception:
            continue

        data['future_ret'] = data['close'].shift(-horizon) / data['close'] - 1
        data['label'] = (data['future_ret'] > threshold).astype(int)
        data['stock_code'] = code
        all_frames.append(data)
        processed += 1
        if processed % 100 == 0:
            log(f"  ... features built for {processed} stocks")

    if not all_frames:
        return pd.DataFrame(), []

    combined = pd.concat(all_frames, ignore_index=True)

    exclude_cols = {'date', 'open', 'high', 'low', 'close', 'volume', 'amount',
                    'pctChg', 'turnover', 'future_ret', 'label', 'stock_code',
                    'ma5', 'ma10', 'ma20', 'ma30', 'ma60', 'ema12', 'ema26',
                    'bb_upper', 'bb_mid', 'bb_lower', 'atr14', 'turnover_ma5', 'turnover_feat'}
    feature_cols = [c for c in combined.columns if c not in exclude_cols
                    and combined[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    feature_cols = [c for c in feature_cols if combined[c].notna().sum() > len(combined) * 0.5]

    log(f"[FEAT] {len(feature_cols)} features, {len(combined)} rows")
    return combined, feature_cols


# ============================================================
# Strategy discovery
# ============================================================
def discover_strategies(combined, feature_cols, horizon=5):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    valid = combined.dropna(subset=feature_cols + ['label', 'future_ret']).copy()
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)

    if len(valid) < 1000:
        log("[ERR] Not enough valid samples")
        return [], {}

    valid = valid.sort_values('date').reset_index(drop=True)
    split_idx = int(len(valid) * (1 - TEST_RATIO))
    train = valid.iloc[:split_idx]
    test = valid.iloc[split_idx:]

    # Subsample training set if too large (speed up training)
    if len(train) > 80000:
        train_sample = train.sample(n=80000, random_state=42)
        log(f"[TRAIN] Subsampled train from {len(train)} to 80000")
    else:
        train_sample = train

    X_train = train_sample[feature_cols].values
    y_train = train_sample['label'].values
    X_test = test[feature_cols].values
    y_test = test['label'].values

    log(f"[TRAIN] Train: {len(train_sample)}, Test: {len(test)}")
    log(f"[TRAIN] Positive rate - Train: {y_train.mean():.1%}, Test: {y_test.mean():.1%}")

    # Model 1: GBM (reduced for speed)
    log("\n[MODEL] Training GradientBoosting...")
    gbm = GradientBoostingClassifier(
        n_estimators=80, max_depth=4, learning_rate=0.1,
        min_samples_leaf=100, subsample=0.8, random_state=42
    )
    gbm.fit(X_train, y_train)
    gbm_acc = float((gbm.predict(X_test) == y_test).mean())
    log(f"  GBM accuracy: {gbm_acc:.1%}")

    importances = gbm.feature_importances_
    feat_rank = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    log("  Top 10 features:")
    for fname, imp in feat_rank[:10]:
        log(f"    {fcn(fname)}: {imp:.4f}")

    # Model 2: RF
    log("\n[MODEL] Training RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_leaf=80,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_acc = float((rf.predict(X_test) == y_test).mean())
    log(f"  RF accuracy: {rf_acc:.1%}")

    # Extract rules
    strategies = []

    # Method A: Decision trees
    log("\n[RULES] Extracting from decision trees...")
    for depth in [2, 3, 4]:
        for seed in range(25):
            tree = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=max(100, len(X_train) // 100),
                random_state=seed
            )
            tree.fit(X_train, y_train)
            rules = _extract_tree_rules(tree, feature_cols)
            strategies.extend(rules)

    log(f"  Tree rules: {len(strategies)}")

    # Method B: Quantile strategies
    log("[RULES] Building quantile strategies...")
    top_features = [f for f, _ in feat_rank[:15]]
    strategies.extend(_build_quantile_strategies(train_sample, feature_cols, top_features))

    # Method C: Typed strategies
    log("[RULES] Building typed strategies...")
    strategies.extend(_build_typed_strategies(train_sample, feature_cols))

    # Deduplicate
    seen = set()
    unique = []
    for s in strategies:
        key = s['description']
        if key not in seen:
            seen.add(key)
            unique.append(s)

    log(f"\n[RULES] {len(unique)} candidate strategies")

    # Backtest on test set
    log("\n[BACKTEST] Validating on out-of-sample data...")
    validated = []
    for i, strat in enumerate(unique):
        if i % 20 == 0:
            log(f"  ... backtesting {i}/{len(unique)}")
        result = backtest_strategy(strat, test, feature_cols, horizon)
        if result is not None:
            result['model_accuracy'] = {'gbm': round(gbm_acc, 4), 'rf': round(rf_acc, 4)}
            validated.append(result)

    validated.sort(
        key=lambda r: r['backtest']['sharpe'] * r['backtest']['win_rate'] / 100,
        reverse=True
    )

    final = validated[:20]
    for i, s in enumerate(final):
        s['rank'] = i + 1

    log(f"[DONE] {len(validated)} valid strategies, keeping Top {len(final)}")

    global_info = {
        'total_stocks': int(combined['stock_code'].nunique()),
        'total_samples': int(len(valid)),
        'train_samples': int(len(train)),
        'test_samples': int(len(test)),
        'feature_count': len(feature_cols),
        'gbm_accuracy': round(gbm_acc, 4),
        'rf_accuracy': round(rf_acc, 4),
        'top_features': [
            {'name': f, 'cn_name': fcn(f), 'importance': round(float(imp), 4)}
            for f, imp in feat_rank[:20]
        ],
    }

    return final, global_info


def _extract_tree_rules(tree, feature_names, min_samples=50, min_rate=0.33):
    tree_ = tree.tree_
    # Compute base rate from root
    root_total = tree_.n_node_samples[0]
    root_pos = tree_.value[0][0][1] if tree_.value[0].shape[1] > 1 else 0
    base_rate = root_pos / root_total if root_total > 0 else 0.3
    # A rule must beat the base rate by at least 15%
    effective_min_rate = max(min_rate, base_rate * 1.15)
    rules = []

    def _recurse(node, conditions):
        if tree_.feature[node] == -2:
            total = tree_.n_node_samples[node]
            pos = tree_.value[node][0][1] if tree_.value[node].shape[1] > 1 else 0
            rate = pos / total if total > 0 else 0

            if rate >= effective_min_rate and total >= min_samples:
                conds_cn = []
                conds_raw = []
                for feat, op, thresh in conditions:
                    conds_cn.append(f"{fcn(feat)} {op} {thresh:.4f}")
                    conds_raw.append((feat, op, float(thresh)))

                rules.append({
                    'conditions': conds_raw,
                    'description': ' AND '.join(conds_cn),
                    'positive_rate': round(rate, 4),
                    'sample_count': int(total),
                    'source': 'decision_tree',
                })
            return

        feat = feature_names[tree_.feature[node]]
        threshold = tree_.threshold[node]
        _recurse(tree_.children_left[node], conditions + [(feat, '<=', threshold)])
        _recurse(tree_.children_right[node], conditions + [(feat, '>', threshold)])

    _recurse(0, [])
    return rules


def _build_quantile_strategies(train_df, feature_cols, top_features):
    strategies = []
    for feat in top_features[:10]:
        if feat not in train_df.columns:
            continue
        vals = train_df[feat].dropna()
        if len(vals) < 100:
            continue

        base_rate = train_df['label'].mean()
        for q_low, q_high, desc in [
            (0.0, 0.2, 'LOW'), (0.8, 1.0, 'HIGH'),
            (0.0, 0.1, 'VLOW'), (0.9, 1.0, 'VHIGH')
        ]:
            low_val = float(vals.quantile(q_low))
            high_val = float(vals.quantile(q_high))
            mask = (train_df[feat] >= low_val) & (train_df[feat] <= high_val)
            subset = train_df[mask]
            if len(subset) < 50:
                continue
            pos_rate = subset['label'].mean()
            # Must beat base rate by 15%
            if pos_rate < base_rate * 1.15:
                continue

            strategies.append({
                'conditions': [(feat, '>=', low_val), (feat, '<=', high_val)],
                'description': f"{fcn(feat)} in {desc} range [{low_val:.4f}, {high_val:.4f}]",
                'positive_rate': round(float(pos_rate), 4),
                'sample_count': int(len(subset)),
                'source': 'quantile',
            })
    return strategies


def _build_typed_strategies(train_df, feature_cols):
    strategies = []

    def try_combo(name, conditions):
        mask = pd.Series([True] * len(train_df), index=train_df.index)
        conds_raw = []
        conds_cn = []
        for feat, op, thresh in conditions:
            if feat not in train_df.columns:
                return
            if op == '>':
                mask = mask & (train_df[feat] > thresh)
            elif op == '>=':
                mask = mask & (train_df[feat] >= thresh)
            elif op == '<':
                mask = mask & (train_df[feat] < thresh)
            elif op == '<=':
                mask = mask & (train_df[feat] <= thresh)
            conds_raw.append((feat, op, float(thresh)))
            conds_cn.append(f"{fcn(feat)} {op} {thresh:.4f}")
        subset = train_df[mask]
        if len(subset) < 30:
            return
        base_rate = train_df['label'].mean()
        pos_rate = subset['label'].mean()
        # Must beat base rate by 10%
        if pos_rate < base_rate * 1.10:
            return
        strategies.append({
            'conditions': conds_raw,
            'description': f"[{name}] " + ' AND '.join(conds_cn),
            'positive_rate': round(float(pos_rate), 4),
            'sample_count': int(len(subset)),
            'source': 'typed_combo',
            'type_name': name,
        })

    # Trend following
    try_combo('趋势跟踪', [('trend_score', '>', 0.6), ('macd_cross', '>', 0.5), ('mom5', '>', 0.0)])
    try_combo('强趋势', [('trend_score', '>', 0.9), ('ma20_slope', '>', 0.0), ('rsi14', '>', 50), ('vol_trend', '>', 1.0)])
    try_combo('趋势+均线斜率', [('ma20_slope', '>', 0.005), ('ma60_slope', '>', 0.0), ('mom10', '>', 0.0)])

    # Mean reversion
    try_combo('超卖反弹', [('rsi14', '<=', 30), ('bb_pos', '<=', 0.2)])
    try_combo('深度超卖', [('rsi6', '<=', 20), ('consec_down', '>=', 3), ('dist_low20', '<=', 0.02)])
    try_combo('布林带底部', [('bb_pos', '<=', 0.1), ('vol_ratio', '>=', 1.5)])
    try_combo('RSI超卖+量缩', [('rsi14', '<=', 25), ('vol_ratio', '<=', 0.8)])

    # Momentum breakout
    try_combo('放量突破', [('vol_ratio', '>=', 2.0), ('dist_high20', '>=', -0.02), ('mom5', '>', 0.02)])
    try_combo('新高突破', [('dist_high20', '>=', 0.0), ('vol_trend', '>', 1.2), ('rsi14', '>', 55)])
    bw_q30 = train_df['bb_width'].quantile(0.3) if 'bb_width' in train_df else 0.1
    try_combo('缩量蓄势', [('vol_ratio_5_20', '<=', 0.7), ('bb_width', '<=', bw_q30), ('ma20_diff', '>=', -0.05)])

    # Volume-price
    try_combo('量价齐升', [('vol_price_corr', '>', 0.3), ('mom5', '>', 0.0), ('vol_trend', '>', 1.0)])
    try_combo('OBV领先', [('obv_slope5', '>', 0.05), ('mom5', '<=', 0.01), ('rsi14', '<=', 55)])

    # Multi-indicator
    try_combo('MACD+KDJ双金叉', [('macd_cross', '>', 0.5), ('kdj_j', '<=', 30)])
    try_combo('RSI+MACD共振', [('rsi14', '<=', 40), ('macd_bar', '>', 0), ('ma5_diff', '>', 0)])
    try_combo('三线金叉', [('macd_cross', '>', 0.5), ('trend_score', '>=', 0.66), ('vol_trend', '>', 1.0)])

    # Low volatility
    vol_q25 = train_df['vol_20'].quantile(0.25) if 'vol_20' in train_df else 0.2
    bw_q25 = train_df['bb_width'].quantile(0.25) if 'bb_width' in train_df else 0.08
    try_combo('低波蓄势', [('vol_20', '<=', vol_q25), ('bb_width', '<=', bw_q25), ('trend_score', '>=', 0.5)])

    # Pattern
    try_combo('连跌后企稳', [('consec_down', '>=', 4), ('lower_shadow', '>=', 0.4)])
    try_combo('大阳线突破', [('body_ratio', '>=', 0.7), ('mom3', '>', 0.03), ('vol_ratio', '>=', 1.5)])

    return strategies


# ============================================================
# Backtest
# ============================================================
def backtest_strategy(strategy, test_df, feature_cols, horizon):
    conditions = strategy['conditions']
    trades = []

    for code, group in test_df.groupby('stock_code'):
        group = group.sort_values('date').reset_index(drop=True)
        i = 0
        while i < len(group) - horizon:
            row = group.iloc[i]
            match = True
            for feat, op, thresh in conditions:
                val = row.get(feat)
                if pd.isna(val):
                    match = False
                    break
                if op == '<=' and val > thresh:
                    match = False
                    break
                if op == '>' and val <= thresh:
                    match = False
                    break
                if op == '>=' and val < thresh:
                    match = False
                    break
                if op == '<' and val >= thresh:
                    match = False
                    break
            if match:
                bp = float(row['close'])
                sp = float(group.iloc[i + horizon]['close'])
                cost = bp * COMMISSION_RATE + sp * (COMMISSION_RATE + STAMP_TAX_RATE)
                ret = (sp - bp) / bp - cost / bp
                trades.append({'date': str(row['date']), 'stock': code, 'ret': ret})
                i += horizon
            else:
                i += 1

    if len(trades) < 15:
        return None

    returns = np.array([t['ret'] for t in trades])
    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    win_rate = len(wins) / len(returns) * 100
    avg_return = float(returns.mean()) * 100
    avg_win = float(wins.mean()) * 100 if len(wins) > 0 else 0
    avg_lose = float(abs(losses.mean())) * 100 if len(losses) > 0 else 0.01
    plr = avg_win / avg_lose if avg_lose > 0 else 999

    equity = (1 + returns).cumprod()
    total_return = float((equity[-1] - 1) * 100)
    peak = np.maximum.accumulate(equity)
    max_dd = float(((equity - peak) / peak).min()) * 100

    sharpe = 0
    if returns.std() > 0:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252 / horizon))

    n_days = (pd.to_datetime(trades[-1]['date']) - pd.to_datetime(trades[0]['date'])).days
    annual_return = 0
    if n_days > 30:
        annual_return = float((pow(max(equity[-1], 0.01), 365 / n_days) - 1) * 100)

    if win_rate >= 55 and sharpe >= 0.8 and plr >= 1.3:
        grade = 'A+'
    elif win_rate >= 52 and sharpe >= 0.5 and plr >= 1.1:
        grade = 'A'
    elif win_rate >= 48 and sharpe >= 0.2:
        grade = 'B'
    elif win_rate >= 45 and avg_return > 0:
        grade = 'C'
    else:
        grade = 'D'

    if grade == 'D':
        return None

    return {
        'strategy_id': f"ai_{hash(strategy['description']) % 10000:04d}",
        'description': strategy['description'],
        'conditions': strategy['conditions'],
        'source': strategy.get('source', 'unknown'),
        'type_name': strategy.get('type_name', 'AI挖掘'),
        'train_positive_rate': strategy['positive_rate'],
        'train_samples': strategy['sample_count'],
        'grade': grade,
        'backtest': {
            'horizon': horizon,
            'total_trades': len(trades),
            'win_rate': round(win_rate, 2),
            'avg_return_per_trade': round(avg_return, 3),
            'total_return': round(total_return, 2),
            'annual_return': round(annual_return, 2),
            'profit_loss_ratio': round(plr, 2),
            'max_drawdown': round(max_dd, 2),
            'sharpe': round(sharpe, 2),
            'avg_win': round(avg_win, 3),
            'avg_lose': round(avg_lose, 3),
        },
    }


# ============================================================
# Combo strategies
# ============================================================
def build_combo_strategies(single_strategies, test_df, feature_cols, horizon):
    log("\n[COMBO] Building combo strategies...")
    combos = []
    good = [s for s in single_strategies if s['grade'] in ('A+', 'A')]
    if len(good) < 2:
        good = single_strategies[:6]

    for i in range(min(len(good), 8)):
        for j in range(i + 1, min(len(good), 8)):
            s1 = good[i]
            s2 = good[j]
            combined_conds = s1['conditions'] + s2['conditions']
            seen_c = set()
            unique_c = []
            for c in combined_conds:
                key = (c[0], c[1])
                if key not in seen_c:
                    seen_c.add(key)
                    unique_c.append(c)

            combo_name = f"{s1.get('type_name', 'A')}+{s2.get('type_name', 'B')}"
            combo_strat = {
                'conditions': unique_c,
                'description': f"COMBO: {combo_name}",
                'positive_rate': min(s1['train_positive_rate'], s2['train_positive_rate']),
                'sample_count': min(s1['train_samples'], s2['train_samples']),
                'source': 'combo',
                'type_name': combo_name,
            }
            result = backtest_strategy(combo_strat, test_df, feature_cols, horizon)
            if result is not None and result['backtest']['win_rate'] >= 50:
                result['combo_from'] = [s1['strategy_id'], s2['strategy_id']]
                combos.append(result)

    combos.sort(key=lambda r: r['backtest']['sharpe'] * r['backtest']['win_rate'] / 100, reverse=True)
    log(f"[COMBO] {len(combos)} valid combos found")
    return combos[:10]


# ============================================================
# Report
# ============================================================
def print_report(singles, combos, info, horizon):
    log("\n" + "=" * 70)
    log("         AI 策略挖掘与回测分析报告")
    log("=" * 70)
    log(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"数据范围: 2023-02-08 ~ 2026-02-06 (约3年)")
    log(f"股票数量: {info['total_stocks']} 只")
    log(f"总样本量: {info['total_samples']:,}")
    log(f"训练集:   {info['train_samples']:,} | 测试集: {info['test_samples']:,}")
    log(f"特征维度: {info['feature_count']} 个技术指标")
    log(f"持有周期: {horizon} 天")
    log(f"GBM准确率: {info['gbm_accuracy']:.1%} | RF准确率: {info['rf_accuracy']:.1%}")

    log(f"\n--- Top 10 最重要特征 ---")
    for i, f in enumerate(info['top_features'][:10], 1):
        log(f"  {i:2d}. {f['cn_name']:15s}  重要度: {f['importance']:.4f}")

    log(f"\n{'=' * 70}")
    log(f"  一、单策略结果 (共 {len(singles)} 套)")
    log(f"{'=' * 70}")

    for i, s in enumerate(singles, 1):
        bt = s['backtest']
        log(f"\n--- 策略 #{i} [{s['grade']}] {s.get('type_name', 'AI')} ---")
        log(f"  规则: {s['description']}")
        log(f"  来源: {s['source']} | 训练正样本率: {s['train_positive_rate']:.1%}")
        log(f"  交易次数:   {bt['total_trades']:>6d} 次")
        log(f"  胜率:       {bt['win_rate']:>6.2f}%")
        log(f"  每笔收益:   {bt['avg_return_per_trade']:>+7.3f}%")
        log(f"  累计收益:   {bt['total_return']:>+8.2f}%")
        log(f"  年化收益:   {bt['annual_return']:>+8.2f}%")
        log(f"  盈亏比:     {bt['profit_loss_ratio']:>6.2f}")
        log(f"  最大回撤:   {bt['max_drawdown']:>+8.2f}%")
        log(f"  夏普比率:   {bt['sharpe']:>6.2f}")
        log(f"  平均盈利:   {bt['avg_win']:>+7.3f}% | 平均亏损: {bt['avg_lose']:>-7.3f}%")

    if combos:
        log(f"\n{'=' * 70}")
        log(f"  二、组合策略结果 (共 {len(combos)} 套)")
        log(f"{'=' * 70}")

        for i, s in enumerate(combos, 1):
            bt = s['backtest']
            log(f"\n--- 组合 #{i} [{s['grade']}] {s.get('type_name', 'COMBO')} ---")
            log(f"  规则: {s['description']}")
            log(f"  交易次数:   {bt['total_trades']:>6d} 次")
            log(f"  胜率:       {bt['win_rate']:>6.2f}%")
            log(f"  每笔收益:   {bt['avg_return_per_trade']:>+7.3f}%")
            log(f"  累计收益:   {bt['total_return']:>+8.2f}%")
            log(f"  年化收益:   {bt['annual_return']:>+8.2f}%")
            log(f"  盈亏比:     {bt['profit_loss_ratio']:>6.2f}")
            log(f"  最大回撤:   {bt['max_drawdown']:>+8.2f}%")
            log(f"  夏普比率:   {bt['sharpe']:>6.2f}")

    log(f"\n{'=' * 70}")
    log("  三、策略推荐总结")
    log(f"{'=' * 70}")

    all_s = singles + combos
    ap = [s for s in all_s if s['grade'] == 'A+']
    ag = [s for s in all_s if s['grade'] == 'A']
    bg = [s for s in all_s if s['grade'] == 'B']
    log(f"  A+级: {len(ap)} 套 | A级: {len(ag)} 套 | B级: {len(bg)} 套")

    best = ap[:3] if ap else ag[:3]
    if best:
        log(f"\n  >>> 最佳推荐:")
        for s in best:
            bt = s['backtest']
            log(f"      {s.get('type_name','')} | 胜率{bt['win_rate']:.1f}% | 年化{bt['annual_return']:+.1f}% | 夏普{bt['sharpe']:.2f} | 回撤{bt['max_drawdown']:.1f}%")

    log(f"\n{'=' * 70}")
    log("  注意: 以上基于历史回测，不代表未来收益。已扣除交易成本。")
    log(f"{'=' * 70}\n")


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()

    log("=" * 70)
    log("  AI Strategy Discovery Engine")
    log("  Mining investment strategies from 3-year market data")
    log("=" * 70)

    stock_data = load_all_stock_data()
    if not stock_data:
        log("[FAIL] No data loaded")
        return

    best_horizon = 5
    best_results = None
    best_info = None
    best_test_df = None
    best_fcols = None

    for horizon in HOLD_DAYS_LIST:
        log(f"\n{'#' * 70}")
        log(f"  === Horizon: {horizon} days ===")
        log(f"{'#' * 70}")

        combined, feature_cols = build_dataset(stock_data, horizon=horizon)
        if combined.empty or not feature_cols:
            log(f"  [SKIP] Not enough data for horizon={horizon}")
            continue

        strategies, ginfo = discover_strategies(combined, feature_cols, horizon)
        if not strategies:
            log(f"  [SKIP] No valid strategies for horizon={horizon}")
            continue

        top_sharpe = strategies[0]['backtest']['sharpe'] if strategies else 0
        if best_results is None or top_sharpe > (best_results[0]['backtest']['sharpe'] if best_results else 0):
            best_horizon = horizon
            best_results = strategies
            best_info = ginfo
            best_fcols = feature_cols

            valid = combined.dropna(subset=feature_cols + ['label', 'future_ret']).copy()
            valid = valid.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
            valid = valid.sort_values('date').reset_index(drop=True)
            split_idx = int(len(valid) * (1 - TEST_RATIO))
            best_test_df = valid.iloc[split_idx:]

    if best_results is None:
        log("[FAIL] No strategies found")
        return

    log(f"\n[BEST] Best horizon: {best_horizon} days")

    combos = build_combo_strategies(best_results, best_test_df, best_fcols, best_horizon)
    print_report(best_results, combos, best_info, best_horizon)

    report = {
        'generated_at': datetime.now().isoformat(),
        'horizon': best_horizon,
        'global_info': best_info,
        'single_strategies': best_results,
        'combo_strategies': combos,
    }

    def _conv(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_conv)

    elapsed = time.time() - t0
    log(f"\n[DONE] Total time: {elapsed:.1f}s")
    log(f"[SAVE] Report saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
