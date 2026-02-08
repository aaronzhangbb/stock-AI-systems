# -*- coding: utf-8 -*-
"""
数据驱动策略发现引擎
- 从历史数据中自动学习哪些特征组合能预测上涨
- 训练模型 → 提取规则 → 生成 "学到的策略"
- 每个策略附带回测指标（胜率、盈亏比、夏普等）
- 替代人工编写的固定规则

核心思路:
  1. 从全市场历史数据中采样，构建大量技术指标特征
  2. 用 GradientBoosting 学习哪些特征组合预测 5 日正收益
  3. 从模型中提取 top 特征和阈值规则
  4. 每条规则就是一个 "学到的策略"，附带回测验证
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import sys
import json
import pickle
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from src.strategy.ai_scoring import build_features

# 模型和策略缓存路径
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
MODEL_PATH = os.path.join(MODEL_DIR, 'learned_model.pkl')
RULES_PATH = os.path.join(MODEL_DIR, 'learned_rules.json')
CACHE_DB = os.path.join(MODEL_DIR, 'stock_cache.db')

# 特征列（技术指标）
FEATURE_COLS = [
    'ret1', 'ret5', 'vol_20', 'ma_diff', 'rsi14',
    'macd_dif', 'macd_dea', 'macd_bar', 'bb_pos',
    'momentum_10', 'vol_ratio',
]

# 特征的中文名
FEATURE_NAMES = {
    'ret1': '日涨幅',
    'ret5': '5日涨幅',
    'vol_20': '20日波动率',
    'ma_diff': 'MA5-MA20偏离',
    'rsi14': 'RSI(14)',
    'macd_dif': 'MACD DIF',
    'macd_dea': 'MACD DEA',
    'macd_bar': 'MACD柱',
    'bb_pos': '布林带位置',
    'momentum_10': '10日动量',
    'vol_ratio': '量比',
}


def _load_sample_data(max_stocks: int = 300, min_rows: int = 200) -> pd.DataFrame:
    """
    从本地缓存中采样股票数据，构建训练集

    只选取数据量充足（>=min_rows行）的股票，最多 max_stocks 只
    """
    if not os.path.exists(CACHE_DB):
        print("[策略发现] 缓存数据库不存在")
        return pd.DataFrame()

    conn = sqlite3.connect(CACHE_DB)

    # 找出数据量充足的股票
    meta = pd.read_sql_query(
        f"""SELECT m.stock_code, COUNT(k.date) as cnt
            FROM cache_meta m
            JOIN daily_kline k ON m.stock_code = k.stock_code
            GROUP BY m.stock_code
            HAVING cnt >= {min_rows}
            ORDER BY RANDOM()
            LIMIT {max_stocks}""",
        conn
    )

    if meta.empty:
        conn.close()
        print(f"[策略发现] 没有足够数据（需要至少 {min_rows} 行）的股票")
        return pd.DataFrame()

    codes = meta['stock_code'].tolist()
    print(f"[策略发现] 选取 {len(codes)} 只股票用于训练")

    all_data = []
    for code in codes:
        df = pd.read_sql_query(
            'SELECT date, open, high, low, close, volume FROM daily_kline '
            'WHERE stock_code = ? ORDER BY date',
            conn, params=[code]
        )
        if df.empty or len(df) < min_rows:
            continue
        df['date'] = pd.to_datetime(df['date'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['stock_code'] = code
        all_data.append(df)

    conn.close()

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    print(f"[策略发现] 共加载 {len(combined)} 条记录")
    return combined


def _build_training_set(combined: pd.DataFrame, horizon: int = 5) -> tuple:
    """
    从原始数据构建训练集

    对每只股票：
    1. 计算技术指标特征
    2. 标注未来 horizon 日涨跌
    3. 合并为大型训练集
    """
    all_X = []
    all_y = []

    stocks = combined['stock_code'].unique()
    for code in stocks:
        df = combined[combined['stock_code'] == code].copy()
        if len(df) < 60:
            continue

        data = build_features(df)
        data['future_ret'] = data['close'].shift(-horizon) / data['close'] - 1
        data['label'] = (data['future_ret'] > 0.02).astype(int)  # 涨2%以上为正样本

        data = data.dropna(subset=FEATURE_COLS + ['label'])
        if len(data) < 30:
            continue

        all_X.append(data[FEATURE_COLS].values)
        all_y.append(data['label'].values)

    if not all_X:
        return np.array([]), np.array([])

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    print(f"[策略发现] 训练集: {len(X)} 样本, 正样本比例: {y.mean():.1%}")
    return X, y


def train_model(max_stocks: int = 300, force: bool = False) -> dict:
    """
    训练策略发现模型

    参数:
        max_stocks: 最多从多少只股票中采样
        force: 强制重新训练（忽略缓存）

    返回:
        dict: 训练结果 {accuracy, feature_importance, rules, model}
    """
    # 检查已有模型
    if not force and os.path.exists(RULES_PATH):
        rules = load_learned_rules()
        if rules:
            age_hours = (datetime.now() - datetime.fromisoformat(
                rules.get('trained_at', '2000-01-01')
            )).total_seconds() / 3600
            if age_hours < 24:
                print(f"[策略发现] 使用缓存模型（{age_hours:.1f}小时前训练）")
                return rules

    print("[策略发现] 开始从历史数据中学习策略...")

    # 1. 加载数据
    combined = _load_sample_data(max_stocks=max_stocks)
    if combined.empty:
        return {'error': '数据不足'}

    # 2. 构建训练集
    X, y = _build_training_set(combined)
    if len(X) < 100:
        return {'error': '样本不足'}

    # 3. 训练模型
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.tree import DecisionTreeClassifier

        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            min_samples_leaf=50, subsample=0.8, random_state=42
        )
        model.fit(X, y)

        # 交叉验证
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        accuracy = float(scores.mean())
        print(f"[策略发现] 模型准确率: {accuracy:.1%} (±{scores.std():.1%})")

        # 4. 提取特征重要度
        importances = model.feature_importances_
        feat_imp = {
            FEATURE_COLS[i]: {
                'importance': round(float(importances[i]), 4),
                'name': FEATURE_NAMES.get(FEATURE_COLS[i], FEATURE_COLS[i]),
                'rank': 0,
            }
            for i in range(len(FEATURE_COLS))
        }
        # 排名
        sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1]['importance'], reverse=True)
        for rank, (k, v) in enumerate(sorted_feats, 1):
            feat_imp[k]['rank'] = rank

        # 5. 提取决策规则（从单棵浅决策树中提取可解释规则）
        rules = _extract_rules(X, y)

        # 6. 对每条规则做回测验证
        rules_with_backtest = _backtest_rules(rules, combined)

        # 7. 保存
        result = {
            'trained_at': datetime.now().isoformat(),
            'sample_stocks': int(combined['stock_code'].nunique()),
            'total_samples': int(len(X)),
            'accuracy': round(accuracy, 4),
            'feature_importance': feat_imp,
            'top_features': [
                {'feature': k, 'name': v['name'], 'importance': v['importance']}
                for k, v in sorted_feats[:5]
            ],
            'learned_rules': rules_with_backtest,
        }

        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(RULES_PATH, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 保存模型
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)

        print(f"[策略发现] 发现 {len(rules_with_backtest)} 条数据驱动策略")
        return result

    except ImportError as e:
        return {'error': f'缺少依赖: {e}'}
    except Exception as e:
        return {'error': f'训练失败: {e}'}


def _extract_rules(X: np.ndarray, y: np.ndarray, max_rules: int = 8) -> list:
    """
    从数据中提取可解释的决策规则

    用多棵浅决策树提取不同的规则组合
    """
    from sklearn.tree import DecisionTreeClassifier

    rules = []
    seen = set()

    for depth in [2, 3]:
        for seed in range(20):
            tree = DecisionTreeClassifier(
                max_depth=depth, min_samples_leaf=max(50, len(X) // 50),
                random_state=seed
            )
            tree.fit(X, y)

            # 提取从根到叶子的路径规则
            extracted = _tree_to_rules(tree, FEATURE_COLS)
            for rule in extracted:
                # 去重
                key = rule['description']
                if key in seen:
                    continue
                seen.add(key)
                rules.append(rule)

            if len(rules) >= max_rules * 3:
                break

    # 按预测正收益概率排序，取 top
    rules.sort(key=lambda r: r.get('positive_rate', 0), reverse=True)
    return rules[:max_rules]


def _tree_to_rules(tree, feature_names: list) -> list:
    """从决策树中提取路径规则"""
    tree_ = tree.tree_
    rules = []

    def _recurse(node, conditions):
        if tree_.feature[node] == -2:  # 叶子节点
            total = tree_.n_node_samples[node]
            positive = tree_.value[node][0][1] if tree_.value[node].shape[1] > 1 else 0
            negative = tree_.value[node][0][0]
            rate = positive / total if total > 0 else 0

            if rate >= 0.55 and total >= 30:  # 只保留正概率 >= 55% 的规则
                conditions_cn = []
                for feat, op, thresh in conditions:
                    fname = FEATURE_NAMES.get(feat, feat)
                    conditions_cn.append(f"{fname} {op} {thresh:.3f}")

                rules.append({
                    'conditions': conditions.copy(),
                    'description': ' 且 '.join(conditions_cn),
                    'positive_rate': round(rate, 3),
                    'sample_count': int(total),
                    'positive_count': int(positive),
                })
            return

        feat = feature_names[tree_.feature[node]]
        threshold = tree_.threshold[node]

        # 左子树: feat <= threshold
        _recurse(tree_.children_left[node], conditions + [(feat, '<=', threshold)])
        # 右子树: feat > threshold
        _recurse(tree_.children_right[node], conditions + [(feat, '>', threshold)])

    _recurse(0, [])
    return rules


def _backtest_rules(rules: list, combined: pd.DataFrame, horizon: int = 5) -> list:
    """
    对每条学到的规则做历史回测

    在全部样本数据上，模拟每条规则的触发 → 持有 horizon 天 → 统计收益
    """
    results = []

    for i, rule in enumerate(rules):
        trades = []
        stocks = combined['stock_code'].unique()

        for code in stocks:
            df = combined[combined['stock_code'] == code].copy()
            if len(df) < 60:
                continue

            data = build_features(df)
            data = data.dropna(subset=FEATURE_COLS).reset_index(drop=True)

            for idx in range(len(data) - horizon):
                row = data.iloc[idx]
                # 检查是否满足所有条件
                match = True
                for feat, op, thresh in rule['conditions']:
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
                if not match:
                    continue

                buy_price = float(data.iloc[idx]['close'])
                sell_price = float(data.iloc[idx + horizon]['close'])
                ret = (sell_price - buy_price) / buy_price
                trades.append(ret)

        if len(trades) < 10:
            continue

        returns = np.array(trades)
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        win_rate = len(wins) / len(returns) * 100
        avg_win = float(wins.mean()) * 100 if len(wins) > 0 else 0
        avg_lose = float(abs(losses.mean())) * 100 if len(losses) > 0 else 0.01
        profit_loss_ratio = avg_win / avg_lose if avg_lose > 0 else 999

        equity = (1 + returns).cumprod()
        peak = np.maximum.accumulate(equity)
        max_dd = float(((equity - peak) / peak).min()) * 100

        sharpe = 0
        if returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * np.sqrt(252 / horizon))

        avg_return = float(returns.mean()) * 100

        rule_result = {
            'rule_id': f"learned_{i+1}",
            'description': rule['description'],
            'conditions': rule['conditions'],
            'positive_rate': rule['positive_rate'],
            'backtest': {
                'total_trades': len(trades),
                'win_rate': round(win_rate, 1),
                'avg_return': round(avg_return, 2),
                'profit_loss_ratio': round(profit_loss_ratio, 2),
                'max_drawdown': round(max_dd, 2),
                'sharpe': round(sharpe, 2),
                'avg_win': round(avg_win, 2),
                'avg_lose': round(avg_lose, 2),
            },
        }

        # 评级
        if win_rate >= 55 and sharpe >= 0.5 and profit_loss_ratio >= 1.2:
            rule_result['grade'] = 'A'
        elif win_rate >= 48 and sharpe >= 0.2:
            rule_result['grade'] = 'B'
        else:
            rule_result['grade'] = 'C'

        results.append(rule_result)

    # 按胜率 × 夏普排序
    results.sort(key=lambda r: r['backtest']['win_rate'] * max(r['backtest']['sharpe'], 0.1), reverse=True)
    return results


def load_learned_rules() -> dict:
    """加载已学习的策略规则"""
    if not os.path.exists(RULES_PATH):
        return {}
    try:
        with open(RULES_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def apply_learned_rules(df: pd.DataFrame) -> list:
    """
    对单只股票应用学习到的策略规则

    参数:
        df: 股票历史 OHLCV DataFrame

    返回:
        list[dict]: 触发的规则信号列表
    """
    rules_data = load_learned_rules()
    if not rules_data or 'learned_rules' not in rules_data:
        return []

    if df.empty or len(df) < 30:
        return []

    data = build_features(df)
    data = data.dropna(subset=FEATURE_COLS)
    if data.empty:
        return []

    last_row = data.iloc[-1]
    triggered = []

    for rule in rules_data['learned_rules']:
        match = True
        for feat, op, thresh in rule['conditions']:
            val = last_row.get(feat)
            if pd.isna(val):
                match = False
                break
            if op == '<=' and val > thresh:
                match = False
                break
            if op == '>' and val <= thresh:
                match = False
                break

        if match:
            bt = rule.get('backtest', {})
            triggered.append({
                'signal': 'buy',
                'strategy': f"ML策略: {rule['description']}",
                'strategy_id': rule['rule_id'],
                'strength': min(95, int(bt.get('win_rate', 50) + bt.get('sharpe', 0) * 10)),
                'reason': (
                    f"胜率{bt.get('win_rate', 0):.0f}% "
                    f"盈亏比{bt.get('profit_loss_ratio', 0):.1f} "
                    f"夏普{bt.get('sharpe', 0):.2f} "
                    f"[{rule.get('grade', 'C')}级]"
                ),
                'grade': rule.get('grade', 'C'),
                'indicators': {
                    'win_rate': bt.get('win_rate', 0),
                    'sharpe': bt.get('sharpe', 0),
                    'max_drawdown': bt.get('max_drawdown', 0),
                },
            })

    triggered.sort(key=lambda x: x['strength'], reverse=True)
    return triggered


def get_discovery_summary() -> dict:
    """获取策略发现摘要（用于UI展示）"""
    rules_data = load_learned_rules()
    if not rules_data:
        return {'status': 'not_trained', 'message': '尚未训练，请先运行策略发现'}

    learned = rules_data.get('learned_rules', [])
    grade_a = [r for r in learned if r.get('grade') == 'A']
    grade_b = [r for r in learned if r.get('grade') == 'B']
    grade_c = [r for r in learned if r.get('grade') == 'C']

    return {
        'status': 'ready',
        'trained_at': rules_data.get('trained_at', ''),
        'sample_stocks': rules_data.get('sample_stocks', 0),
        'total_samples': rules_data.get('total_samples', 0),
        'accuracy': rules_data.get('accuracy', 0),
        'total_rules': len(learned),
        'grade_a': len(grade_a),
        'grade_b': len(grade_b),
        'grade_c': len(grade_c),
        'top_features': rules_data.get('top_features', []),
        'rules': learned,
    }
