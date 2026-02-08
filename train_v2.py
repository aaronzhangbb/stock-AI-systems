# -*- coding: utf-8 -*-
"""
AI策略V2.0 第一阶段: XGBoost GPU 训练 + 特征重要性分析

目标:
1. 用5008只可交易A股全部历史数据训练XGBoost
2. 让AI自己发现哪些特征最能预测股票上涨
3. 在验证集/测试集上评估效果
4. 输出SHAP分析结果
"""
import sys
import os
import time
import json
import pickle

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, 
    classification_report, confusion_matrix
)

from src.data.data_cache import DataCache
from src.data.stock_pool import StockPool
from src.strategy.ai_engine_v2 import (
    compute_v2_features, create_labels, build_dataset,
    get_feature_columns, split_by_time
)

print("=" * 80)
print("  AI策略V2.0 — XGBoost GPU 训练引擎")
print("  RTX 5080 16GB | 5008只可交易A股 | 100+高阶特征")
print("=" * 80)

# ============================================================
# 1. 构建数据集
# ============================================================
print("\n[阶段1] 构建全市场特征数据集...")
t0 = time.time()

cache = DataCache()
pool = StockPool()

def progress(current, total):
    pct = current / total * 100
    print(f"  特征计算: {current}/{total} ({pct:.0f}%)", flush=True)

X, y, meta = build_dataset(
    cache, pool,
    future_days=5,          # 预测未来5天
    target_return=0.03,     # 涨幅>3%
    min_bars=100,
    progress_callback=progress,
)

elapsed_build = time.time() - t0
print(f"\n  数据集构建完成! 耗时: {elapsed_build:.0f}秒")
print(f"  总样本数: {len(X):,}")
print(f"  特征数: {X.shape[1]}")
print(f"  正样本(涨>3%): {y.sum():,.0f} ({y.mean()*100:.1f}%)")
print(f"  负样本(不涨): {len(y)-y.sum():,.0f} ({(1-y.mean())*100:.1f}%)")
print(f"  股票数: {meta['stock_code'].nunique()}")
print(f"  日期范围: {meta['date'].min()} ~ {meta['date'].max()}")

# ============================================================
# 2. 时间切分
# ============================================================
print("\n[阶段2] 按时间切分数据集...")
splits = split_by_time(X, y, meta, 
                       train_end='2025-06-30', 
                       val_end='2025-12-31')

print(f"  训练集: {len(splits['X_train']):,} 样本 "
      f"(~2023.02~2025.06, 正样本率{splits['y_train'].mean()*100:.1f}%)")
print(f"  验证集: {len(splits['X_val']):,} 样本 "
      f"(2025.07~2025.12, 正样本率{splits['y_val'].mean()*100:.1f}%)")
print(f"  测试集: {len(splits['X_test']):,} 样本 "
      f"(2026.01~2026.02, 正样本率{splits['y_test'].mean()*100:.1f}%)")

# ============================================================
# 3. 处理缺失值和无穷值
# ============================================================
print("\n[阶段3] 数据清洗...")

feat_cols = get_feature_columns(X)

for key in ['X_train', 'X_val', 'X_test']:
    df = splits[key]
    # 替换无穷值
    df = df.replace([np.inf, -np.inf], np.nan)
    splits[key] = df

# 统计缺失情况
nan_pct = splits['X_train'].isna().mean()
high_nan = nan_pct[nan_pct > 0.3].sort_values(ascending=False)
if len(high_nan) > 0:
    print(f"  高缺失率特征 (>30%): {len(high_nan)} 个")
    for col, pct in high_nan.head(10).items():
        print(f"    {col}: {pct*100:.1f}%")
    
    # 移除高缺失率特征
    drop_cols = high_nan.index.tolist()
    for key in ['X_train', 'X_val', 'X_test']:
        splits[key] = splits[key].drop(columns=drop_cols, errors='ignore')
    print(f"  已移除 {len(drop_cols)} 个高缺失特征")

remaining_feat = splits['X_train'].shape[1]
print(f"  剩余特征数: {remaining_feat}")

# ============================================================
# 4. XGBoost GPU 训练
# ============================================================
print("\n[阶段4] XGBoost GPU 训练...")
print("  使用 RTX 5080 GPU 加速...")

t_train = time.time()

# 构建DMatrix (XGBoost专用数据格式)
dtrain = xgb.DMatrix(splits['X_train'], label=splits['y_train'], 
                      enable_categorical=False)
dval = xgb.DMatrix(splits['X_val'], label=splits['y_val'],
                    enable_categorical=False)
dtest = xgb.DMatrix(splits['X_test'], label=splits['y_test'],
                     enable_categorical=False)

# 计算正负样本比例 (处理类别不平衡)
pos_count = splits['y_train'].sum()
neg_count = len(splits['y_train']) - pos_count
scale_pos_weight = neg_count / (pos_count + 1)

params = {
    'device': 'cuda',                    # GPU加速
    'tree_method': 'hist',               # 直方图方法
    'objective': 'binary:logistic',      # 二分类
    'eval_metric': ['auc', 'logloss'],   # 评估指标
    'max_depth': 8,                      # 树深度
    'learning_rate': 0.05,               # 学习率
    'subsample': 0.8,                    # 行采样
    'colsample_bytree': 0.8,            # 列采样
    'min_child_weight': 50,              # 最小叶节点样本
    'gamma': 0.1,                        # 剪枝阈值
    'reg_alpha': 0.1,                    # L1正则
    'reg_lambda': 1.0,                   # L2正则
    'scale_pos_weight': scale_pos_weight, # 类别平衡
    'verbosity': 0,
    'seed': 42,
}

print(f"  参数: max_depth={params['max_depth']}, lr={params['learning_rate']}, "
      f"scale_pos_weight={scale_pos_weight:.2f}")

# 训练 (带早停)
evals = [(dtrain, 'train'), (dval, 'val')]
evals_result = {}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,               # 最多1000棵树
    evals=evals,
    evals_result=evals_result,
    early_stopping_rounds=50,            # 50轮无改善则停止
    verbose_eval=100,                    # 每100轮打印
)

train_time = time.time() - t_train
best_round = model.best_iteration
print(f"\n  训练完成! 耗时: {train_time:.1f}秒")
print(f"  最佳迭代: {best_round} 轮")
print(f"  训练AUC: {evals_result['train']['auc'][best_round]:.4f}")
print(f"  验证AUC: {evals_result['val']['auc'][best_round]:.4f}")

# ============================================================
# 5. 模型评估
# ============================================================
print("\n[阶段5] 模型评估...")

def evaluate(model, dmatrix, y_true, set_name):
    """评估模型表现"""
    y_prob = model.predict(dmatrix)
    y_pred = (y_prob >= 0.5).astype(int)
    
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    # Top K 精度 (最核心指标: 排名前100的股票中有多少真正上涨)
    top_indices = np.argsort(y_prob)[::-1]
    for k in [50, 100, 200, 500]:
        if k <= len(y_prob):
            top_k_true = y_true.iloc[top_indices[:k]].mean()
            print(f"    Precision@{k}: {top_k_true*100:.1f}% "
                  f"(Top{k}中实际上涨比例)")
    
    # 分位对比
    q90 = np.percentile(y_prob, 90)
    q10 = np.percentile(y_prob, 10)
    high_conf = y_true.iloc[y_prob >= q90].mean()
    low_conf = y_true.iloc[y_prob <= q10].mean()
    
    print(f"  [{set_name}] AUC={auc:.4f} Precision={prec:.3f} Recall={rec:.3f}")
    print(f"    高置信(Top10%): 实际上涨率 {high_conf*100:.1f}%")
    print(f"    低置信(Bottom10%): 实际上涨率 {low_conf*100:.1f}%")
    print(f"    区分度: {(high_conf - low_conf)*100:.1f}% 差异")
    
    return {
        'auc': auc, 'precision': prec, 'recall': rec,
        'top10_rate': high_conf, 'bottom10_rate': low_conf,
    }

print("\n  --- 验证集 (2025.07~2025.12) ---")
val_metrics = evaluate(model, dval, splits['y_val'], '验证集')

print("\n  --- 测试集 (2026.01~2026.02) ---")
test_metrics = evaluate(model, dtest, splits['y_test'], '测试集')

# ============================================================
# 6. 特征重要性分析
# ============================================================
print("\n[阶段6] 特征重要性分析...")

# XGBoost内置重要性
importance = model.get_score(importance_type='gain')
imp_df = pd.DataFrame([
    {'feature': k, 'importance': v} 
    for k, v in importance.items()
]).sort_values('importance', ascending=False)

print("\n  ===== AI认为最重要的Top 30特征 =====")
print(f"  {'排名':<4} {'特征名':<28} {'重要性':>10} {'类别':<15}")
print("  " + "-" * 60)

# 特征分类
def classify_feature(name):
    if 'ret_' in name or 'momentum' in name:
        return '动量'
    elif 'ma' in name and ('diff' in name or 'slope' in name or 'cross' in name or 'alignment' in name):
        return '均线'
    elif 'rsi' in name or 'stoch' in name:
        return 'RSI'
    elif 'bb_' in name:
        return '布林带'
    elif 'vol_' in name or 'volatility' in name or 'atr' in name:
        return '波动率/量'
    elif 'obv' in name or 'mfi' in name or 'vwap' in name:
        return '资金流'
    elif 'macd' in name:
        return 'MACD'
    elif 'kdj' in name:
        return 'KDJ'
    elif 'body' in name or 'shadow' in name or 'doji' in name:
        return 'K线形态'
    elif 'dist_' in name or 'price_pos' in name:
        return '价格位置'
    elif 'consec' in name or 'up_ratio' in name:
        return '涨跌统计'
    elif 'skew' in name or 'kurt' in name or 'efficiency' in name:
        return '统计特征'
    elif 'month' in name or 'weekday' in name or 'quarter' in name:
        return '时间'
    elif 'cci' in name or 'williams' in name:
        return '其他指标'
    return '其他'

for i, row in imp_df.head(30).iterrows():
    rank = imp_df.index.get_loc(i) + 1
    cat = classify_feature(row['feature'])
    print(f"  {rank:<4} {row['feature']:<28} {row['importance']:>10.1f} {cat}")

# 统计各类特征贡献
print("\n  ===== 各类别特征贡献度 =====")
imp_df['category'] = imp_df['feature'].apply(classify_feature)
cat_imp = imp_df.groupby('category')['importance'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
total_imp = cat_imp['sum'].sum()
for cat, row in cat_imp.iterrows():
    pct = row['sum'] / total_imp * 100
    print(f"  {cat:<15} 贡献{pct:>5.1f}% ({row['count']:.0f}个特征, 平均{row['mean']:.0f})")

# ============================================================
# 7. SHAP分析 (深度特征解释)
# ============================================================
print("\n[阶段7] SHAP深度分析...")
try:
    import shap
    
    # 用验证集的一个子集做SHAP (太大会很慢)
    shap_sample_size = min(5000, len(splits['X_val']))
    X_shap = splits['X_val'].iloc[:shap_sample_size].copy()
    X_shap = X_shap.replace([np.inf, -np.inf], np.nan)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    
    # SHAP特征重要性
    shap_imp = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        'feature': X_shap.columns,
        'shap_importance': shap_imp
    }).sort_values('shap_importance', ascending=False)
    
    print("\n  ===== SHAP分析: AI决策依据Top 20 =====")
    print(f"  {'排名':<4} {'特征名':<28} {'SHAP重要性':>12}")
    print("  " + "-" * 50)
    for i, row in shap_df.head(20).iterrows():
        rank = shap_df.index.get_loc(i) + 1
        print(f"  {rank:<4} {row['feature']:<28} {row['shap_importance']:>12.6f}")
    
    shap_saved = True
except Exception as e:
    print(f"  SHAP分析跳过: {e}")
    shap_saved = False
    shap_df = None

# ============================================================
# 8. 保存结果
# ============================================================
print("\n[阶段8] 保存模型和结果...")

# 保存模型
model_path = os.path.join('data', 'xgb_v2_model.json')
model.save_model(model_path)
print(f"  模型已保存: {model_path}")

# 保存特征重要性
result = {
    'train_info': {
        'total_samples': len(X),
        'total_features': X.shape[1],
        'remaining_features': remaining_feat,
        'total_stocks': meta['stock_code'].nunique(),
        'date_range': f"{meta['date'].min()} ~ {meta['date'].max()}",
        'future_days': 5,
        'target_return': '3%',
        'train_time_seconds': round(train_time, 1),
        'best_iteration': best_round,
        'gpu': 'RTX 5080 16GB',
    },
    'metrics': {
        'train_auc': round(evals_result['train']['auc'][best_round], 4),
        'val_auc': round(evals_result['val']['auc'][best_round], 4),
        'val_metrics': {k: round(float(v), 4) for k, v in val_metrics.items()},
        'test_metrics': {k: round(float(v), 4) for k, v in test_metrics.items()},
    },
    'feature_importance_top50': [
        {'rank': i+1, 'feature': row['feature'], 
         'importance': round(row['importance'], 2),
         'category': classify_feature(row['feature'])}
        for i, (_, row) in enumerate(imp_df.head(50).iterrows())
    ],
    'category_contribution': {
        cat: {'pct': round(row['sum']/total_imp*100, 1), 
              'count': int(row['count']),
              'avg': round(row['mean'], 1)}
        for cat, row in cat_imp.iterrows()
    },
}

if shap_df is not None:
    result['shap_top20'] = [
        {'rank': i+1, 'feature': row['feature'],
         'shap_importance': round(float(row['shap_importance']), 6)}
        for i, (_, row) in enumerate(shap_df.head(20).iterrows())
    ]

result_path = os.path.join('data', 'xgb_v2_result.json')
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2, default=str)
print(f"  结果已保存: {result_path}")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 80)
print("  AI策略V2.0 第一阶段训练完成!")
print("=" * 80)
print(f"\n  ★ 模型在验证集AUC: {val_metrics['auc']:.4f}")
print(f"  ★ 模型在测试集AUC: {test_metrics['auc']:.4f}")
print(f"  ★ Top10%高置信预测的实际上涨率: {test_metrics['top10_rate']*100:.1f}%")
print(f"  ★ Bottom10%低置信的实际上涨率: {test_metrics['bottom10_rate']*100:.1f}%")
print(f"  ★ 区分能力: {(test_metrics['top10_rate']-test_metrics['bottom10_rate'])*100:.1f}%")
print(f"\n  AI认为最重要的3个特征:")
for _, row in imp_df.head(3).iterrows():
    cat = classify_feature(row['feature'])
    print(f"    → {row['feature']} ({cat})")
print(f"\n  训练耗时: {train_time:.1f}秒 (GPU加速)")
print(f"  总耗时: {time.time()-t0:.0f}秒")
