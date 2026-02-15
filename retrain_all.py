# -*- coding: utf-8 -*-
"""
AI超级策略 — 三层模型一键重训脚本
==================================
当积累了新数据后，运行此脚本重新训练所有三层模型。

使用方法:
    python retrain_all.py              # 重训全部三层
    python retrain_all.py --layer 1    # 只重训第一层 XGBoost
    python retrain_all.py --layer 2    # 只重训第二层 形态聚类
    python retrain_all.py --layer 3    # 只重训第三层 Transformer
    python retrain_all.py --layer 1,2  # 重训第一和第二层

建议频率:
    - 每1~2周重训一次 (纳入最新数据)
    - 市场发生剧变时立即重训
    - 每日只需运行前端扫描按钮即可 (推理不需要重训)

耗时估算 (RTX 5080 GPU):
    第一层 XGBoost:     ~5分钟 (特征计算3min + 训练2min)
    第二层 形态聚类:     ~10分钟 (向量提取9min + 聚类5s)
    第三层 Transformer: ~25分钟 (数据构建1min + GPU训练23min)
    总计:               ~40分钟
"""
import sys
import os
import time
import json
import argparse
import shutil
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def backup_model(filepath, label):
    """备份旧模型"""
    if os.path.exists(filepath):
        ext = os.path.splitext(filepath)[1]
        backup = filepath.replace(ext, f'_backup_{datetime.now().strftime("%Y%m%d")}{ext}')
        shutil.copy2(filepath, backup)
        print(f"  已备份: {os.path.basename(filepath)} → {os.path.basename(backup)}")


# ============================================================
# 第一层: XGBoost GPU
# ============================================================
def train_layer1():
    """重训 XGBoost 模型"""
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    from src.data.data_cache import DataCache
    from src.data.stock_pool import StockPool
    from src.strategy.ai_engine_v2 import (
        build_dataset, get_feature_columns, split_by_time
    )

    print("\n" + "=" * 70)
    print("  第一层: XGBoost GPU 重训")
    print("=" * 70)
    t0 = time.time()

    cache = DataCache()
    pool = StockPool()

    # 1. 构建数据集
    print("\n  [1/4] 构建全市场特征数据集...")
    def progress(c, t):
        if c % 500 == 0:
            print(f"    特征计算: {c}/{t} ({c/t*100:.0f}%)")

    X, y, meta = build_dataset(cache, pool, future_days=5, target_return=0.03,
                                min_bars=100, progress_callback=progress)
    print(f"    样本数: {len(X):,}, 特征数: {X.shape[1]}, 正样本率: {y.mean()*100:.1f}%")

    # 2. 时间切分
    print("\n  [2/4] 时间切分...")
    splits = split_by_time(X, y, meta, train_end='2025-06-30', val_end='2025-12-31')
    print(f"    训练: {len(splits['X_train']):,}  验证: {len(splits['X_val']):,}  测试: {len(splits['X_test']):,}")

    # 3. 数据清洗
    print("\n  [3/4] 数据清洗...")
    for key in ['X_train', 'X_val', 'X_test']:
        splits[key] = splits[key].replace([np.inf, -np.inf], np.nan)
    
    nan_pct = splits['X_train'].isna().mean()
    high_nan = nan_pct[nan_pct > 0.3].sort_values(ascending=False)
    if len(high_nan) > 0:
        drop_cols = high_nan.index.tolist()
        for key in ['X_train', 'X_val', 'X_test']:
            splits[key] = splits[key].drop(columns=drop_cols, errors='ignore')
        print(f"    移除高缺失特征: {len(drop_cols)} 个")

    # 尝试加载精选特征列表
    selected_features_path = os.path.join(DATA_DIR, 'v3_selected_features.json')
    if os.path.exists(selected_features_path):
        with open(selected_features_path, 'r') as f:
            raw = json.load(f)
        # 兼容两种格式: list 或 {"features": list}
        selected_features = raw if isinstance(raw, list) else raw.get('features', [])
        # 始终保留外部特征 (f_ 开头)
        extra_feat_cols = [c for c in splits['X_train'].columns if c.startswith('f_')]
        available = [f for f in selected_features if f in splits['X_train'].columns]
        # 合并精选特征 + 外部特征 (去重)
        all_selected = list(dict.fromkeys(available + extra_feat_cols))
        if len(all_selected) > 50:
            for key in ['X_train', 'X_val', 'X_test']:
                splits[key] = splits[key][all_selected]
            print(f"    使用精选特征: {len(available)} 个技术面 + {len(extra_feat_cols)} 个外部特征 = {len(all_selected)} 个")

    print(f"    最终特征数: {splits['X_train'].shape[1]}")

    # 4. XGBoost训练
    print("\n  [4/4] XGBoost GPU 训练...")
    dtrain = xgb.DMatrix(splits['X_train'], label=splits['y_train'])
    dval = xgb.DMatrix(splits['X_val'], label=splits['y_val'])
    dtest = xgb.DMatrix(splits['X_test'], label=splits['y_test'])

    pos_count = splits['y_train'].sum()
    neg_count = len(splits['y_train']) - pos_count
    scale_pos_weight = neg_count / (pos_count + 1)

    params = {
        'device': 'cuda', 'tree_method': 'hist',
        'objective': 'binary:logistic', 'eval_metric': ['auc', 'logloss'],
        'max_depth': 8, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'min_child_weight': 50, 'gamma': 0.1,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'scale_pos_weight': scale_pos_weight,
        'verbosity': 0, 'seed': 42,
    }

    evals_result = {}
    model = xgb.train(
        params, dtrain, num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        evals_result=evals_result,
        early_stopping_rounds=50, verbose_eval=100,
    )

    best_round = model.best_iteration
    val_auc = evals_result['val']['auc'][best_round]

    # 测试集评估
    test_probs = model.predict(dtest)
    test_labels = splits['y_test'].values
    test_auc = roc_auc_score(test_labels, test_probs)

    top50_idx = np.argsort(test_probs)[-50:]
    p_at_50 = test_labels[top50_idx].mean()

    top10_idx = np.argsort(test_probs)[-int(len(test_probs) * 0.1):]
    bot10_idx = np.argsort(test_probs)[:int(len(test_probs) * 0.1)]
    discrimination = test_labels[top10_idx].mean() - test_labels[bot10_idx].mean()

    # 保存模型
    model_path = os.path.join(DATA_DIR, 'xgb_v2_model.json')
    backup_model(model_path, 'xgb')
    model.save_model(model_path)

    elapsed = time.time() - t0
    result = {
        'val_auc': round(val_auc, 4),
        'test_auc': round(test_auc, 4),
        'test_p@50': round(float(p_at_50), 4),
        'discrimination': round(discrimination, 4),
        'best_round': best_round,
        'features': splits['X_train'].shape[1],
        'train_samples': len(splits['X_train']),
        'elapsed': round(elapsed, 0),
    }

    print(f"\n  ✅ 第一层训练完成 ({elapsed:.0f}秒)")
    print(f"     验证AUC: {val_auc:.4f}  测试AUC: {test_auc:.4f}")
    print(f"     P@50: {p_at_50:.3f}  区分度: {discrimination:.4f}")

    return result


# ============================================================
# 第二层: 形态聚类
# ============================================================
def train_layer2():
    """重训形态聚类模型"""
    from src.data.data_cache import DataCache
    from src.data.stock_pool import StockPool
    from src.strategy.pattern_engine import PatternEngine

    print("\n" + "=" * 70)
    print("  第二层: 形态聚类引擎 重训")
    print("=" * 70)
    t0 = time.time()

    cache = DataCache()
    pool = StockPool()
    engine = PatternEngine(n_clusters=200, window=20)

    # 1. 提取形态向量
    print("\n  [1/3] 提取形态向量...")
    def progress(c, t):
        if c % 1000 == 0:
            print(f"    提取: {c}/{t} ({c/t*100:.0f}%)")

    vectors, labels, meta = engine.build_training_data(cache, pool, progress)
    print(f"    形态数: {len(vectors):,}, 维度: {vectors.shape[1]}, 正样本率: {labels.mean()*100:.1f}%")

    # 2. 聚类训练
    print("\n  [2/3] K-Means 聚类训练...")
    report = engine.train(vectors, labels, meta)
    print(f"    有效聚类: {report['valid_clusters']}")
    print(f"    胜率>60%: {report['high_winrate_clusters']}种")
    print(f"    胜率>70%: {report['very_high_winrate_clusters']}种")

    # 3. 保存模型
    print("\n  [3/3] 保存模型...")
    model_path = os.path.join(DATA_DIR, 'pattern_engine.pkl')
    backup_model(model_path, 'pattern')
    engine.save(model_path)

    elapsed = time.time() - t0
    result = {
        'total_patterns': len(vectors),
        'valid_clusters': report['valid_clusters'],
        'high_wr_clusters': report['high_winrate_clusters'],
        'very_high_wr_clusters': report['very_high_winrate_clusters'],
        'avg_win_rate': report.get('avg_win_rate', 0),
        'max_win_rate': report.get('max_win_rate', 0),
        'high_wr_val_avg': report.get('high_wr_val_avg', 0),
        'elapsed': round(elapsed, 0),
    }

    print(f"\n  ✅ 第二层训练完成 ({elapsed:.0f}秒)")
    print(f"     {report['valid_clusters']}种有效形态, {report['high_winrate_clusters']}种高胜率")

    return result


# ============================================================
# 第三层: Transformer
# ============================================================
def train_layer3():
    """重训 Transformer 模型"""
    from src.data.data_cache import DataCache
    from src.data.stock_pool import StockPool
    from src.strategy.transformer_engine import StockTransformer

    print("\n" + "=" * 70)
    print("  第三层: Transformer 时序引擎 重训")
    print("=" * 70)
    t0 = time.time()

    import torch
    print(f"  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    cache = DataCache()
    pool = StockPool()

    # 1. 构建序列数据集
    print("\n  [1/3] 构建序列数据集...")
    engine = StockTransformer(d_input=12, d_model=64, nhead=4, num_layers=4, d_ff=256, seq_len=60)

    def progress(c, t):
        if c % 1000 == 0:
            print(f"    提取: {c}/{t} ({c/t*100:.0f}%)")

    dataset = engine.build_dataset(cache, pool, progress)
    n_total = dataset['total']
    print(f"    总样本: {n_total:,}, 训练: {len(dataset['X_train']):,}")

    # 2. 训练
    print("\n  [2/3] GPU 训练...")
    report = engine.train_model(dataset, epochs=40, batch_size=2048, lr=5e-4, weight_decay=1e-4)

    # 3. 保存模型
    print("\n  [3/3] 保存模型...")
    model_path = os.path.join(DATA_DIR, 'transformer_model.pt')
    backup_model(model_path, 'transformer')
    engine.save(model_path)

    elapsed = time.time() - t0
    result = {
        'val_auc': report['best_val_auc'],
        'test_auc': report['test_auc'],
        'discrimination': report['test_discrimination'],
        'test_p@50': report.get('test_p@50', 0),
        'epochs_trained': report['epochs_trained'],
        'model_params': report['model_params'],
        'total_samples': n_total,
        'elapsed': round(elapsed, 0),
    }

    print(f"\n  ✅ 第三层训练完成 ({elapsed:.0f}秒)")
    print(f"     验证AUC: {report['best_val_auc']:.4f}  测试AUC: {report['test_auc']:.4f}")
    print(f"     区分度: {report['test_discrimination']:.4f}")

    return result


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='AI超级策略 三层模型一键重训')
    parser.add_argument('--layer', type=str, default='1,2,3',
                        help='要训练的层: 1=XGBoost, 2=形态聚类, 3=Transformer (逗号分隔, 默认全部)')
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layer.split(',')]

    print("=" * 70)
    print("  AI超级策略 — 三层模型重训")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  训练层: {layers}")
    print("=" * 70)

    total_t0 = time.time()
    all_results = {}

    if 1 in layers:
        try:
            all_results['layer1_xgboost'] = train_layer1()
        except Exception as e:
            print(f"\n  ❌ 第一层训练失败: {e}")
            import traceback; traceback.print_exc()

    if 2 in layers:
        try:
            all_results['layer2_pattern'] = train_layer2()
        except Exception as e:
            print(f"\n  ❌ 第二层训练失败: {e}")
            import traceback; traceback.print_exc()

    if 3 in layers:
        try:
            all_results['layer3_transformer'] = train_layer3()
        except Exception as e:
            print(f"\n  ❌ 第三层训练失败: {e}")
            import traceback; traceback.print_exc()

    total_elapsed = time.time() - total_t0

    # 保存训练报告
    all_results['train_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    all_results['total_elapsed'] = round(total_elapsed, 0)
    all_results['layers_trained'] = layers

    report_path = os.path.join(DATA_DIR, 'retrain_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    # 汇总
    print("\n" + "=" * 70)
    print("  重训完成汇总")
    print("=" * 70)

    if 'layer1_xgboost' in all_results:
        r = all_results['layer1_xgboost']
        print(f"  第一层 XGBoost:     AUC={r['test_auc']:.4f}  P@50={r['test_p@50']:.3f}  ({r['elapsed']:.0f}秒)")

    if 'layer2_pattern' in all_results:
        r = all_results['layer2_pattern']
        print(f"  第二层 形态聚类:     {r['high_wr_clusters']}种高胜率形态  ({r['elapsed']:.0f}秒)")

    if 'layer3_transformer' in all_results:
        r = all_results['layer3_transformer']
        print(f"  第三层 Transformer: AUC={r['test_auc']:.4f}  区分度={r['discrimination']:.4f}  ({r['elapsed']:.0f}秒)")

    print(f"\n  总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"  报告: {report_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
