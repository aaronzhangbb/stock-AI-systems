# -*- coding: utf-8 -*-
"""
每日AI评分扫描
对全市场5008只可交易A股进行AI评分,输出Top推荐
"""
import sys
import os
import time
import json

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))

from src.data.data_cache import DataCache
from src.data.stock_pool import StockPool
from src.strategy.ai_engine_v2 import AIScorer

print("=" * 90)
print("  AI策略V2.0 — 每日全市场评分扫描")
print("=" * 90)

t0 = time.time()

# 初始化
cache = DataCache()
pool = StockPool()
scorer = AIScorer()

print(f"模型加载完成: {scorer.model_path}")

def progress(current, total):
    print(f"  扫描进度: {current}/{total} ({current/total*100:.0f}%)", flush=True)

# 全市场扫描
df = scorer.scan_market(cache, pool, top_n=50, progress_callback=progress)

elapsed = time.time() - t0
print(f"\n扫描完成! {len(df)} 只股票已评分, 耗时{elapsed:.0f}秒")

# 评分分布
print(f"\n{'='*90}")
print(f"  评分分布")
print(f"{'='*90}")
print(f"  90+分 (强烈推荐): {len(df[df['ai_score'] >= 90])} 只")
print(f"  80-90 (推荐):     {len(df[(df['ai_score'] >= 80) & (df['ai_score'] < 90)])} 只")
print(f"  70-80 (关注):     {len(df[(df['ai_score'] >= 70) & (df['ai_score'] < 80)])} 只")
print(f"  60-70 (中性):     {len(df[(df['ai_score'] >= 60) & (df['ai_score'] < 70)])} 只")
print(f"  <60   (回避):     {len(df[df['ai_score'] < 60])} 只")
print(f"  平均分: {df['ai_score'].mean():.1f}")

# Top 30
print(f"\n{'='*90}")
print(f"  AI评分 Top 30 推荐")
print(f"{'='*90}")
print(f"{'排名':<4} {'代码':<8} {'名称':<10} {'行业':<10} {'AI评分':>7} {'收盘价':>8} "
      f"{'波动率':>7} {'布林位':>7} {'RSI':>6} {'5日涨跌':>8} {'量比':>6} {'MA60偏离':>8}")
print("-" * 105)

for _, row in df.head(30).iterrows():
    vol20 = f"{row['volatility_20d']:.2f}" if row['volatility_20d'] is not None else "N/A"
    bb = f"{row['bb_pos']:.3f}" if row['bb_pos'] is not None else "N/A"
    rsi = f"{row['rsi_14']:.0f}" if row['rsi_14'] is not None else "N/A"
    ret5 = f"{row['ret_5d']:+.1f}%" if row['ret_5d'] is not None else "N/A"
    volr = f"{row['vol_ratio']:.2f}" if row['vol_ratio'] is not None else "N/A"
    ma60 = f"{row['ma60_diff']:+.1f}%" if row['ma60_diff'] is not None else "N/A"
    
    star = " ★" if row['ai_score'] >= 90 else ""
    print(f"{row['rank']:<4} {row['stock_code']:<8} {row['stock_name']:<10} {row['board_name']:<10} "
          f"{row['ai_score']:>6.1f}  {row['close']:>8.2f} "
          f"{vol20:>7} {bb:>7} {rsi:>6} {ret5:>8} {volr:>6} {ma60:>8}{star}")

# Bottom 10
print(f"\n{'='*90}")
print(f"  AI评分 Bottom 10 (回避)")
print(f"{'='*90}")
for _, row in df.tail(10).iterrows():
    print(f"  {row['stock_code']} {row['stock_name']:<10} AI评分={row['ai_score']:.1f} "
          f"收盘={row['close']:.2f}")

# 行业分布
print(f"\n{'='*90}")
print(f"  Top 50 行业分布")
print(f"{'='*90}")
top50 = df.head(50)
board_dist = top50['board_name'].value_counts()
for board, count in board_dist.items():
    pct = count / len(top50) * 100
    bar = "█" * int(pct / 2)
    print(f"  {board:<12} {count:>3}只 ({pct:>4.0f}%) {bar}")

# 保存结果
output = {
    'scan_date': time.strftime('%Y-%m-%d'),
    'scan_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    'total_scored': len(df),
    'elapsed_seconds': round(elapsed, 1),
    'score_distribution': {
        'above_90': int(len(df[df['ai_score'] >= 90])),
        'above_80': int(len(df[df['ai_score'] >= 80])),
        'above_70': int(len(df[df['ai_score'] >= 70])),
        'mean_score': round(float(df['ai_score'].mean()), 1),
    },
    'top50': df.head(50).to_dict(orient='records'),
    'bottom10': df.tail(10).to_dict(orient='records'),
}

out_path = os.path.join('data', 'ai_daily_scores.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=str)
print(f"\n结果已保存: {out_path}")
