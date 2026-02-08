# -*- coding: utf-8 -*-
"""
按股票分类验证AI策略
检查不同类型股票上策略表现是否有差异
"""
import sqlite3
import pandas as pd
import numpy as np
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

from src.strategy.ai_strategies import AI_STRATEGIES, compute_ai_features, check_strategy_signal
from src.strategy.stock_categories import INDUSTRY_TO_STYLE

DB_CACHE = 'data/stock_cache.db'
DB_POOL = 'data/stock_pool.db'
HORIZON = 10
SAMPLE_PER_CATEGORY = 80  # 每个分类采样股票数


def log(msg):
    print(msg, flush=True)


def main():
    log("=" * 70)
    log("  按股票分类验证AI策略效果")
    log("=" * 70)

    conn_cache = sqlite3.connect(DB_CACHE)
    conn_pool = sqlite3.connect(DB_POOL)

    # 获取所有股票的分类
    meta = pd.read_sql_query('SELECT stock_code FROM cache_meta', conn_cache)
    pool = pd.read_sql_query('SELECT stock_code, board_name FROM all_stocks', conn_pool)
    merged = meta.merge(pool, on='stock_code', how='left')
    merged['style'] = merged['board_name'].map(INDUSTRY_TO_STYLE).fillna('F-未分类')

    styles = ['A-大盘稳健', 'B-科技成长', 'C-消费医药', 'D-周期制造', 'E-制造装备']

    # 只测试前3个核心策略 + 1个超卖策略
    test_strategies = [s for s in AI_STRATEGIES if s['id'] in [
        'ai_core_01', 'ai_core_02', 'ai_core_03', 'ai_balanced_01', 'ai_balanced_02'
    ]]

    results = {}  # {style: {strategy_id: {win_rate, avg_ret, trades, ...}}}

    for style in styles:
        log(f"\n{'=' * 60}")
        log(f"  分类: {style}")
        log(f"{'=' * 60}")

        style_stocks = merged[merged['style'] == style]['stock_code'].tolist()
        if len(style_stocks) > SAMPLE_PER_CATEGORY:
            np.random.seed(42)
            style_stocks = list(np.random.choice(style_stocks, SAMPLE_PER_CATEGORY, replace=False))

        log(f"  采样 {len(style_stocks)} 只股票")

        style_results = {}
        for strat in test_strategies:
            trades = []
            for code in style_stocks:
                df = pd.read_sql_query(
                    'SELECT date, open, high, low, close, volume FROM daily_kline '
                    'WHERE stock_code = ? ORDER BY date',
                    conn_cache, params=[code]
                )
                if len(df) < 100:
                    continue
                df['date'] = pd.to_datetime(df['date'])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df[df['close'] > 0].reset_index(drop=True)

                data = compute_ai_features(df)
                data = data.dropna(subset=['ma30_diff', 'rsi14', 'bb_pos'])

                # Only use last 30% as test
                split = int(len(data) * 0.7)
                test_data = data.iloc[split:]

                i = 0
                while i < len(test_data) - HORIZON:
                    row = test_data.iloc[i]
                    if check_strategy_signal(row, strat):
                        bp = float(row['close'])
                        sp = float(test_data.iloc[i + HORIZON]['close'])
                        ret = (sp - bp) / bp
                        trades.append(ret)
                        i += HORIZON
                    else:
                        i += 1

            if len(trades) >= 5:
                returns = np.array(trades)
                wr = (returns > 0).mean() * 100
                avg_ret = returns.mean() * 100
                wins = returns[returns > 0]
                losses = returns[returns <= 0]
                avg_win = wins.mean() * 100 if len(wins) > 0 else 0
                avg_lose = abs(losses.mean()) * 100 if len(losses) > 0 else 0.01
                plr = avg_win / avg_lose if avg_lose > 0 else 999
                sharpe = 0
                if returns.std() > 0:
                    sharpe = returns.mean() / returns.std() * np.sqrt(252 / HORIZON)

                style_results[strat['id']] = {
                    'name': strat['name'],
                    'trades': len(trades),
                    'win_rate': round(wr, 1),
                    'avg_return': round(avg_ret, 2),
                    'profit_loss_ratio': round(plr, 2),
                    'sharpe': round(float(sharpe), 2),
                    'avg_win': round(avg_win, 2),
                    'avg_lose': round(avg_lose, 2),
                }
                log(f"  {strat['name']:<20s} | {len(trades):>4d}次 | 胜率{wr:5.1f}% | 均收益{avg_ret:+5.2f}% | 夏普{sharpe:5.2f} | 盈亏比{plr:4.2f}")
            else:
                log(f"  {strat['name']:<20s} | 交易不足")

        results[style] = style_results

    conn_cache.close()
    conn_pool.close()

    # Summary comparison
    log(f"\n{'=' * 70}")
    log("  分类对比总结")
    log(f"{'=' * 70}")

    for strat in test_strategies:
        sid = strat['id']
        log(f"\n  策略: {strat['name']}")
        log(f"  {'分类':<14s} {'交易':>5s} {'胜率':>7s} {'均收益':>8s} {'夏普':>6s} {'盈亏比':>7s}")
        log(f"  {'-' * 50}")
        for style in styles:
            r = results.get(style, {}).get(sid)
            if r:
                log(f"  {style:<12s} {r['trades']:>5d} {r['win_rate']:>6.1f}% {r['avg_return']:>+7.2f}% {r['sharpe']:>6.2f} {r['profit_loss_ratio']:>6.2f}")
            else:
                log(f"  {style:<12s}    ---")

    log(f"\n{'=' * 70}")
    log("  结论: 不同分类下策略表现差异")
    log(f"{'=' * 70}")
    log("  (请根据上表数据得出哪类股票更适合哪种策略)")


if __name__ == '__main__':
    t0 = time.time()
    main()
    log(f"\n[DONE] {time.time() - t0:.1f}s")
