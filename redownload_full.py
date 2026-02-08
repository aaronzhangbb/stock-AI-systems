# -*- coding: utf-8 -*-
"""
强制重新下载所有数据不足的股票（需要2年≈480+交易日）
步骤：
1. 找出缓存中数据不足200条的股票
2. 清除这些股票的旧缓存
3. 重新从API下载完整2年数据
"""
import sqlite3
import os
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from src.data.data_cache import DataCache
from src.data.data_fetcher import _fetch_from_api, _get_cache
from src.data.stock_pool import StockPool

CACHE_DB = os.path.join(os.path.dirname(__file__), 'data', 'stock_cache.db')
MIN_RECORDS = 600  # 少于这个数认为数据不足（3年≈730个交易日）

# 北交所跳过
SKIP_PREFIXES = ('92',)


def main():
    cache = _get_cache()
    pool = StockPool()
    
    # 1. 统计当前缓存情况
    conn = sqlite3.connect(CACHE_DB)
    stats = conn.execute("""
        SELECT stock_code, COUNT(*) as cnt 
        FROM daily_kline 
        GROUP BY stock_code
    """).fetchall()
    conn.close()
    
    record_map = {code: cnt for code, cnt in stats}
    
    # 获取股票池
    all_stocks = pool.get_all_stocks()
    if all_stocks.empty:
        print("股票池为空，请先同步")
        return
    
    all_stocks = all_stocks.drop_duplicates(subset=['stock_code'])
    total = len(all_stocks)
    
    # 2. 找出需要重新下载的股票
    need_download = []
    sufficient = 0
    skipped = 0
    
    for _, row in all_stocks.iterrows():
        code = row['stock_code']
        name = row['stock_name']
        
        if code.startswith(SKIP_PREFIXES):
            skipped += 1
            continue
        
        cnt = record_map.get(code, 0)
        if cnt < MIN_RECORDS:
            need_download.append((code, name, cnt))
        else:
            sufficient += 1
    
    print(f"=" * 60)
    print(f"缓存诊断：")
    print(f"  股票池总数: {total}")
    print(f"  数据充足 (>={MIN_RECORDS}条): {sufficient}")
    print(f"  数据不足 (<{MIN_RECORDS}条): {len(need_download)}")
    print(f"  跳过(北交所): {skipped}")
    print(f"=" * 60)
    
    if not need_download:
        print("所有股票数据已充足，无需重新下载！")
        return
    
    # 3. 日期范围：3年
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=1095)).strftime('%Y%m%d')
    
    print(f"\n开始重新下载 {len(need_download)} 只股票的3年数据...")
    print(f"日期范围: {start_date} ~ {end_date}")
    print(f"预计耗时: {len(need_download) * 0.2 / 60:.0f} ~ {len(need_download) * 0.5 / 60:.0f} 分钟\n")
    
    success = 0
    failed = 0
    start_time = time.time()
    
    for i, (code, name, old_cnt) in enumerate(need_download):
        try:
            # 清除旧的不完整数据
            cache.clear_cache(code)
            
            # 重新全量下载
            df = _fetch_from_api(code, start_date, end_date)
            
            if not df.empty:
                cache.save_kline(code, df)
                success += 1
                if (i + 1) % 50 == 0 or i < 5:
                    elapsed = time.time() - start_time
                    eta = elapsed / (i + 1) * (len(need_download) - i - 1) / 60
                    print(f"  [{i+1}/{len(need_download)}] {name}({code}): "
                          f"{old_cnt} → {len(df)}条  "
                          f"(已用{elapsed/60:.1f}分 预计还需{eta:.1f}分)")
            else:
                failed += 1
                if failed <= 10:
                    print(f"  [{i+1}] {name}({code}): 下载失败")
                    
        except Exception as e:
            failed += 1
            if failed <= 10:
                print(f"  [{i+1}] {name}({code}): 异常 {e}")
        
        # 每100只打印进度
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(need_download) - i - 1) / 60
            print(f"\n  === 进度: {i+1}/{len(need_download)} | "
                  f"成功={success} 失败={failed} | "
                  f"已用{elapsed/60:.1f}分 预计还需{eta:.1f}分 ===\n")
    
    total_time = time.time() - start_time
    
    print(f"\n{'=' * 60}")
    print(f"下载完成！")
    print(f"  成功: {success}/{len(need_download)}")
    print(f"  失败: {failed}")
    print(f"  耗时: {total_time/60:.1f} 分钟")
    
    # 4. 验证结果
    conn = sqlite3.connect(CACHE_DB)
    stats2 = conn.execute("""
        SELECT 
            COUNT(DISTINCT stock_code) as total_stocks,
            COUNT(*) as total_records,
            AVG(cnt) as avg_records
        FROM (
            SELECT stock_code, COUNT(*) as cnt 
            FROM daily_kline 
            GROUP BY stock_code
        )
    """).fetchone()
    
    insufficient = conn.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT stock_code, COUNT(*) as cnt 
            FROM daily_kline 
            GROUP BY stock_code
            HAVING cnt < {MIN_RECORDS}
        )
    """).fetchone()[0]
    conn.close()
    
    print(f"\n最终缓存状态：")
    print(f"  总股票数: {stats2[0]}")
    print(f"  总记录数: {stats2[1]:,.0f}")
    print(f"  平均每只: {stats2[2]:.0f} 条")
    print(f"  仍不足{MIN_RECORDS}条: {insufficient}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
