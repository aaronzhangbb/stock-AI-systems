# -*- coding: utf-8 -*-
"""
导出历史数据到 Excel，直观查看
同时打印缓存统计信息
"""
import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'stock_cache.db')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'historical_data_sample.xlsx')


def main():
    conn = sqlite3.connect(DB_PATH)

    # 1. 统计信息
    meta = pd.read_sql_query('SELECT * FROM cache_meta ORDER BY stock_code', conn)
    total_stocks = len(meta)
    total_records = pd.read_sql_query('SELECT COUNT(*) as cnt FROM daily_kline', conn).iloc[0]['cnt']
    print(f"=== 缓存统计 ===")
    print(f"已缓存股票数: {total_stocks}")
    print(f"总K线记录数: {total_records:,.0f}")
    if not meta.empty:
        print(f"日期范围: {meta['first_date'].min()} ~ {meta['last_date'].max()}")

    # 2. 选取几只代表性股票导出
    sample_codes = ['600519', '000858', '601318', '000001', '300750']
    # 也取缓存中实际存在的前10只
    existing = meta['stock_code'].tolist()
    extra = [c for c in existing if c not in sample_codes][:5]
    sample_codes = [c for c in sample_codes if c in existing] + extra

    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        # Sheet1: 缓存概览
        overview = meta[['stock_code', 'first_date', 'last_date', 'updated_at']].copy()
        # 加入名称
        names = pd.read_sql_query('SELECT stock_code, stock_name FROM stock_names', conn)
        overview = overview.merge(names, on='stock_code', how='left')
        # 加入记录数
        counts = pd.read_sql_query(
            'SELECT stock_code, COUNT(*) as record_count FROM daily_kline GROUP BY stock_code',
            conn
        )
        overview = overview.merge(counts, on='stock_code', how='left')
        overview = overview[['stock_code', 'stock_name', 'first_date', 'last_date', 'record_count', 'updated_at']]
        overview.columns = ['股票代码', '股票名称', '数据起始日', '数据截止日', '记录条数', '最后更新']
        overview.to_excel(writer, sheet_name='缓存概览', index=False)
        print(f"\nSheet「缓存概览」: {len(overview)} 只股票")

        # Sheet2~N: 各只样本股票的完整K线
        for code in sample_codes:
            df = pd.read_sql_query(
                'SELECT date, open, high, low, close, volume, amount, pctChg, turnover '
                'FROM daily_kline WHERE stock_code = ? ORDER BY date',
                conn, params=[code]
            )
            if df.empty:
                continue
            name_row = names[names['stock_code'] == code]
            name = name_row.iloc[0]['stock_name'] if not name_row.empty else code
            sheet_name = f"{name}({code})"[:31]  # Excel sheet name max 31 chars
            df.columns = ['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额', '涨跌幅%', '换手率%']
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Sheet「{sheet_name}」: {len(df)} 条记录")

    conn.close()
    print(f"\n已导出到: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
