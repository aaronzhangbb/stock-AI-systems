# -*- coding: utf-8 -*-
"""
同步A股全部行业板块和个股数据到本地数据库
使用申万行业分类（不依赖 push2.eastmoney.com）
运行方式: python sync_stock_pool.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from src.data.stock_pool import StockPool


def main():
    pool = StockPool()
    stats = pool.get_stats()
    print(f"当前股票池: {stats['board_count']}个行业, {stats['stock_count']}只股票")
    print(f"最后更新: {stats['last_update']}")
    print()
    print("开始同步（申万行业分类）...")
    pool.update_industry_boards()
    print()
    stats = pool.get_stats()
    print(f"同步后: {stats['board_count']}个行业, {stats['stock_count']}只股票")


if __name__ == '__main__':
    main()
