# -*- coding: utf-8 -*-
"""
同步股市数据：增量更新本地K线缓存到最新
可单独运行，不执行AI扫描/邮件等后续步骤

用法:
    python sync_data.py
    python sync_data.py --days 365
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from daily_job import sync_stock_data


def main():
    parser = argparse.ArgumentParser(description="同步股市K线数据到本地缓存")
    parser.add_argument("--days", type=int, default=730, help="拉取历史天数，默认730")
    args = parser.parse_args()

    def on_progress(c, t, n):
        pct = c / t * 100 if t > 0 else 0
        print(f"\r[{c}/{t}] {pct:.0f}% {n[:12]:<12}", end="", flush=True)

    print("开始同步数据...")
    result = sync_stock_data(days=args.days, progress_callback=on_progress)
    print(f"\n完成！{result['message']}")
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
