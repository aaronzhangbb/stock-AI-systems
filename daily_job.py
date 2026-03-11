# -*- coding: utf-8 -*-
"""
每日收盘后闭环任务（精简版 - 纯AI策略）：
0. 采集大盘情绪（涨跌比/成交额/主力资金/北向/融资）
1. 增量更新本地缓存
2. AI三层超级策略扫描（XGBoost + 形态聚类 + Transformer）
3. 检查已持仓股的卖出时机
4. 一封邮件推送：情绪 + AI操作清单 + 卖出提醒
"""

import os
import sys
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.services.ai_scan_service import run_ai_super_scan as _run_ai_super_scan_service
from src.services.daily_job_service import run_daily_cycle as _run_daily_cycle_service
from src.services.daily_job_service import sync_stock_data as _sync_stock_data_service
import config

# ========== 日志配置 ==========
LOG_DIR = config.LOG_ROOT
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"daily_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def run_ai_super_scan():
    """兼容旧入口：转发到统一 AI 扫描服务。"""
    return _run_ai_super_scan_service()


def sync_stock_data(days: int = 730, progress_callback=None) -> dict:
    """兼容旧入口：转发到统一同步服务。"""
    return _sync_stock_data_service(days=days, progress_callback=progress_callback)


def run_daily_job():
    """执行每日收盘闭环任务（统一服务层入口）。"""
    return _run_daily_cycle_service()


if __name__ == "__main__":
    run_daily_job()
