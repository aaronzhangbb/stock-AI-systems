# -*- coding: utf-8 -*-
"""盘中狙击引擎启动脚本"""
import os
import sys
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import config

LOG_DIR = config.LOG_ROOT
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"sniper_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

from src.services.sniper_service import run_sniper_loop

if __name__ == "__main__":
    run_sniper_loop()
