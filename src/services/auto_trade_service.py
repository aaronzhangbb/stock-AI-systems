import logging
import os
from datetime import datetime

import config
from src.trading.auto_trader import AutoTrader
from src.trading.paper_trading import PaperTradingAccount
from src.utils.runtime_guard import file_lock
from src.utils.state_store import write_json_atomic


logger = logging.getLogger(__name__)

AUTO_TRADE_RESULT_PATH = os.path.join(config.DATA_ROOT, "last_auto_result.json")
AUTO_TRADE_LOCK_PATH = os.path.join(config.LOG_ROOT, "locks", "auto_trade.lock")


def execute_auto_trade(account: PaperTradingAccount | None = None, *, rescan: bool = True, progress_callback=None) -> dict:
    account = account or PaperTradingAccount()
    with file_lock(AUTO_TRADE_LOCK_PATH, stale_seconds=7200, metadata={"job": "auto_trade"}):
        trader = AutoTrader(account)
        result = trader.execute(rescan=rescan, progress_callback=progress_callback)
        write_json_atomic(
            AUTO_TRADE_RESULT_PATH,
            {k: v for k, v in result.items() if k != "scan_result"},
        )
        return result


def build_auto_trade_error_result(error_message: str) -> dict:
    return {
        "sell_actions": [],
        "buy_actions": [],
        "hold_alerts": [],
        "skipped": [],
        "summary": f"❌ 执行异常: {error_message}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": error_message,
    }
