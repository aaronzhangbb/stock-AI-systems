"""
实盘与回测共享的轻量决策内核。

这里先统一最核心的执行语义：
1. 何时把“建议卖出”升级为自动执行
2. 如何按目标金额换算 A 股整手股数
3. 传统回测信号如何映射到统一的买卖动作
"""

import config


def should_execute_sell_advice(advice: str, auto_sell_urgency: int | None = None) -> bool:
    auto_sell_urgency = config.AUTO_SELL_URGENCY if auto_sell_urgency is None else auto_sell_urgency
    if advice == "立即卖出":
        return True
    if advice == "建议卖出" and auto_sell_urgency <= 1:
        return True
    return False


def calc_a_share_lot_shares(target_amount: float, price: float) -> int:
    if price <= 0 or target_amount <= 0:
        return 0
    return int(target_amount / price / 100) * 100


def should_enter_backtest_position(signal_value: int, current_shares: int) -> bool:
    return signal_value == 1 and current_shares == 0


def should_exit_backtest_position(signal_value: int, current_shares: int) -> bool:
    return signal_value == -1 and current_shares > 0
