# -*- coding: utf-8 -*-
"""
盘中实时监控脚本
- 交易时间每15分钟检查一次持仓
- 使用实时价格检测止损/止盈/追踪止损
- 触发卖出信号时立即发送邮件
- 非交易时间自动退出
"""

import os
import sys
import time
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import config
from src.trading.position_monitor import (
    check_all_manual_positions, get_sell_alerts, format_sell_alerts_text,
    is_trading_time,
)
from src.trading.paper_trading import PaperTradingAccount
from src.utils.email_notifier import send_email

# ========== 配置 ==========
CHECK_INTERVAL_MINUTES = 15   # 检查间隔（分钟）
MAX_EMAIL_PER_STOCK = 2       # 单只股票每日最多发送邮件次数（避免轰炸）

# ========== 日志 ==========
LOG_DIR = os.path.join(os.path.dirname(__file__), 'data', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"intraday_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def run_single_check(account: PaperTradingAccount,
                     email_counter: dict) -> list:
    """
    执行一次持仓检查

    参数:
        account: 交易账户
        email_counter: {stock_code: 已发送次数} 用于限频

    返回:
        list: 本次触发的卖出提醒列表
    """
    logger.info("开始持仓实时检查...")

    results = check_all_manual_positions(account, use_realtime=True)
    alerts = get_sell_alerts(results)

    if not alerts:
        logger.info(f"检查完毕: {len(results)} 只持仓，无卖出信号")
        return []

    # 筛选需要发邮件的（未超过限频）
    new_alerts = []
    for a in alerts:
        code = a['stock_code']
        sent_count = email_counter.get(code, 0)
        if sent_count < MAX_EMAIL_PER_STOCK:
            new_alerts.append(a)
        else:
            logger.info(f"  {code} {a['stock_name']} 已达到邮件上限({MAX_EMAIL_PER_STOCK}次), 跳过")

    if not new_alerts:
        logger.info(f"有 {len(alerts)} 条提醒, 但均已达邮件上限")
        return alerts

    # 发送邮件
    now_str = datetime.now().strftime('%H:%M')
    alert_text = format_sell_alerts_text(new_alerts)
    subject = f"【盘中预警 {now_str}】{len(new_alerts)} 只持仓触发卖出信号"

    body_lines = [
        f"盘中实时监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"检查持仓: {len(results)} 只",
        f"触发提醒: {len(new_alerts)} 只",
        "",
        alert_text,
        "",
        "----",
        "QuantX 盘中监控 · 请及时处理卖出信号",
    ]
    body = "\n".join(body_lines)

    ok = send_email(subject, body)
    if ok:
        logger.info(f"预警邮件已发送: {len(new_alerts)} 只")
        for a in new_alerts:
            code = a['stock_code']
            email_counter[code] = email_counter.get(code, 0) + 1
    else:
        logger.warning("预警邮件发送失败!")

    # 日志记录每条提醒
    for a in alerts:
        pnl_sign = "+" if a['pnl_pct'] >= 0 else ""
        logger.info(
            f"  {'🔴' if a['advice'] == '立即卖出' else '🟡'} "
            f"{a['stock_code']} {a['stock_name']} "
            f"买:{a['buy_price']:.2f} 现:{a['current_price']:.2f} "
            f"({pnl_sign}{a['pnl_pct']:.1f}%) - {a['advice']}"
        )
        for msg in a['alerts']:
            logger.info(f"    · {msg}")

    return alerts


def wait_for_trading_time():
    """
    等待直到交易时间开始，或判断今天已收盘则退出
    """
    now = datetime.now()
    weekday = now.weekday()

    if weekday >= 5:
        logger.info(f"今天是周{'六' if weekday == 5 else '日'}, 非交易日, 退出")
        return False

    hour, minute = now.hour, now.minute
    t = hour * 60 + minute

    if t >= 900:  # 15:00 之后
        logger.info("今日已收盘, 退出")
        return False

    if t < 570:  # 9:30 之前
        wait_min = 570 - t
        logger.info(f"距开盘还有 {wait_min} 分钟, 等待中...")
        # 每分钟检查一次，直到开盘
        while not is_trading_time():
            now2 = datetime.now()
            if now2.hour * 60 + now2.minute >= 900:
                return False
            time.sleep(60)
        return True

    if 690 < t < 780:  # 11:30 - 13:00 午休
        wait_min = 780 - t
        logger.info(f"午休时间, 还有 {wait_min} 分钟恢复, 等待中...")
        while not is_trading_time():
            now2 = datetime.now()
            if now2.hour * 60 + now2.minute >= 900:
                return False
            time.sleep(60)
        return True

    return is_trading_time()


def run_intraday_monitor():
    """
    盘中实时监控主循环

    运行逻辑:
    1. 等待交易时间
    2. 每 CHECK_INTERVAL_MINUTES 分钟检查一次
    3. 收盘后自动退出
    """
    logger.info("=" * 50)
    logger.info(f"盘中监控启动 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"检查间隔: {CHECK_INTERVAL_MINUTES} 分钟")
    logger.info(f"单股邮件上限: {MAX_EMAIL_PER_STOCK} 次/日")
    logger.info("=" * 50)

    account = PaperTradingAccount()
    email_counter = {}  # {stock_code: 已发送次数}
    check_count = 0

    # 先检查是否有持仓
    manual_df = account.list_manual_positions()
    if manual_df.empty:
        logger.info("当前无持仓, 无需监控, 退出")
        return

    logger.info(f"当前持仓 {len(manual_df)} 只:")
    for _, row in manual_df.iterrows():
        logger.info(f"  {row['stock_code']} {row.get('stock_name', '')} "
                     f"买入价:{row['buy_price']}")

    while True:
        # 检查是否在交易时间
        if not is_trading_time():
            can_continue = wait_for_trading_time()
            if not can_continue:
                break

        # 执行检查
        check_count += 1
        logger.info(f"--- 第 {check_count} 轮检查 ({datetime.now().strftime('%H:%M:%S')}) ---")

        try:
            run_single_check(account, email_counter)
        except Exception as e:
            logger.error(f"检查异常: {e}", exc_info=True)

        # 等待下一轮
        logger.info(f"下一轮检查在 {CHECK_INTERVAL_MINUTES} 分钟后")
        time.sleep(CHECK_INTERVAL_MINUTES * 60)

        # 检查是否还在交易时间
        now = datetime.now()
        t = now.hour * 60 + now.minute
        if t >= 900:  # 15:00
            logger.info("已收盘, 监控结束")
            break

    # 收盘总结
    logger.info("=" * 50)
    logger.info(f"盘中监控结束 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"共检查 {check_count} 轮")
    if email_counter:
        logger.info(f"邮件发送记录: {email_counter}")
    logger.info("=" * 50)


if __name__ == "__main__":
    run_intraday_monitor()
