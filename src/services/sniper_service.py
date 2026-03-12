# -*- coding: utf-8 -*-
"""
盘中狙击引擎

核心思路:
  1. 盘后: AI 扫描生成「作战计划」(battle_plan.json)
     - 买入候选: 股票代码、目标买入价、最高可接受价、止损价、止盈价、AI评分
     - 卖出监控: 当前持仓的止损/止盈/追踪止损条件
  2. 盘中: 狙击引擎每 N 秒循环:
     - 拉实时价格
     - 买入狙击: 价格落入 [buy_price, buy_upper] → 执行模拟买入
     - 卖出狙击: 触碰止损/止盈价 → 执行模拟卖出
     - 每次触发发微信通知
  3. 前端: 「作战计划」Tab 展示计划表、狙击状态、今日触发记录
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Optional

import config
from src.data.data_fetcher import batch_get_realtime_prices, get_realtime_price
from src.trading.auto_trader import AutoTrader
from src.trading.paper_trading import PaperTradingAccount
from src.trading.position_monitor import is_trading_time
from src.trading.decision_kernel import calc_a_share_lot_shares
from src.utils.state_store import load_json_safe, write_json_atomic
from src.utils.wechat_notifier import send_notification, format_buy_html, format_sell_html

logger = logging.getLogger(__name__)

PLAN_PATH = os.path.join(config.DATA_ROOT, "battle_plan.json")
SNIPER_LOG_PATH = os.path.join(config.DATA_ROOT, "sniper_log.json")


def generate_battle_plan(account: Optional[PaperTradingAccount] = None) -> dict:
    """
    从最新 AI 评分和当前持仓生成「作战计划」

    返回: battle_plan dict，同时写入 battle_plan.json
    """
    account = account or PaperTradingAccount()
    today = datetime.now().strftime("%Y-%m-%d")

    # 买入候选：从 ai_daily_scores.json 读取
    scores_path = os.path.join(config.DATA_ROOT, "ai_daily_scores.json")
    scores_data = load_json_safe(scores_path, default=None, log_prefix="作战计划")

    buy_targets = []
    if scores_data and scores_data.get("status") == "ok":
        items = scores_data.get("top50") or (scores_data.get("all_scores", [])[:50])
        positions = account.get_positions()
        held_codes = set(positions["stock_code"].tolist()) if not positions.empty else set()

        for rec in items:
            code = rec.get("stock_code", "")
            score = rec.get("final_score", 0) or rec.get("ai_score", 0)
            buy_price = rec.get("buy_price", 0)
            buy_upper = rec.get("buy_upper", 0)

            if not code or code in held_codes:
                continue
            if score < config.AUTO_SCORE_THRESHOLD:
                continue
            if buy_price <= 0 or buy_upper <= 0:
                continue

            buy_targets.append({
                "stock_code": code,
                "stock_name": rec.get("stock_name", ""),
                "ai_score": round(score, 1),
                "buy_price": round(buy_price, 2),
                "buy_upper": round(buy_upper, 2),
                "sell_target": round(rec.get("sell_target", 0), 2),
                "sell_stop": round(rec.get("sell_stop", 0), 2),
                "risk_reward": round(rec.get("risk_reward", 0), 1) if rec.get("risk_reward") else None,
                "hold_days": rec.get("hold_days", ""),
                "status": "waiting",
            })

    # 卖出监控：从当前持仓构造
    sell_targets = []
    positions = account.get_positions()
    if not positions.empty:
        for _, pos in positions.iterrows():
            code = pos["stock_code"]
            avg_cost = pos["avg_cost"]
            sell_targets.append({
                "stock_code": code,
                "stock_name": pos.get("stock_name", ""),
                "avg_cost": round(avg_cost, 2),
                "shares": int(pos["shares"]),
                "sell_stop": round(avg_cost * (1 - config.STOP_LOSS_PCT), 2),
                "sell_target": round(avg_cost * (1 + config.TAKE_PROFIT_PCT), 2),
                "status": "monitoring",
            })

    plan = {
        "plan_date": today,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "buy_targets": buy_targets[:20],
        "sell_targets": sell_targets,
        "config": {
            "interval_seconds": config.SNIPER_INTERVAL_SECONDS,
            "buy_enabled": config.SNIPER_BUY_ENABLED,
            "sell_enabled": config.SNIPER_SELL_ENABLED,
        },
    }

    write_json_atomic(PLAN_PATH, plan)
    logger.info("[作战计划] 已生成: 买入候选%d只, 卖出监控%d只", len(buy_targets[:20]), len(sell_targets))
    return plan


def load_battle_plan() -> dict:
    return load_json_safe(PLAN_PATH, default={}, log_prefix="作战计划")


def _append_sniper_log(entry: dict):
    logs = load_json_safe(SNIPER_LOG_PATH, default=[], log_prefix="狙击日志")
    if not isinstance(logs, list):
        logs = []
    logs.append(entry)
    if len(logs) > 200:
        logs = logs[-200:]
    write_json_atomic(SNIPER_LOG_PATH, logs)


def run_sniper_cycle(account: PaperTradingAccount, plan: dict, trader: AutoTrader = None) -> dict:
    """
    执行一轮狙击检查

    返回: {"buy_triggered": [...], "sell_triggered": [...]}
    """
    result = {"buy_triggered": [], "sell_triggered": []}
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if trader is None:
        trader = AutoTrader(account)

    all_codes = []
    buy_map = {}
    sell_map = {}

    if config.SNIPER_BUY_ENABLED:
        for t in plan.get("buy_targets", []):
            if t.get("status") == "waiting":
                code = t["stock_code"]
                all_codes.append(code)
                buy_map[code] = t

    if config.SNIPER_SELL_ENABLED:
        for t in plan.get("sell_targets", []):
            if t.get("status") == "monitoring":
                code = t["stock_code"]
                all_codes.append(code)
                sell_map[code] = t

    if not all_codes:
        return result

    prices = {}
    try:
        rt_map = batch_get_realtime_prices(list(set(all_codes)))
        for code, info in rt_map.items():
            p = info.get("close", 0) or info.get("price", 0)
            if p > 0:
                prices[code] = float(p)
    except Exception as exc:
        logger.warning("[狙击] 批量获取价格失败: %s", exc)
        return result

    # 买入狙击
    for code, target in buy_map.items():
        price = prices.get(code, 0)
        if price <= 0:
            continue
        buy_lower = target["buy_price"]
        buy_upper = target["buy_upper"]

        if buy_lower <= price <= buy_upper:
            account_info = account.get_account_info()
            position_ratio = min(config.POSITION_RATIO, config.MAX_SINGLE_POSITION)
            target_amount = account_info["initial_capital"] * position_ratio
            target_amount = min(target_amount, account_info["cash"] * 0.95)
            shares = calc_a_share_lot_shares(target_amount, price)

            if shares < 100:
                continue

            buy_result = account.buy(code, target["stock_name"], price, shares)
            if buy_result.get("success"):
                target["status"] = "filled"
                trigger_text = f"现价{price:.2f}落入[{buy_lower:.2f}, {buy_upper:.2f}]"
                entry = {
                    "time": now_str, "action": "买入", "stock_code": code,
                    "stock_name": target["stock_name"], "price": price,
                    "shares": shares, "ai_score": target.get("ai_score", 0),
                    "trigger": trigger_text,
                }
                result["buy_triggered"].append(entry)
                _append_sniper_log(entry)

                trader._log_trade(
                    action="买入", stock_code=code, stock_name=target["stock_name"],
                    price=price, shares=shares,
                    reason=f"[狙击买入] {trigger_text}",
                    ai_score=target.get("ai_score", 0),
                    stop_price=target.get("sell_stop", 0),
                    target_price=target.get("sell_target", 0),
                    trade_id=buy_result.get("trade_id", 0),
                )

                title = f"[狙击买入] {target['stock_name']}({code}) @{price:.2f}"
                body = format_buy_html(
                    target["stock_name"], code, price, shares,
                    target.get("ai_score", 0), buy_lower, buy_upper,
                    target.get("sell_target", 0), target.get("sell_stop", 0), now_str,
                )
                send_notification(title, body, template="html")
                logger.info("[狙击买入] %s @%.2f %d股", code, price, shares)

    # 卖出狙击
    for code, target in sell_map.items():
        price = prices.get(code, 0)
        if price <= 0:
            continue
        stop = target.get("sell_stop", 0)
        tp = target.get("sell_target", 0)
        shares = target.get("shares", 0)

        triggered = False
        trigger_reason = ""

        if stop > 0 and price <= stop:
            triggered = True
            trigger_reason = f"触碰止损 {stop:.2f}"
        elif tp > 0 and price >= tp:
            triggered = True
            trigger_reason = f"触碰止盈 {tp:.2f}"

        if triggered and shares > 0:
            sell_result = account.sell(code, target.get("stock_name", ""), price, shares)
            if sell_result.get("success"):
                target["status"] = "exited"
                pnl = sell_result.get("profit", 0)
                pnl_pct = sell_result.get("profit_pct", 0)

                entry = {
                    "time": now_str, "action": "卖出", "stock_code": code,
                    "stock_name": target.get("stock_name", ""), "price": price,
                    "shares": shares, "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
                    "trigger": trigger_reason,
                }
                result["sell_triggered"].append(entry)
                _append_sniper_log(entry)

                trader._log_trade(
                    action="卖出", stock_code=code, stock_name=target.get("stock_name", ""),
                    price=price, shares=shares,
                    reason=f"[狙击卖出] {trigger_reason}",
                    pnl=round(pnl, 2), pnl_pct=round(pnl_pct, 2),
                    trade_id=sell_result.get("trade_id", 0),
                )

                title = f"[狙击卖出] {target.get('stock_name', '')}({code}) @{price:.2f} {pnl_pct:+.1f}%"
                body = format_sell_html(
                    target.get("stock_name", ""), code, price, shares,
                    pnl, pnl_pct, trigger_reason, now_str,
                )
                send_notification(title, body, template="html")
                logger.info("[狙击卖出] %s @%.2f %s", code, price, trigger_reason)

    write_json_atomic(PLAN_PATH, plan)
    return result


def run_sniper_loop():
    """
    盘中狙击主循环（供 run_sniper.py 或定时任务调用）
    """
    logger.info("=" * 50)
    logger.info("盘中狙击引擎启动 %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("间隔: %d秒, 买入:%s, 卖出:%s",
                config.SNIPER_INTERVAL_SECONDS,
                "开启" if config.SNIPER_BUY_ENABLED else "关闭",
                "开启" if config.SNIPER_SELL_ENABLED else "关闭")
    logger.info("=" * 50)

    account = PaperTradingAccount()
    trader = AutoTrader(account)
    plan = load_battle_plan()

    if not plan or plan.get("plan_date") != datetime.now().strftime("%Y-%m-%d"):
        logger.info("[狙击] 无今日作战计划，自动生成...")
        plan = generate_battle_plan(account)

    cycle = 0
    while True:
        if not is_trading_time():
            now = datetime.now()
            t = now.hour * 60 + now.minute
            if t >= 900:
                logger.info("[狙击] 已收盘，结束")
                break
            if now.weekday() >= 5:
                logger.info("[狙击] 非交易日，退出")
                break
            time.sleep(60)
            continue

        cycle += 1
        try:
            triggered = run_sniper_cycle(account, plan, trader=trader)
            n_buy = len(triggered.get("buy_triggered", []))
            n_sell = len(triggered.get("sell_triggered", []))
            if n_buy or n_sell:
                logger.info("[狙击] 第%d轮: 买入%d只 卖出%d只", cycle, n_buy, n_sell)
        except Exception as exc:
            logger.error("[狙击] 第%d轮异常: %s", cycle, exc, exc_info=True)

        time.sleep(config.SNIPER_INTERVAL_SECONDS)

    logger.info("盘中狙击引擎结束, 共%d轮", cycle)
