import logging
import os
from datetime import datetime

import config
from src.data.market_sentiment import format_sentiment_text, get_market_sentiment
from src.services.ai_scan_service import has_fresh_ai_scores, run_ai_super_scan
from src.services.auto_trade_service import execute_auto_trade
from src.strategy.scanner import MarketScanner
from src.trading.paper_trading import PaperTradingAccount
from src.trading.position_monitor import check_all_manual_positions, format_sell_alerts_text, get_sell_alerts
from src.utils.email_notifier import send_email
from src.utils.runtime_guard import file_lock


logger = logging.getLogger(__name__)
DAILY_JOB_LOCK_PATH = os.path.join(config.LOG_ROOT, "locks", "daily_job.lock")


def sync_stock_data(days: int = 730, progress_callback=None) -> dict:
    logger.info("========== 同步数据开始 %s ==========", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    scanner = MarketScanner()

    def _cb(c, t, n, s):
        if progress_callback:
            progress_callback(c, t, n)

    result = scanner.warmup_cache(days=days, progress_callback=_cb)
    logger.info(
        "========== 同步数据完成 成功=%d 失败=%d / 共%d ==========",
        result["success"],
        result["failed"],
        result["total"],
    )
    return {
        "total": result["total"],
        "success": result["success"],
        "failed": result["failed"],
        "message": f"已更新 {result['success']}/{result['total']} 只股票",
    }


def run_daily_cycle() -> dict:
    logger.info("========== 每日闭环任务启动 %s ==========", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    with file_lock(DAILY_JOB_LOCK_PATH, stale_seconds=10800, metadata={"job": "daily_job"}):
        account = PaperTradingAccount()

        logger.info("第0步: 采集大盘情绪指标...")
        sentiment_data = {}
        try:
            sentiment_data = get_market_sentiment(verbose=True)
            logger.info(
                "大盘情绪: %s分 (%s)",
                sentiment_data.get("sentiment_score", 50),
                sentiment_data.get("sentiment_level", "未知"),
            )
        except Exception as exc:
            logger.warning("情绪采集失败: %s（将使用默认中性值）", exc)
            sentiment_data = {"sentiment_score": 50, "sentiment_level": "未知", "sentiment_advice": "情绪数据获取失败"}

        logger.info("第1步: 增量更新缓存...")
        sync_stock_data(days=730)
        logger.info("缓存更新完成")

        logger.info("第2步: AI三层超级策略扫描...")
        ai_result = {"action_list": []}
        try:
            ai_result = run_ai_super_scan()
        except Exception as exc:
            logger.error("AI三层扫描失败: %s", exc, exc_info=True)

        auto_trade_result = {"sell_actions": [], "buy_actions": [], "hold_alerts": [], "summary": ""}
        try:
            logger.info("第2.5步: AI自动模拟交易...")
            if ai_result.get("status") == "ok" and has_fresh_ai_scores():
                auto_trade_result = execute_auto_trade(account, rescan=False)
            else:
                auto_trade_result["summary"] = "AI扫描未成功或评分未通过新鲜度校验，已跳过自动买入流程"
                logger.warning(auto_trade_result["summary"])
        except Exception as exc:
            logger.error("AI自动交易失败: %s", exc, exc_info=True)

        logger.info("第3步: 检查持仓卖出...")
        all_positions = check_all_manual_positions(account)
        sell_alert_list = get_sell_alerts(all_positions)
        logger.info("持仓 %d 只，需操作 %d 只", len(all_positions), len(sell_alert_list))

        logger.info("第4步: 发送邮件...")
        body_parts = []
        sentiment_text = format_sentiment_text(sentiment_data) if sentiment_data else ""
        if sentiment_text:
            body_parts.append(sentiment_text)

        if auto_trade_result.get("sell_actions") or auto_trade_result.get("buy_actions"):
            lines = ["\n🤖 AI自动交易报告\n" + "=" * 40]
            for sa in auto_trade_result.get("sell_actions", []):
                lines.append(
                    f"  卖出 {sa['stock_name']}({sa['stock_code']}) @{sa['price']:.2f} "
                    f"盈亏{sa['pnl']:+,.0f}({sa['pnl_pct']:+.1f}%) | {sa['reason'][:50]}"
                )
            for ba in auto_trade_result.get("buy_actions", []):
                lines.append(
                    f"  买入 {ba['stock_name']}({ba['stock_code']}) @{ba['price']:.2f} "
                    f"x{ba['shares']}股 AI={ba['ai_score']} 止损{ba['sell_stop']:.2f}"
                )
            for ha in auto_trade_result.get("hold_alerts", []):
                lines.append(
                    f"  提醒 {ha['stock_name']}({ha['stock_code']}) {ha.get('pnl_pct', 0):+.1f}% | "
                    f"{'; '.join(ha.get('alerts', []))[:60]}"
                )
            lines.append(f"摘要: {auto_trade_result.get('summary', '')}")
            body_parts.append("\n".join(lines))

        if ai_result.get("action_list"):
            lines = ["\n📋 AI超级策略 · 今日操作清单\n" + "=" * 40]
            for item in ai_result["action_list"]:
                lines.append(f"\n{item['stock_code']} {item['stock_name']}  综合{item['final_score']}分")
                lines.append(f"  当前价: {item['close']}")
                lines.append(f"  买入价: {item['buy_price']}  (最高: {item.get('buy_upper', 'N/A')})")
                lines.append(f"  止盈价: {item.get('sell_target', 'N/A')} {item.get('sell_target_pct', '')}")
                lines.append(f"  止损价: {item.get('sell_stop', 'N/A')} {item.get('sell_stop_pct', '')}")
                lines.append(f"  持有期: {item['hold_days']}  仓位: {item['position_pct']}")
                lines.append(f"  到期策略: {item['expire_rule']}")
            body_parts.append("\n".join(lines))

        if sell_alert_list:
            body_parts.append("\n⚠️ 持仓卖出提醒\n" + "=" * 40 + "\n" + format_sell_alerts_text(sell_alert_list))

        if not body_parts:
            body_parts.append("今日无AI推荐信号，无持仓卖出提醒。")

        subject = (
            f"QuantX每日信号 - {datetime.now().strftime('%Y-%m-%d')} | "
            f"情绪{sentiment_data.get('sentiment_score', 50)}分({sentiment_data.get('sentiment_level', '未知')}) · "
            f"AI精选{len(ai_result.get('action_list', []))}只 · 卖出提醒{len(sell_alert_list)}只"
        )
        ok = send_email(subject, "\n\n".join(body_parts))
        logger.info("邮件%s", "发送成功" if ok else "发送失败")

        logger.info("========== 每日任务完成 %s ==========", datetime.now().strftime("%H:%M:%S"))
        return {
            "ai_picks": ai_result.get("action_list", []),
            "sell_alerts": sell_alert_list,
            "sentiment": sentiment_data,
            "email_sent": ok,
            "auto_trade_result": auto_trade_result,
        }
