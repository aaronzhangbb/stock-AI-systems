# -*- coding: utf-8 -*-
"""
邮件通知模块
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


def send_email(subject: str, body: str, to_list: list = None) -> bool:
    if not config.EMAIL_ENABLE:
        return False

    to_list = to_list or config.EMAIL_TO
    if not to_list:
        return False

    if not config.SMTP_PASSWORD:
        print("[邮件] 未配置 SMTP_PASSWORD")
        return False

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = config.SMTP_USER
    msg["To"] = ",".join(to_list)

    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP_SSL(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.login(config.SMTP_USER, config.SMTP_PASSWORD)
            server.sendmail(config.SMTP_USER, to_list, msg.as_string())
        return True
    except Exception as e:
        print(f"[邮件发送失败] {e}")
        return False


def build_daily_email(buy_recs: list, sell_alerts: list, stats: dict,
                      explain_texts: list = None) -> str:
    """
    构建每日闭环邮件（买入推荐 + 卖出提醒，合为一封）

    参数:
        buy_recs: 买入推荐列表（已按 composite_score 排序）
        sell_alerts: 卖出提醒列表（来自 position_monitor）
        stats: 扫描统计信息
        explain_texts: AI解读文本列表

    返回:
        str: 邮件正文
    """
    lines = []
    lines.append(f"==== 每日策略信号 {stats.get('scan_date', '')} ====")
    lines.append(f"扫描股票: {stats.get('scanned', 0)} | "
                 f"买入信号: {stats.get('buy_count', 0)} | "
                 f"卖出信号: {stats.get('sell_count', 0)} | "
                 f"耗时: {stats.get('duration', 0)}s")
    lines.append("")

    # ---- 买入推荐 ----
    lines.append(f"【买入推荐 Top {len(buy_recs)}】")
    lines.append("")

    if buy_recs:
        for i, sig in enumerate(buy_recs, 1):
            risk = sig.get('risk_level', '')
            strat_count = sig.get('strategy_count', 1)
            validated = sig.get('validated_count', 0)
            strategies = sig.get('strategies', sig.get('strategy_name', ''))
            lines.append(
                f"{i}. {sig.get('stock_name', '')}({sig.get('stock_code', '')}) | "
                f"综合评分 {sig.get('composite_score', 0):.0f} | 风险:{risk}"
            )
            lines.append(
                f"   策略共振: {strategies} ({strat_count}个策略, {validated}个验证通过)"
            )
            bp = sig.get('buy_price', 0)
            tp = sig.get('target_price', 0)
            sp = sig.get('stop_price', 0)
            if bp > 0:
                lines.append(
                    f"   建议买入价: {bp:.2f} | 目标价: {tp:.2f} | 止损价: {sp:.2f}"
                )
            reason = sig.get('reason', '')
            if reason:
                lines.append(f"   触发: {reason}")
            lines.append("")
    else:
        lines.append("   今日无推荐")
        lines.append("")

    # ---- AI解读 ----
    if explain_texts:
        lines.append("【AI策略解读】")
        lines.append("")
        for text in explain_texts:
            lines.append(f"  {text}")
        lines.append("")

    # ---- 卖出提醒 ----
    lines.append("【持仓卖出提醒】")
    lines.append("")

    if sell_alerts:
        for i, a in enumerate(sell_alerts, 1):
            pnl_sign = "+" if a.get('pnl_pct', 0) >= 0 else ""
            lines.append(
                f"{i}. {a.get('stock_name', '')}({a.get('stock_code', '')}) "
                f"买入价:{a.get('buy_price', 0):.2f} 现价:{a.get('current_price', 0):.2f} "
                f"({pnl_sign}{a.get('pnl_pct', 0):.1f}%)"
            )
            lines.append(f"   状态: {a.get('advice', '持有')}")
            for alert_msg in a.get('alerts', []):
                lines.append(f"   · {alert_msg}")
            lines.append("")
    else:
        lines.append("   无持仓需操作")
        lines.append("")

    lines.append("----")
    lines.append("QuantX 量化系统 · 仅供参考，不构成投资建议")

    return "\n".join(lines)


def build_signal_email(signals: list, stats: dict) -> str:
    """兼容旧接口"""
    lines = []
    lines.append("每日策略信号推送")
    lines.append(f"扫描股票: {stats.get('scanned', 0)}")
    lines.append(f"买入信号: {stats.get('buy_count', 0)}")
    lines.append(f"卖出信号: {stats.get('sell_count', 0)}")
    lines.append("")

    for sig in signals[:30]:
        lines.append(
            f"{sig.get('stock_code')} {sig.get('stock_name')} | "
            f"{sig.get('strategy_name')} | 强度{sig.get('strength')} | "
            f"AI评分{sig.get('ml_score', 0)} | 风险{sig.get('risk_level', '')}"
        )

    return "\n".join(lines)
