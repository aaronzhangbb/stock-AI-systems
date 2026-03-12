# -*- coding: utf-8 -*-
"""
微信推送通知模块 (PushPlus)

使用方法:
  1. 访问 https://www.pushplus.plus/ 用微信扫码登录
  2. 复制你的 Token
  3. 在 .env 中设置 PUSHPLUS_TOKEN=你的token

支持:
  - 纯文本消息 (template=txt)
  - HTML 格式消息 (template=html)，带颜色/表格
"""

import logging
import os
import sys

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config

logger = logging.getLogger(__name__)

PUSHPLUS_API = "http://www.pushplus.plus/send"


def _get_token() -> str:
    return os.getenv("PUSHPLUS_TOKEN", getattr(config, "PUSHPLUS_TOKEN", ""))


def send_wechat(title: str, content: str, *, template: str = "html") -> bool:
    """
    通过 PushPlus 发送微信推送

    参数:
        title: 消息标题（微信通知栏可见，最长100字）
        content: 消息正文（支持 HTML/txt/markdown）
        template: 模板类型 "html" / "txt" / "markdown"

    返回:
        bool: 是否发送成功
    """
    token = _get_token()
    if not token:
        logger.warning("[微信] 未配置 PUSHPLUS_TOKEN，跳过推送")
        return False

    payload = {
        "token": token,
        "title": title[:100],
        "content": content,
        "template": template,
        "channel": "wechat",
    }

    try:
        resp = requests.post(PUSHPLUS_API, json=payload, timeout=15)
        data = resp.json()
        if data.get("code") == 200:
            logger.info("[微信] 推送成功: %s", title[:40])
            return True
        else:
            logger.warning("[微信] 推送失败(code=%s): %s", data.get("code"), data.get("msg", resp.text[:200]))
            return False
    except Exception as exc:
        logger.warning("[微信] 推送异常: %s", exc)
        return False


def _build_html_card(title: str, rows: list[tuple[str, str]], *, color: str = "#1a73e8") -> str:
    """
    构建一个 HTML 卡片块

    参数:
        title: 卡片标题
        rows: [(标签, 值), ...]
        color: 标题颜色
    """
    row_html = ""
    for label, value in rows:
        row_html += f'<tr><td style="color:#888;padding:4px 8px;white-space:nowrap;">{label}</td><td style="padding:4px 8px;font-weight:600;">{value}</td></tr>'
    return f"""
<div style="background:#fff;border-radius:8px;padding:16px;margin-bottom:12px;border-left:4px solid {color};">
<div style="font-size:16px;font-weight:700;color:{color};margin-bottom:8px;">{title}</div>
<table style="font-size:14px;color:#333;">{row_html}</table>
</div>"""


def format_buy_html(stock_name: str, stock_code: str, price: float, shares: int,
                    ai_score: float, buy_lower: float, buy_upper: float,
                    sell_target: float, sell_stop: float, time_str: str) -> str:
    return _build_html_card(
        f"狙击买入 {stock_name}({stock_code})",
        [
            ("成交价", f"¥{price:.2f} x {shares}股"),
            ("AI评分", f"{ai_score:.1f}"),
            ("目标区间", f"¥{buy_lower:.2f} ~ ¥{buy_upper:.2f}"),
            ("止盈", f"¥{sell_target:.2f}" if sell_target else "-"),
            ("止损", f"¥{sell_stop:.2f}" if sell_stop else "-"),
            ("时间", time_str),
        ],
        color="#e53935",
    )


def format_sell_html(stock_name: str, stock_code: str, price: float, shares: int,
                     pnl: float, pnl_pct: float, trigger: str, time_str: str) -> str:
    pnl_color = "#4caf50" if pnl >= 0 else "#e53935"
    return _build_html_card(
        f"狙击卖出 {stock_name}({stock_code})",
        [
            ("卖出价", f"¥{price:.2f} x {shares}股"),
            ("盈亏", f'<span style="color:{pnl_color};">¥{pnl:+,.0f} ({pnl_pct:+.1f}%)</span>'),
            ("触发", trigger),
            ("时间", time_str),
        ],
        color="#4caf50" if pnl >= 0 else "#e53935",
    )


def format_daily_summary_html(sentiment_score: int, sentiment_level: str,
                              n_buy: int, n_sell: int, n_hold_alerts: int,
                              total_equity: float, total_pnl_pct: float) -> str:
    pnl_color = "#4caf50" if total_pnl_pct >= 0 else "#e53935"
    return f"""
<div style="background:#fff;border-radius:8px;padding:16px;margin-bottom:12px;border-left:4px solid #1a73e8;">
<div style="font-size:16px;font-weight:700;color:#1a73e8;margin-bottom:8px;">每日交易摘要</div>
<table style="font-size:14px;color:#333;">
<tr><td style="color:#888;padding:4px 8px;">市场情绪</td><td style="padding:4px 8px;font-weight:600;">{sentiment_score}分 ({sentiment_level})</td></tr>
<tr><td style="color:#888;padding:4px 8px;">今日操作</td><td style="padding:4px 8px;font-weight:600;">买入{n_buy}只 · 卖出{n_sell}只 · 提醒{n_hold_alerts}只</td></tr>
<tr><td style="color:#888;padding:4px 8px;">总资产</td><td style="padding:4px 8px;font-weight:600;">¥{total_equity:,.0f}</td></tr>
<tr><td style="color:#888;padding:4px 8px;">总收益</td><td style="padding:4px 8px;font-weight:600;"><span style="color:{pnl_color};">{total_pnl_pct:+.2f}%</span></td></tr>
</table>
</div>"""


def send_notification(title: str, content: str, *, template: str = "html") -> dict:
    """
    统一通知入口：同时发邮件 + 微信（各自独立，互不影响）

    返回: {"email": bool, "wechat": bool}
    """
    results = {"email": False, "wechat": False}

    try:
        from src.utils.email_notifier import send_email
        results["email"] = send_email(title, content)
    except Exception as exc:
        logger.warning("[通知] 邮件发送失败: %s", exc)

    try:
        results["wechat"] = send_wechat(title, content, template=template)
    except Exception as exc:
        logger.warning("[通知] 微信推送失败: %s", exc)

    return results
