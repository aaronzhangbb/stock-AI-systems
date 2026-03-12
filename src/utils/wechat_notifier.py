# -*- coding: utf-8 -*-
"""
微信推送通知模块 (PushPlus)

使用方法:
  1. 访问 https://www.pushplus.plus/ 用微信扫码登录
  2. 复制你的 Token
  3. 在 .env 中设置 PUSHPLUS_TOKEN=你的token

支持:
  - 纯文本消息
  - HTML 格式消息（带颜色/表格）
  - Markdown 格式消息
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


def send_wechat(title: str, content: str, *, template: str = "txt") -> bool:
    """
    通过 PushPlus 发送微信推送

    参数:
        title: 消息标题（微信通知栏可见）
        content: 消息正文
        template: 模板类型 "txt" / "html" / "markdown"

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
    }

    try:
        resp = requests.post(PUSHPLUS_API, json=payload, timeout=15)
        data = resp.json()
        if data.get("code") == 200:
            logger.info("[微信] 推送成功: %s", title[:40])
            return True
        else:
            logger.warning("[微信] 推送失败: %s", data.get("msg", resp.text[:200]))
            return False
    except Exception as exc:
        logger.warning("[微信] 推送异常: %s", exc)
        return False


def send_notification(title: str, content: str, *, template: str = "txt") -> dict:
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
