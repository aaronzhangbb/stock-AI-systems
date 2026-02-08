# -*- coding: utf-8 -*-
"""
大模型解释（阿里云百炼）
"""

import os
import json
import requests

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


def _get_api_key() -> str:
    return os.environ.get("BAILIAN_API_KEY") or config.BAILIAN_API_KEY


def explain_signal(stock_code: str, stock_name: str, signals: list, ml_score: float,
                   risk_level: str, risk_score: float, metrics: dict) -> str:
    """
    调用百炼生成解释文本
    """
    api_key = _get_api_key()
    if not api_key:
        return "未配置阿里云百炼 Key，无法生成AI解读。"

    strategies = "、".join([s.get('strategy_name', '') for s in signals]) or "策略信号"
    prompt = (
        f"你是一名量化投研助手，请用中文输出简洁的策略解读（150字以内）。\\n"
        f"股票: {stock_name}({stock_code})\\n"
        f"触发策略: {strategies}\\n"
        f"AI评分: {ml_score:.1f}/100\\n"
        f"风险等级: {risk_level}（{risk_score:.1f}/100）\\n"
        f"指标: 波动率{metrics.get('volatility', 0):.1f}%, "
        f"最大回撤{metrics.get('max_drawdown', 0):.1f}%, "
        f"夏普{metrics.get('sharpe', 0):.2f}, "
        f"胜率{metrics.get('win_rate', 0):.1f}%\\n"
        f"要求: 说明风险、提示注意事项，不要给出明确买卖指令。"
    )

    payload = {
        "model": config.BAILIAN_MODEL,
        "input": {"prompt": prompt},
        "parameters": {"result_format": "message"},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            config.BAILIAN_API_URL, headers=headers,
            data=json.dumps(payload), timeout=config.BAILIAN_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()

        # 兼容不同返回结构
        if "output" in data and "text" in data["output"]:
            return str(data["output"]["text"]).strip()
        if "output" in data and "choices" in data["output"]:
            return str(data["output"]["choices"][0]["message"]["content"]).strip()
        return json.dumps(data, ensure_ascii=False)

    except Exception as e:
        return f"AI解读失败：{e}"
