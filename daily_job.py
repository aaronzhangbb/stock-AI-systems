# -*- coding: utf-8 -*-
"""
每日收盘后闭环任务：
1. 增量更新本地缓存
2. 全市场扫描 + AI评分 + 策略验证
3. 生成买入推荐 Top N（含买入价/目标价/止损价）
4. 检查已持仓股的卖出时机
5. 一封邮件推送：买入推荐 + 卖出提醒
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import config
from src.strategy.scanner import MarketScanner
from src.strategy.strategy_discovery import train_model
from src.trading.position_monitor import check_all_manual_positions, get_sell_alerts, format_sell_alerts_text
from src.trading.paper_trading import PaperTradingAccount
from src.utils.email_notifier import send_email, build_daily_email
from src.utils.llm_explainer import explain_signal


def run_daily_job():
    """执行每日收盘闭环任务"""
    print(f"[任务] ========== 每日闭环任务启动 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==========")

    scanner = MarketScanner()
    account = PaperTradingAccount()

    # ========== 1. 增量更新缓存 ==========
    print("[任务] 第1步: 增量更新缓存...")
    scanner.warmup_cache(days=730)
    print("[任务] 缓存更新完成")

    # ========== 1.5 训练/更新数据学习策略 ==========
    print("[任务] 第1.5步: 更新数据学习策略...")
    try:
        discovery = train_model(max_stocks=200, force=False)
        n_rules = len(discovery.get('learned_rules', []))
        print(f"[任务] 数据学习策略: {n_rules} 条规则可用")
    except Exception as e:
        print(f"[任务] 数据学习策略训练失败: {e}（将继续使用技术策略）")

    # ========== 2. 全市场扫描 ==========
    print("[任务] 第2步: 全市场扫描...")
    result = scanner.scan_market(days=730)

    stats = result.get('stats', {})
    buy_signals = result.get('buy_signals', [])
    sell_signals = result.get('sell_signals', [])
    print(f"[任务] 扫描完成: 买入={len(buy_signals)} 卖出={len(sell_signals)}")

    # ========== 3. 生成买入推荐 Top N（多策略共振） ==========
    print("[任务] 第3步: 生成买入推荐（多策略共振）...")

    # 使用聚合推荐（2+ 策略共振 + 验证通过）
    buy_recs = result.get('buy_recommendations', [])
    # 额外筛选综合评分 >= 最低推荐分
    min_score = config.RECOMMEND_MIN_SCORE
    buy_recs = [r for r in buy_recs if r.get('composite_score', 0) >= min_score]
    buy_recs = buy_recs[:config.RECOMMEND_TOP_N]
    print(f"[任务] 多策略共振推荐: {len(buy_recs)} 只")

    # ========== 4. AI解读（Top5 推荐） ==========
    print("[任务] 第4步: 生成AI解读...")
    explain_texts = []
    for rec in buy_recs[:5]:
        try:
            # 构造信号列表供 explain_signal 使用
            fake_sigs = [{'strategy_name': rec.get('strategies', '')}]
            text = explain_signal(
                rec['stock_code'], rec['stock_name'], fake_sigs,
                rec.get('composite_score', rec.get('ml_score', 0)),
                rec.get('risk_level', ''),
                rec.get('risk_score', 0),
                {}
            )
            explain_texts.append(f"{rec['stock_name']}({rec['stock_code']}): {text}")
        except Exception as e:
            explain_texts.append(f"{rec['stock_name']}({rec['stock_code']}): 解读生成失败 - {e}")

    # ========== 5. 检查持仓卖出时机 ==========
    print("[任务] 第5步: 检查持仓卖出...")
    all_positions = check_all_manual_positions(account)
    sell_alert_list = get_sell_alerts(all_positions)
    print(f"[任务] 持仓 {len(all_positions)} 只，需操作 {len(sell_alert_list)} 只")

    # ========== 6. 发送邮件 ==========
    print("[任务] 第6步: 发送邮件...")
    body = build_daily_email(
        buy_recs=buy_recs,
        sell_alerts=sell_alert_list,
        stats=stats,
        explain_texts=explain_texts if explain_texts else None,
    )

    subject = f"每日策略信号 - {datetime.now().strftime('%Y-%m-%d')} | " \
              f"推荐{len(buy_recs)}只 · 卖出提醒{len(sell_alert_list)}只"

    ok = send_email(subject, body)
    print(f"[任务] 邮件{'发送成功' if ok else '发送失败'}")

    print(f"[任务] ========== 每日任务完成 {datetime.now().strftime('%H:%M:%S')} ==========")

    return {
        'stats': stats,
        'buy_recs': buy_recs,
        'sell_alerts': sell_alert_list,
        'email_sent': ok,
    }


if __name__ == "__main__":
    run_daily_job()
