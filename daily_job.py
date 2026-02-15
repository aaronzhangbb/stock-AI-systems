# -*- coding: utf-8 -*-
"""
每日收盘后闭环任务（精简版 - 纯AI策略）：
0. 采集大盘情绪（涨跌比/成交额/主力资金/北向/融资）
1. 增量更新本地缓存
2. AI三层超级策略扫描（XGBoost + 形态聚类 + Transformer）
3. 检查已持仓股的卖出时机
4. 一封邮件推送：情绪 + AI操作清单 + 卖出提醒
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.strategy.scanner import MarketScanner
from src.trading.position_monitor import check_all_manual_positions, get_sell_alerts, format_sell_alerts_text
from src.trading.paper_trading import PaperTradingAccount
from src.utils.email_notifier import send_email
from src.data.market_sentiment import get_market_sentiment, format_sentiment_text

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# ========== 日志配置 ==========
LOG_DIR = os.path.join(DATA_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"daily_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def run_ai_super_scan():
    """
    运行AI三层超级策略扫描，生成极简操作清单

    返回:
        {
            'action_list': [  # 操作清单（最多5只）
                {
                    'stock_code': '600340',
                    'stock_name': '华夏幸福',
                    'close': 1.68,
                    'buy_price': 1.49,       # 建议买入价
                    'buy_upper': 1.71,        # 最高可接受价
                    'sell_target': 1.81,      # 止盈价
                    'sell_stop': 1.55,        # 止损价
                    'hold_days': '3~5天',     # 持有周期
                    'final_score': 89.6,      # 综合评分
                    'expire_rule': '盈利则..., 亏损则...',  # 到期处理
                },
                ...
            ],
            'scan_time': '2026-02-07 15:35:00',
            'total_scored': 100,
        }
    """
    print("[AI扫描] === AI三层超级策略扫描开始 ===")
    t0 = time.time()

    from src.strategy.ai_engine_v2 import AIScorer
    from src.data.data_cache import DataCache
    from src.data.stock_pool import StockPool
    import pandas as pd
    import numpy as np

    cache = DataCache()
    pool = StockPool()

    # ---- 第1层: XGBoost评分 ----
    print("[AI扫描] [1/3] XGBoost评分全市场...")
    scorer = AIScorer()
    def xgb_prog(c, t):
        if c % 500 == 0:
            print(f"  XGBoost: {c}/{t} ({c/t*100:.0f}%)")
    ai_df = scorer.scan_market(cache, pool, top_n=100, progress_callback=xgb_prog)
    print(f"  XGBoost Top100 完成, 均分={ai_df['ai_score'].mean():.1f}" if not ai_df.empty else "  XGBoost无结果")

    if ai_df.empty:
        print("[AI扫描] 无评分结果, 跳过")
        return {'action_list': [], 'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'total_scored': 0}

    # ---- 第2层: 形态匹配 ----
    print("[AI扫描] [2/3] 形态匹配中...")
    pattern_scores = {}
    try:
        from src.strategy.pattern_engine import PatternEngine
        pe_path = os.path.join(DATA_DIR, 'pattern_engine.pkl')
        if os.path.exists(pe_path):
            pe = PatternEngine.load(pe_path)
            for code in ai_df['stock_code'].tolist():
                try:
                    kdf = cache.load_kline(code)
                    if kdf is not None:
                        pr = pe.predict_single(kdf)
                        if pr and pr['is_valid']:
                            pattern_scores[code] = pr
                except Exception:
                    pass
            print(f"  形态匹配: {len(pattern_scores)}/{len(ai_df)} 只")
    except Exception as e:
        print(f"  形态匹配跳过: {e}")

    # ---- 第3层: Transformer时序评分 ----
    print("[AI扫描] [3/3] Transformer时序评分中...")
    tf_scores = {}
    try:
        from src.strategy.transformer_engine import StockTransformer
        tf_path = os.path.join(DATA_DIR, 'transformer_model.pt')
        if os.path.exists(tf_path):
            tf_engine = StockTransformer.load(tf_path)
            for code in ai_df['stock_code'].tolist():
                try:
                    kdf = cache.load_kline(code)
                    if kdf is not None:
                        ts = tf_engine.predict_single(kdf)
                        if ts is not None:
                            tf_scores[code] = ts
                except Exception:
                    pass
            print(f"  Transformer: {len(tf_scores)}/{len(ai_df)} 只")
    except Exception as e:
        print(f"  Transformer跳过: {e}")

    # ---- 融合评分 ----
    print("[AI扫描] 融合计算...")
    pat_win_rates = []
    pat_descs = []
    pat_confs = []
    tf_score_list = []
    final_scores = []

    for _, row in ai_df.iterrows():
        code = row['stock_code']
        xgb_score = row.get('ai_score', 50)

        pr = pattern_scores.get(code)
        if pr:
            pat_wr = pr['win_rate']
            pat_win_rates.append(pat_wr)
            pat_descs.append(pr.get('pattern_desc', ''))
            pat_confs.append(pr.get('confidence'))
        else:
            pat_wr = 52.6
            pat_win_rates.append(None)
            pat_descs.append('')
            pat_confs.append(None)

        ts = tf_scores.get(code)
        if ts is not None:
            tf_s = ts
            tf_score_list.append(tf_s)
        else:
            tf_s = 52.9
            tf_score_list.append(None)

        fused = xgb_score * 0.5 + pat_wr * 0.3 + tf_s * 0.2
        final_scores.append(round(fused, 1))

    ai_df['pattern_win_rate'] = pat_win_rates
    ai_df['pattern_desc'] = pat_descs
    ai_df['pattern_confidence'] = pat_confs
    ai_df['transformer_score'] = tf_score_list
    ai_df['final_score'] = final_scores
    ai_df = ai_df.sort_values('final_score', ascending=False).reset_index(drop=True)

    # ---- 保存完整扫描结果 (供前端读取) ----
    output = {
        'scan_date': datetime.now().strftime('%Y-%m-%d'),
        'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_scored': len(ai_df),
        'pattern_matched': len(pattern_scores),
        'transformer_matched': len(tf_scores),
        'fusion': '0.5*XGBoost + 0.3*Pattern + 0.2*Transformer',
        'score_distribution': {
            'above_90': int(len(ai_df[ai_df['final_score'] >= 90])),
            'above_80': int(len(ai_df[ai_df['final_score'] >= 80])),
        },
        'top50': ai_df.head(50).to_dict(orient='records'),
    }
    score_path = os.path.join(DATA_DIR, 'ai_daily_scores.json')
    with open(score_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    # ---- 生成极简操作清单 (>=85分) ----
    strong = ai_df[ai_df['final_score'] >= 85].head(5)
    action_list = []

    for _, row in strong.iterrows():
        close = row.get('close', 0)
        buy_p = row.get('buy_price')
        buy_up = row.get('buy_upper')
        sell_tgt = row.get('sell_target')
        sell_stp = row.get('sell_stop')
        hold = row.get('hold_days', '3~5天')
        rr = row.get('risk_reward')
        pos = row.get('position_pct', '10%')
        final = row.get('final_score', 0)

        # 到期处理规则
        expire = "若盈利：止损上移至成本价，再观察1~2天；若亏损：收盘无条件卖出"

        # 计算涨跌幅
        tgt_pct = f"+{(sell_tgt/close-1)*100:.1f}%" if pd.notna(sell_tgt) and close > 0 else ""
        stp_pct = f"-{(1-sell_stp/close)*100:.1f}%" if pd.notna(sell_stp) and close > 0 else ""

        action_list.append({
            'stock_code': row.get('stock_code', ''),
            'stock_name': row.get('stock_name', ''),
            'board_name': row.get('board_name', ''),
            'close': round(close, 2),
            'buy_price': round(float(buy_p), 2) if pd.notna(buy_p) else round(close, 2),
            'buy_upper': round(float(buy_up), 2) if pd.notna(buy_up) else None,
            'sell_target': round(float(sell_tgt), 2) if pd.notna(sell_tgt) else None,
            'sell_target_pct': tgt_pct,
            'sell_stop': round(float(sell_stp), 2) if pd.notna(sell_stp) else None,
            'sell_stop_pct': stp_pct,
            'hold_days': str(hold),
            'risk_reward': round(float(rr), 1) if pd.notna(rr) else None,
            'position_pct': str(pos),
            'final_score': round(float(final), 1),
            'pattern_win_rate': round(float(row['pattern_win_rate']), 1) if pd.notna(row.get('pattern_win_rate')) else None,
            'pattern_desc': row.get('pattern_desc', ''),
            'transformer_score': round(float(row['transformer_score']), 1) if pd.notna(row.get('transformer_score')) else None,
            'expire_rule': expire,
        })

    # 保存操作清单
    action_output = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'count': len(action_list),
        'picks': action_list,
        'rules': {
            'buy': '在"买入价"附近挂限价单，不追高超过"最高可接受价"',
            'take_profit': '触及"止盈价"立即卖出',
            'stop_loss': '跌破"止损价"无条件卖出',
            'expire': '持有天数到期: 盈利→止损上移至成本价再观察1~2天; 亏损→收盘前无条件卖出',
            'position': '单只不超过建议仓位, 总持仓不超过3~5只',
        },
    }
    action_path = os.path.join(DATA_DIR, 'ai_action_list.json')
    with open(action_path, 'w', encoding='utf-8') as f:
        json.dump(action_output, f, ensure_ascii=False, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"[AI扫描] 完成! {len(action_list)} 只操作推荐, 耗时 {elapsed:.0f}秒")
    print(f"[AI扫描] 操作清单已保存: {action_path}")

    for item in action_list:
        print(f"  {item['stock_code']} {item['stock_name']}  "
              f"买:{item['buy_price']}  止盈:{item.get('sell_target','N/A')}({item.get('sell_target_pct','')})  "
              f"止损:{item.get('sell_stop','N/A')}({item.get('sell_stop_pct','')})  "
              f"评分:{item['final_score']}")

    return {
        'action_list': action_list,
        'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_scored': len(ai_df),
    }


def run_daily_job():
    """执行每日收盘闭环任务（精简版 - 纯AI策略）"""
    logger.info(f"========== 每日闭环任务启动 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==========")

    scanner = MarketScanner()
    account = PaperTradingAccount()

    # ========== 0. 采集大盘情绪 ==========
    logger.info("第0步: 采集大盘情绪指标...")
    sentiment_data = {}
    try:
        sentiment_data = get_market_sentiment(verbose=True)
        s_score = sentiment_data.get('sentiment_score', 50)
        s_level = sentiment_data.get('sentiment_level', '未知')
        logger.info(f"大盘情绪: {s_score}分 ({s_level})")
    except Exception as e:
        logger.warning(f"情绪采集失败: {e}（将使用默认中性值）")
        sentiment_data = {'sentiment_score': 50, 'sentiment_level': '未知', 'sentiment_advice': '情绪数据获取失败'}

    # ========== 1. 增量更新缓存 ==========
    logger.info("第1步: 增量更新缓存...")
    scanner.warmup_cache(days=730)
    logger.info("缓存更新完成")

    # ========== 2. AI三层超级策略扫描 ==========
    logger.info("第2步: AI三层超级策略扫描...")
    ai_result = {'action_list': []}
    try:
        ai_result = run_ai_super_scan()
    except Exception as e:
        logger.error(f"AI三层扫描失败: {e}", exc_info=True)

    # ========== 3. 检查持仓卖出时机 ==========
    logger.info("第3步: 检查持仓卖出...")
    all_positions = check_all_manual_positions(account)
    sell_alert_list = get_sell_alerts(all_positions)
    logger.info(f"持仓 {len(all_positions)} 只，需操作 {len(sell_alert_list)} 只")

    # ========== 4. 发送邮件（情绪 + AI操作清单 + 卖出提醒） ==========
    logger.info("第4步: 发送邮件...")

    # 构建情绪温度计文本
    sentiment_text = ""
    if sentiment_data:
        sentiment_text = format_sentiment_text(sentiment_data)

    # 构建AI操作清单文本
    ai_action_text = ""
    if ai_result.get('action_list'):
        lines = ["\n📋 AI超级策略 · 今日操作清单\n" + "=" * 40]
        for item in ai_result['action_list']:
            lines.append(f"\n{item['stock_code']} {item['stock_name']}  综合{item['final_score']}分")
            lines.append(f"  当前价: {item['close']}")
            lines.append(f"  买入价: {item['buy_price']}  (最高: {item.get('buy_upper', 'N/A')})")
            lines.append(f"  止盈价: {item.get('sell_target', 'N/A')} {item.get('sell_target_pct', '')}")
            lines.append(f"  止损价: {item.get('sell_stop', 'N/A')} {item.get('sell_stop_pct', '')}")
            lines.append(f"  持有期: {item['hold_days']}  仓位: {item['position_pct']}")
            lines.append(f"  到期策略: {item['expire_rule']}")
        lines.append("\n操作规则: 限价买入不追高, 触止盈立即卖, 破止损无条件出, 到期亏损清仓")
        ai_action_text = "\n".join(lines)

    # 构建卖出提醒文本
    sell_text = ""
    if sell_alert_list:
        sell_text = format_sell_alerts_text(sell_alert_list)

    # 组装邮件正文
    body_parts = []
    if sentiment_text:
        body_parts.append(sentiment_text)
    if ai_action_text:
        body_parts.append(ai_action_text)
    if sell_text:
        body_parts.append("\n⚠️ 持仓卖出提醒\n" + "=" * 40 + "\n" + sell_text)
    if not body_parts:
        body_parts.append("今日无AI推荐信号，无持仓卖出提醒。")

    body = "\n\n".join(body_parts)

    s_level = sentiment_data.get('sentiment_level', '')
    s_score = sentiment_data.get('sentiment_score', 50)
    n_ai_picks = len(ai_result.get('action_list', []))
    subject = (f"QuantX每日信号 - {datetime.now().strftime('%Y-%m-%d')} | "
               f"情绪{s_score}分({s_level}) · "
               f"AI精选{n_ai_picks}只 · 卖出提醒{len(sell_alert_list)}只")

    ok = send_email(subject, body)
    logger.info(f"邮件{'发送成功' if ok else '发送失败'}")

    logger.info(f"========== 每日任务完成 {datetime.now().strftime('%H:%M:%S')} ==========")

    return {
        'ai_picks': ai_result.get('action_list', []),
        'sell_alerts': sell_alert_list,
        'sentiment': sentiment_data,
        'email_sent': ok,
    }


if __name__ == "__main__":
    run_daily_job()
