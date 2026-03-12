import logging
import os
import time
from datetime import datetime

import pandas as pd

import config
from src.data.market_sentiment import get_market_sentiment
from src.strategy.ai_engine_v2 import AIScorer
from src.data.data_cache import DataCache
from src.data.stock_pool import StockPool
from src.strategy.scanner import MarketScanner
from src.utils.state_store import (
    build_failure_payload,
    build_ai_scores_payload,
    is_ai_scores_fresh,
    load_json_safe,
    validate_ai_scores_payload,
    write_json_atomic,
)


logger = logging.getLogger(__name__)

AI_SCORES_PATH = os.path.join(config.DATA_ROOT, "ai_daily_scores.json")
AI_ACTION_LIST_PATH = os.path.join(config.DATA_ROOT, "ai_action_list.json")
STRATEGY_INSIGHTS_PATH = os.path.join(config.DATA_ROOT, "strategy_insights.json")


def _load_learned_params() -> dict:
    """
    从策略进化模块的学习报告中加载推荐参数, 供扫描/交易层使用。
    如果文件不存在或功能关闭, 返回空 dict (调用方 fallback 到默认值)。
    """
    if not getattr(config, 'SCAN_USE_LEARNED_PARAMS', True):
        return {}
    try:
        report = load_json_safe(STRATEGY_INSIGHTS_PATH, default={}, log_prefix="策略学习")
        if not report or report.get('status') != 'ok':
            return {}
        return report.get('optimal_params', {})
    except Exception:
        return {}


def _get_sentiment_weights() -> tuple[float, float, float, float]:
    """
    根据大盘情绪返回 (xgb_w, pattern_w, tf_w, threshold_adj) 的环境自适应权重。
    threshold_adj 为阈值调整量 (弱势市 +5, 正常 0, 强势 0)。
    """
    try:
        sentiment_path = os.path.join(config.DATA_ROOT, "market_sentiment.json")
        sentiment = load_json_safe(sentiment_path, default={}, log_prefix="情绪")
        score = sentiment.get('sentiment_score', 50)
        if score >= 70:
            return (0.40, 0.35, 0.25, 0)
        elif score <= 30:
            return (0.55, 0.25, 0.20, 5)
        else:
            return (0.50, 0.30, 0.20, 0)
    except Exception:
        return (0.50, 0.30, 0.20, 0)


def get_ai_scores_payload():
    return load_json_safe(
        AI_SCORES_PATH,
        default=None,
        validator=validate_ai_scores_payload,
        log_prefix="AI评分",
    )


def has_fresh_ai_scores(expected_date: str | None = None) -> bool:
    payload = get_ai_scores_payload()
    return is_ai_scores_fresh(payload, expected_date=expected_date)


def _save_action_list(action_list: list) -> None:
    payload = {
        "schema_version": 1,
        "status": "ok",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(action_list),
        "picks": action_list,
        "rules": {
            "buy": '在"买入价"附近挂限价单，不追高超过"最高可接受价"',
            "take_profit": '触及"止盈价"立即卖出',
            "stop_loss": '跌破"止损价"无条件卖出',
            "expire": "持有天数到期: 盈利→止损上移至成本价再观察1~2天; 亏损→收盘前无条件卖出",
            "position": "单只不超过建议仓位, 总持仓不超过3~5只",
        },
    }
    write_json_atomic(AI_ACTION_LIST_PATH, payload)


def _build_action_list(ai_df: pd.DataFrame) -> list:
    learned = _load_learned_params()
    threshold = learned.get('score_threshold', 85)
    threshold = max(threshold, 70)
    position_map = learned.get('recommended_position_map', {})

    strong = ai_df[ai_df["final_score"] >= threshold].head(5)
    action_list: list[dict] = []
    for _, row in strong.iterrows():
        close = row.get("close", 0)
        buy_p = row.get("buy_price")
        buy_up = row.get("buy_upper")
        sell_tgt = row.get("sell_target")
        sell_stp = row.get("sell_stop")
        hold = row.get("hold_days", "3~5天")
        rr = row.get("risk_reward")
        final = row.get("final_score", 0)

        pos = row.get("position_pct", "10%")
        if position_map:
            ai_s = row.get("ai_score", 0) or 0
            if ai_s >= 90 and '90+' in position_map:
                pos = f"{int(position_map['90+'] * 100)}%"
            elif ai_s >= 85 and '85-90' in position_map:
                pos = f"{int(position_map['85-90'] * 100)}%"
            elif ai_s >= 80 and '80-85' in position_map:
                pos = f"{int(position_map['80-85'] * 100)}%"
            elif '75-80' in position_map:
                pos = f"{int(position_map['75-80'] * 100)}%"

        expire = "若盈利：止损上移至成本价，再观察1~2天；若亏损：收盘无条件卖出"
        tgt_pct = f"+{(sell_tgt / close - 1) * 100:.1f}%" if pd.notna(sell_tgt) and close > 0 else ""
        stp_pct = f"-{(1 - sell_stp / close) * 100:.1f}%" if pd.notna(sell_stp) and close > 0 else ""

        action_list.append(
            {
                "stock_code": row.get("stock_code", ""),
                "stock_name": row.get("stock_name", ""),
                "board_name": row.get("board_name", ""),
                "close": round(close, 2),
                "buy_price": round(float(buy_p), 2) if pd.notna(buy_p) else round(close, 2),
                "buy_upper": round(float(buy_up), 2) if pd.notna(buy_up) else None,
                "sell_target": round(float(sell_tgt), 2) if pd.notna(sell_tgt) else None,
                "sell_target_pct": tgt_pct,
                "sell_stop": round(float(sell_stp), 2) if pd.notna(sell_stp) else None,
                "sell_stop_pct": stp_pct,
                "hold_days": str(hold),
                "risk_reward": round(float(rr), 1) if pd.notna(rr) else None,
                "position_pct": str(pos),
                "final_score": round(float(final), 1),
                "pattern_win_rate": round(float(row["pattern_win_rate"]), 1)
                if pd.notna(row.get("pattern_win_rate"))
                else None,
                "pattern_desc": row.get("pattern_desc", ""),
                "transformer_score": round(float(row["transformer_score"]), 1)
                if pd.notna(row.get("transformer_score"))
                else None,
                "expire_rule": expire,
            }
        )
    return action_list


def run_ai_super_scan(progress_callback=None, warmup_days: int = 730) -> dict:
    """
    统一 AI 三层扫描服务，供页面和批任务复用。
    """
    t0 = time.time()

    def _progress(stage: str, current: int | None = None, total: int | None = None, message: str = ""):
        if progress_callback:
            progress_callback(stage, current, total, message)

    try:
        try:
            get_market_sentiment(verbose=False)
        except Exception as exc:
            logger.warning("[AI扫描] 大盘情绪更新失败(不影响扫描): %s", exc)

        try:
            _progress("warmup", message="增量更新缓存")
            scanner = MarketScanner()
            warmup_result = scanner.warmup_cache(days=warmup_days)
            logger.info(
                "[AI扫描] 缓存已更新: 成功%d/共%d",
                warmup_result.get("success", 0),
                warmup_result.get("total", 0),
            )
        except Exception as exc:
            logger.warning("[AI扫描] 缓存更新失败(将使用已有缓存): %s", exc)

        cache = DataCache()
        pool = StockPool()
        scorer = AIScorer()

        def xgb_progress(current: int, total: int):
            _progress("xgb", current, total, f"XGBoost评分: {current}/{total}")

        ai_df = scorer.scan_market(cache, pool, top_n=100, progress_callback=xgb_progress)
        if ai_df.empty:
            logger.warning("[AI扫描] 无评分结果")
            payload = build_failure_payload("AI扫描无评分结果", scan_date=datetime.now().strftime("%Y-%m-%d"))
            write_json_atomic(AI_SCORES_PATH, payload)
            return {
                "status": "empty",
                "action_list": [],
                "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_scored": 0,
                "payload": payload,
            }

        pattern_scores = {}
        if config.ENABLE_HEAVY_MODEL:
            try:
                from src.strategy.pattern_engine import PatternEngine

                pe_path = os.path.join(config.DATA_ROOT, "pattern_engine.pkl")
                if os.path.exists(pe_path):
                    pe = PatternEngine.load(pe_path)
                    total_codes = len(ai_df)
                    for idx, code in enumerate(ai_df["stock_code"].tolist(), start=1):
                        try:
                            kdf = cache.load_kline(code)
                            if kdf is not None:
                                pr = pe.predict_single(kdf)
                                if pr and pr["is_valid"]:
                                    pattern_scores[code] = pr
                        except Exception as exc:
                            logger.warning("[AI扫描] 形态匹配失败 %s: %s", code, exc)
                        if idx % 20 == 0 or idx == total_codes:
                            _progress("pattern", idx, total_codes, f"形态匹配: {idx}/{total_codes}")
            except Exception as exc:
                logger.warning("[AI扫描] 形态匹配跳过: %s", exc)

        tf_scores = {}
        if config.ENABLE_HEAVY_MODEL:
            try:
                from src.strategy.transformer_engine import StockTransformer

                tf_path = os.path.join(config.DATA_ROOT, "transformer_model.pt")
                if os.path.exists(tf_path):
                    tf_engine = StockTransformer.load(tf_path)
                    total_codes = len(ai_df)
                    for idx, code in enumerate(ai_df["stock_code"].tolist(), start=1):
                        try:
                            kdf = cache.load_kline(code)
                            if kdf is not None:
                                ts = tf_engine.predict_single(kdf)
                                if ts is not None:
                                    tf_scores[code] = ts
                        except Exception as exc:
                            logger.warning("[AI扫描] Transformer评分失败 %s: %s", code, exc)
                        if idx % 20 == 0 or idx == total_codes:
                            _progress("transformer", idx, total_codes, f"Transformer: {idx}/{total_codes}")
            except Exception as exc:
                logger.warning("[AI扫描] Transformer跳过: %s", exc)

        xgb_w, pat_w, tf_w, _th_adj = _get_sentiment_weights()
        logger.info("[AI扫描] 环境权重: XGB=%.2f, Pattern=%.2f, TF=%.2f, 阈值调整=%+d",
                    xgb_w, pat_w, tf_w, _th_adj)

        pat_win_rates = []
        pat_descs = []
        pat_confs = []
        tf_score_list = []
        final_scores = []
        for _, row in ai_df.iterrows():
            code = row["stock_code"]
            xgb_score = row.get("ai_score", 50)

            pr = pattern_scores.get(code)
            if pr:
                pat_wr = pr["win_rate"]
                pat_win_rates.append(pat_wr)
                pat_descs.append(pr.get("pattern_desc", ""))
                pat_confs.append(pr.get("confidence"))
            else:
                pat_wr = 52.6
                pat_win_rates.append(None)
                pat_descs.append("")
                pat_confs.append(None)

            ts = tf_scores.get(code)
            if ts is not None:
                tf_s = ts
                tf_score_list.append(tf_s)
            else:
                tf_s = 52.9
                tf_score_list.append(None)

            fused = xgb_score * xgb_w + pat_wr * pat_w + tf_s * tf_w
            final_scores.append(round(fused, 1))

        ai_df["pattern_win_rate"] = pat_win_rates
        ai_df["pattern_desc"] = pat_descs
        ai_df["pattern_confidence"] = pat_confs
        ai_df["transformer_score"] = tf_score_list
        ai_df["final_score"] = final_scores
        ai_df = ai_df.sort_values("final_score", ascending=False).reset_index(drop=True)

        payload = build_ai_scores_payload(ai_df, pattern_scores=pattern_scores, tf_scores=tf_scores)
        write_json_atomic(AI_SCORES_PATH, payload)

        action_list = _build_action_list(ai_df)
        _save_action_list(action_list)

        elapsed = time.time() - t0
        logger.info(
            "[AI扫描] 完成: 评分%d只, 形态%d只, TF%d只, 推荐%d只, 耗时%.1fs",
            len(ai_df),
            len(pattern_scores),
            len(tf_scores),
            len(action_list),
            elapsed,
        )

        return {
            "status": "ok",
            "action_list": action_list,
            "scan_time": payload["scan_time"],
            "total_scored": len(ai_df),
            "payload": payload,
            "elapsed_seconds": round(elapsed, 1),
        }
    except Exception as exc:
        logger.error("[AI扫描] 扫描失败: %s", exc, exc_info=True)
        payload = build_failure_payload(str(exc), scan_date=datetime.now().strftime("%Y-%m-%d"))
        write_json_atomic(AI_SCORES_PATH, payload)
        return {
            "status": "error",
            "action_list": [],
            "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_scored": 0,
            "payload": payload,
            "error": str(exc),
        }
