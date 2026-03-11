import json
import logging
import os
import tempfile
from datetime import datetime
from typing import Any, Callable, Optional


logger = logging.getLogger(__name__)

JSON_SCHEMA_VERSION = 1


def write_json_atomic(path: str, data: Any, *, default=str) -> None:
    """
    原子写入 JSON，避免写到一半时被其他读取方看到坏文件。
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=parent or None)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, default=default)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        raise


def load_json_safe(
    path: str,
    default: Any = None,
    *,
    validator: Optional[Callable[[Any], bool]] = None,
    log_prefix: str = "JSON",
) -> Any:
    """
    安全读取 JSON，读取失败或校验不通过时返回 default，并记录 warning。
    """
    if not os.path.exists(path):
        return default

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        logger.warning("[%s] 读取失败: %s | %s", log_prefix, path, exc)
        return default

    if validator and not validator(data):
        logger.warning("[%s] 结构校验失败: %s", log_prefix, path)
        return default

    return data


def write_pickle_atomic(path: str, obj: Any) -> None:
    """原子写入 pickle 文件，避免写到一半时被读取方看到损坏文件。"""
    import pickle
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".pkl", dir=parent or None)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(obj, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        raise


def build_ai_scores_payload(
    ai_df,
    *,
    pattern_scores: Optional[dict] = None,
    tf_scores: Optional[dict] = None,
    fusion_desc: str = "0.5*XGBoost + 0.3*Pattern + 0.2*Transformer",
) -> dict:
    pattern_scores = pattern_scores or {}
    tf_scores = tf_scores or {}
    today = datetime.now()
    return {
        "schema_version": JSON_SCHEMA_VERSION,
        "status": "ok",
        "scan_date": today.strftime("%Y-%m-%d"),
        "scan_time": today.strftime("%Y-%m-%d %H:%M:%S"),
        "total_scored": len(ai_df),
        "pattern_matched": len(pattern_scores),
        "transformer_matched": len(tf_scores),
        "fusion": fusion_desc,
        "score_distribution": {
            "above_90": int(len(ai_df[ai_df["final_score"] >= 90])) if "final_score" in ai_df.columns else 0,
            "above_80": int(len(ai_df[ai_df["final_score"] >= 80])) if "final_score" in ai_df.columns else 0,
        },
        "all_scores": ai_df.to_dict(orient="records"),
        "top50": ai_df.head(50).to_dict(orient="records"),
    }


def validate_ai_scores_payload(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("status") != "ok":
        return False
    if not data.get("scan_date") or not data.get("scan_time"):
        return False
    all_scores = data.get("all_scores")
    if not isinstance(all_scores, list):
        return False
    top50 = data.get("top50")
    if top50 is not None and not isinstance(top50, list):
        return False
    return True


def is_ai_scores_fresh(data: Any, *, expected_date: Optional[str] = None) -> bool:
    if not validate_ai_scores_payload(data):
        return False
    expected_date = expected_date or datetime.now().strftime("%Y-%m-%d")
    return data.get("scan_date") == expected_date


def build_failure_payload(message: str, **extra) -> dict:
    payload = {
        "schema_version": JSON_SCHEMA_VERSION,
        "status": "error",
        "message": message,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    payload.update(extra)
    return payload
