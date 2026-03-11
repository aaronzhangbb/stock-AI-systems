import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime


logger = logging.getLogger(__name__)


def make_run_id(prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


@contextmanager
def file_lock(lock_path: str, *, stale_seconds: int = 7200, metadata: dict | None = None):
    """
    基于独占创建的轻量运行锁，兼容 Windows。
    """
    lock_dir = os.path.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "created_ts": time.time(),
    }
    if metadata:
        payload.update(metadata)

    if os.path.exists(lock_path):
        try:
            with open(lock_path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
            created_ts = float(existing.get("created_ts", 0))
            if time.time() - created_ts > stale_seconds:
                os.remove(lock_path)
                logger.warning("[锁] 检测到过期锁，已清理: %s", lock_path)
            else:
                raise RuntimeError(f"发现运行中的任务锁: {lock_path}")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"发现运行中的任务锁: {lock_path} ({exc})") from exc

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        yield payload
    finally:
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except OSError as exc:
            logger.warning("[锁] 清理失败: %s | %s", lock_path, exc)
