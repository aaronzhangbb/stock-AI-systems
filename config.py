"""
A股量化交易辅助系统 - 全局配置文件

配置优先级: .env 环境变量 > 本文件默认值
机器差异（性能参数、开关、路径）通过 .env 覆盖，不要直接修改本文件。
参见 DEV_RULES.md 和 .env.example
"""

import os


def _load_dotenv(dotenv_path: str = ".env"):
    """
    轻量加载 .env 到进程环境变量（不覆盖已存在环境变量）
    避免引入额外依赖，兼容 Windows 本地运行。
    """
    if not os.path.exists(dotenv_path):
        return
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as exc:
        import sys
        print(f"[config] 警告: 加载 .env 文件失败: {exc}", file=sys.stderr)


_load_dotenv()


def _env_bool(key: str, default: str = "true") -> bool:
    return os.getenv(key, default).strip().lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int = 0) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_float(key: str, default: float = 0.0) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


# ==================== 运行实例标识 ====================
INSTANCE_NAME = os.getenv("INSTANCE_NAME", "default")

# ==================== 数据根目录（核心：所有数据路径的基础） ====================
DATA_ROOT = os.getenv("DATA_ROOT", "data")
os.makedirs(DATA_ROOT, exist_ok=True)

LOG_ROOT = os.path.join(DATA_ROOT, "logs")
os.makedirs(LOG_ROOT, exist_ok=True)

# ==================== 策略参数 ====================
MA_SHORT = 5
MA_LONG = 20

RSI_PERIOD = 14
RSI_OVERBOUGHT = 80
RSI_OVERSOLD = 20

# ==================== 模拟账户参数 ====================
INITIAL_CAPITAL = _env_float("INITIAL_CAPITAL", 1000000.0)
COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0
STAMP_TAX_RATE = 0.001
POSITION_RATIO = 0.3

# ==================== 数据参数 ====================
DEFAULT_PERIOD = "daily"
DEFAULT_ADJUST = "qfq"
HISTORY_DAYS = _env_int("HISTORY_DAYS", 365)

# ==================== 提醒参数 ====================
ENABLE_NOTIFICATION = _env_bool("ENABLE_NOTIFICATION", "true")

# ==================== 邮件通知参数 ====================
EMAIL_ENABLE = _env_bool("EMAIL_ENABLE", "true")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.qq.com")
SMTP_PORT = _env_int("SMTP_PORT", 465)
SMTP_USER = os.getenv("SMTP_USER", "360928477@qq.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
_email_to_env = os.getenv("EMAIL_TO", "")
if _email_to_env.strip():
    EMAIL_TO = [x.strip() for x in _email_to_env.split(",") if x.strip()]
else:
    EMAIL_TO = ["360928477@qq.com"]

# ==================== 大模型参数（阿里云百炼） ====================
BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY", "")
BAILIAN_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
BAILIAN_MODEL = "qwen-plus"
BAILIAN_TIMEOUT = 20

# ==================== AI策略参数 ====================
AI_PREDICT_HORIZON = 5
AI_MIN_TRAIN_SAMPLES = 80

# ==================== 风控参数 ====================
# 以下固定百分比仅作为「降级保底」使用
# AI策略的止盈止损/仓位由 ai_engine_v2._compute_trade_advice() 动态计算
STOP_LOSS_PCT = 0.08
TAKE_PROFIT_PCT = 0.20
TRAILING_STOP_PCT = 0.10

KELLY_FRACTION = 0.5
TARGET_PORTFOLIO_VOL = 0.15
MAX_SINGLE_POSITION = 0.35
MIN_SINGLE_POSITION = 0.05

# ==================== 推荐参数 ====================
RECOMMEND_TOP_N = 20
RECOMMEND_MIN_SCORE = 55
STRATEGY_HOLD_DAYS = 5

# ==================== AI自动交易参数（可通过 .env 覆盖） ====================
AUTO_SCORE_THRESHOLD = _env_int("AUTO_SCORE_THRESHOLD", 75)
AUTO_MAX_POSITIONS = _env_int("AUTO_MAX_POSITIONS", 50)
AUTO_SELL_URGENCY = _env_int("AUTO_SELL_URGENCY", 1)
AUTO_USE_KELLY_SIZE = _env_bool("AUTO_USE_KELLY_SIZE", "true")
AUTO_ENABLED = _env_bool("AUTO_ENABLED", "true")
AI_SELL_SCORE_DROP = _env_int("AI_SELL_SCORE_DROP", 15)

# ==================== 性能参数（低配机/高配机/服务器差异化） ====================
SCAN_LIMIT = _env_int("SCAN_LIMIT", 0)                        # 扫描股票上限, 0=全量
SCAN_WORKERS = _env_int("SCAN_WORKERS", 4)                     # 扫描/预热并发线程数
ENABLE_HEAVY_MODEL = _env_bool("ENABLE_HEAVY_MODEL", "true")   # 是否启用重模型 (Transformer等)

# ==================== 数据库路径（基于 DATA_ROOT） ====================
DB_PATH = os.path.join(DATA_ROOT, "trading.db")

# ==================== 常用股票池（可自定义）====================
WATCHLIST = {
    "600519": "贵州茅台",
    "000858": "五粮液",
    "601318": "中国平安",
    "000001": "平安银行",
    "600036": "招商银行",
    "000333": "美的集团",
    "601012": "隆基绿能",
    "300750": "宁德时代",
}
