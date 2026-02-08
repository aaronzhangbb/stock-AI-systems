"""
A股量化交易辅助系统 - 全局配置文件
"""

# ==================== 策略参数 ====================
# 双均线策略
MA_SHORT = 5        # 短期均线天数
MA_LONG = 20        # 长期均线天数

# RSI 过滤参数
RSI_PERIOD = 14     # RSI 计算周期
RSI_OVERBOUGHT = 80  # 超买阈值（高于此值不买入）
RSI_OVERSOLD = 20    # 超卖阈值（低于此值不卖出）

# ==================== 模拟账户参数 ====================
INITIAL_CAPITAL = 100000.0   # 初始资金（10万元）
COMMISSION_RATE = 0.0003     # 佣金费率（万三）
MIN_COMMISSION = 5.0         # 最低佣金（5元）
STAMP_TAX_RATE = 0.001       # 印花税（千一，仅卖出时收取）
POSITION_RATIO = 0.3         # 单次买入仓位比例（30%）

# ==================== 数据参数 ====================
DEFAULT_PERIOD = "daily"     # 默认K线周期
DEFAULT_ADJUST = "qfq"       # 默认复权方式：前复权
HISTORY_DAYS = 365           # 默认拉取历史天数

# ==================== 提醒参数 ====================
ENABLE_NOTIFICATION = True   # 是否启用桌面弹窗提醒

# ==================== 邮件通知参数 ====================
EMAIL_ENABLE = True
SMTP_HOST = "smtp.qq.com"
SMTP_PORT = 465
SMTP_USER = "360928477@qq.com"
SMTP_PASSWORD = "mkapexeuhqosbghg"  # QQ邮箱需要授权码
EMAIL_TO = ["360928477@qq.com"]

# ==================== 大模型参数（阿里云百炼） ====================
BAILIAN_API_KEY = ""  # 建议通过环境变量 BAILIAN_API_KEY 配置
BAILIAN_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
BAILIAN_MODEL = "qwen-plus"
BAILIAN_TIMEOUT = 20

# ==================== AI策略参数 ====================
AI_PREDICT_HORIZON = 5        # 预测未来N天涨跌
AI_MIN_TRAIN_SAMPLES = 80

# ==================== 风控参数 ====================
STOP_LOSS_PCT = 0.08          # 止损 8%
TAKE_PROFIT_PCT = 0.20        # 止盈 20%
TRAILING_STOP_PCT = 0.10      # 追踪止损 10%

# ==================== 推荐参数 ====================
RECOMMEND_TOP_N = 20          # 每日推荐股票数
RECOMMEND_MIN_SCORE = 55      # 最低推荐评分
STRATEGY_HOLD_DAYS = 5        # 策略验证持有天数

# ==================== 数据库路径 ====================
DB_PATH = "data/trading.db"  # SQLite 数据库路径

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
