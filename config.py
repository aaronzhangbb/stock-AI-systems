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
INITIAL_CAPITAL = 1000000.0  # 初始资金（100万元，供AI虚拟盘50只股票用）
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
# 注意: 以下固定百分比仅作为「降级保底」使用
# AI策略的止盈止损/仓位由 ai_engine_v2._compute_trade_advice() 动态计算:
#   - 止损: ATR × 动态倍数(1.0~3.5), 结合趋势强度/波动率/支撑位
#   - 止盈: 多目标体系(T1/T2/T3), ATR驱动 + 阻力位自适应
#   - 仓位: 改进Kelly公式 × 波动率缩放 × AI置信度 (5%~35%)
#   - 持有: ATR标准化目标距离 + 动量加速度推算
STOP_LOSS_PCT = 0.08          # 降级止损 8% (ATR计算失败时使用)
TAKE_PROFIT_PCT = 0.20        # 降级止盈 20% (ATR计算失败时使用)
TRAILING_STOP_PCT = 0.10      # 降级追踪止损 10% (ATR计算失败时使用)

# Kelly仓位管理参数
KELLY_FRACTION = 0.5          # 使用半Kelly (保守系数)
TARGET_PORTFOLIO_VOL = 0.15   # 目标组合年化波动率 15%
MAX_SINGLE_POSITION = 0.35    # 单只最大仓位 35%
MIN_SINGLE_POSITION = 0.05    # 单只最小仓位 5%

# ==================== 推荐参数 ====================
RECOMMEND_TOP_N = 20          # 每日推荐股票数
RECOMMEND_MIN_SCORE = 55      # 最低推荐评分
STRATEGY_HOLD_DAYS = 5        # 策略验证持有天数

# ==================== AI自动交易参数 ====================
AUTO_SCORE_THRESHOLD = 80     # 自动买入最低AI评分
AUTO_MAX_POSITIONS = 50       # 最大同时持仓数 (买入所有推荐)
AUTO_SELL_URGENCY = 1         # 卖出紧急度阈值 (1=建议卖出+立即卖出, 2=仅立即卖出)
AUTO_USE_KELLY_SIZE = True    # 使用Kelly仓位 (False则用固定POSITION_RATIO)
AUTO_ENABLED = True           # 自动交易总开关

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
