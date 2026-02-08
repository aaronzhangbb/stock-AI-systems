# A股量化交易辅助系统 - 配置与运行文档

## 1. 环境信息

| 项目 | 值 |
|:---|:---|
| 项目目录 | `F:\project\my finance` |
| Python 版本 | 3.12.10 |
| 虚拟环境 | `.\venv\` |
| 数据缓存 | `.\data\stock_cache.db` |
| 模拟交易数据库 | `.\data\trading.db` |
| Skills 目录 | `C:\Users\Administrator\.claude\skills` |

---

## 2. 启动项目

### 方式一：双击启动（推荐）

直接双击项目根目录下的 `start.bat`。

### 方式二：命令行启动

```powershell
cd "F:\project\my finance"
.\venv\Scripts\Activate.ps1
streamlit run app.py --server.port 8501 --server.headless true
```

启动后浏览器访问：**http://localhost:8501**

停止服务：终端按 `Ctrl+C`

---

## 3. 依赖安装

如果重新安装或新增依赖，使用阿里云镜像源：

```powershell
cd "F:\project\my finance"
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

---

## 4. 项目结构

```
F:\project\my finance\
├── app.py                    # Streamlit 主界面入口
├── config.py                 # 全局配置（策略参数、费率、股票池）
├── start.bat                 # 一键启动脚本
├── requirements.txt          # Python 依赖清单
├── AGENTS.md                 # Claude Skills 注册清单
├── SETUP.md                  # 本文档
│
├── src/                      # 源代码
│   ├── data/
│   │   ├── data_fetcher.py   # 数据获取（AkShare + 本地缓存）
│   │   └── data_cache.py     # SQLite 缓存管理
│   ├── strategy/
│   │   └── strategy.py       # 策略引擎（双均线 + RSI）
│   ├── backtest/
│   │   └── backtester.py     # 回测系统
│   ├── trading/
│   │   └── paper_trading.py  # 模拟交易账户
│   └── utils/
│       └── notifier.py       # Windows 桌面弹窗通知
│
├── data/                     # 数据目录
│   ├── stock_cache.db        # 历史行情缓存（自动生成）
│   └── trading.db            # 模拟交易记录（自动生成）
│
├── venv/                     # Python 虚拟环境
└── logs/                     # 日志目录
```

---

## 5. 核心配置项（config.py）

### 策略参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `MA_SHORT` | 5 | 短期均线天数 |
| `MA_LONG` | 20 | 长期均线天数 |
| `RSI_PERIOD` | 14 | RSI 计算周期 |
| `RSI_OVERBOUGHT` | 80 | 超买阈值（高于此不买入） |
| `RSI_OVERSOLD` | 20 | 超卖阈值（低于此不卖出） |

### 交易参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `INITIAL_CAPITAL` | 100,000 | 模拟盘初始资金（元） |
| `COMMISSION_RATE` | 0.0003 | 佣金费率（万三） |
| `MIN_COMMISSION` | 5.0 | 最低佣金（元） |
| `STAMP_TAX_RATE` | 0.001 | 印花税（千一，仅卖出） |
| `POSITION_RATIO` | 0.3 | 单次买入仓位比例（30%） |

### AI策略与风控参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `AI_PREDICT_HORIZON` | 5 | 预测未来N天涨跌 |
| `AI_MIN_TRAIN_SAMPLES` | 80 | 最小训练样本数 |
| `STOP_LOSS_PCT` | 0.08 | 止损比例 |
| `TAKE_PROFIT_PCT` | 0.20 | 止盈比例 |
| `TRAILING_STOP_PCT` | 0.10 | 追踪止损比例 |

### 邮件通知参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `EMAIL_ENABLE` | True | 是否启用邮件通知 |
| `SMTP_HOST` | smtp.qq.com | SMTP服务器 |
| `SMTP_PORT` | 465 | SMTP端口 |
| `SMTP_USER` | 你的QQ邮箱 | 发件邮箱 |
| `SMTP_PASSWORD` | 空 | QQ邮箱授权码 |
| `EMAIL_TO` | 收件人列表 | 邮件接收地址 |

### 大模型参数（阿里云百炼）

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `BAILIAN_API_KEY` | 空 | 百炼API Key（可用环境变量 `BAILIAN_API_KEY`） |
| `BAILIAN_API_URL` | DashScope URL | 百炼API地址 |
| `BAILIAN_MODEL` | qwen-plus | 默认模型 |
| `BAILIAN_TIMEOUT` | 20 | 超时时间（秒） |

### 股票观察池

在 `config.py` 的 `WATCHLIST` 字典中添加或删除：

```python
WATCHLIST = {
    "600519": "贵州茅台",
    "000858": "五粮液",
    "601318": "中国平安",
    # 添加新股票...
}
```

---

## 6. 每日收盘任务（自动推送）

项目提供 `daily_job.py`，用于：
1) 增量更新缓存  
2) 全市场扫描  
3) 邮件推送  

可使用 Windows 任务计划程序每日收盘后执行：

```powershell
cd "F:\project\my finance"
.\venv\Scripts\Activate.ps1
python daily_job.py
```

---

## 7. Git 代理配置

系统 git 原有代理 `127.0.0.1:17890`，已清除。如需恢复：

```powershell
# 设置代理
git config --global http.proxy http://127.0.0.1:17890
git config --global https.proxy http://127.0.0.1:17890

# 清除代理
git config --global --unset http.proxy
git config --global --unset https.proxy
```

---

## 8. 常见问题

### Q: 启动后浏览器没有自动打开？

手动访问 http://localhost:8501

### Q: 数据获取失败？

检查网络连接。AkShare 从东方财富获取数据，需要能访问外网。

### Q: 如何清除缓存数据？

删除 `data/stock_cache.db` 文件，下次启动时会自动重新创建。

### Q: 如何重置模拟账户？

在系统界面「模拟交易」页面点击「重置模拟账户」按钮，或删除 `data/trading.db` 文件。
