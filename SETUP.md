# A股量化交易辅助系统 - 搭建与运行指南

---

## 1. 已有项目的电脑同步最新改造

如果这台电脑上已经有项目（之前 clone 过），只需要同步代码并配置 `.env`：

```powershell
# 进入项目目录
cd "你的项目目录\stock-AI-systems"

# 如果需要代理
$env:HTTPS_PROXY="http://127.0.0.1:17890"

# 拉取最新代码（切到最新的功能分支）
git fetch
git checkout feature/config-refactor
git pull

# 创建本机配置文件（只需做一次，以后不用再做）
Copy-Item .env.example .env
```

然后编辑 `.env`，按这台电脑的性能设置参数（参见上面第三步的推荐配置）。

如果虚拟环境依赖有更新：

```powershell
.\venv\Scripts\Activate.ps1
$env:PYTHONUTF8=1
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

完成后直接启动即可：

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py --server.port 8501 --server.headless true
```

---

## 2. 本机（低配电脑）快速启动

```powershell
cd "c:\xunqing\project\finace dock system\stock-AI-systems"
.\venv\Scripts\Activate.ps1
streamlit run app.py --server.port 8501 --server.headless true
```

浏览器访问：**http://localhost:8501**

> 也可以直接双击 `start.bat` 启动。

---

## 3. 本轮改造后的关键变化

### 3.1 自动交易现在会校验 AI 评分是否为“当天最新”

当前版本新增了 **AI 评分新鲜度校验**：

- `ai_daily_scores.json` 必须结构完整
- `scan_date` 必须是当天
- 扫描失败时不会再偷偷沿用旧评分去自动买入

这意味着：

- **自动卖出仍可执行**
- **自动买入会在评分无效时被跳过**

### 3.2 第一次升级到这个版本后，需要先重新跑一次 AI 扫描

如果你的 `data/ai_daily_scores.json` 是旧版本生成的，它没有新的 `status / schema_version / all_scores` 字段。

所以升级后第一次使用时，建议先执行一次：

```powershell
.\venv\Scripts\Activate.ps1
python daily_job.py
```

或者在前端点击：

- `每日信号` → `运行AI超级策略扫描`
- 或 `模拟交易` → `AI一键执行交易`

这样系统会生成新版评分文件，后续自动交易才能正常自动买入。

### 3.3 AI 一键执行交易的流程已统一

当前流程为：

```text
重新AI扫描最新推荐 → 检查持仓卖出 → 筛选新标的买入 → 保存资产快照
```

其中：

- 卖出检查会使用最新 AI 评分
- “AI未确认，仅提醒”的卖出信号会显示在前端的“持仓提醒”卡片中

### 3.4 增加了运行锁，避免重复执行

当前版本会对以下任务加锁：

- `daily_job`
- `AI一键执行交易`

目的是避免：

- 重复点击按钮
- 定时任务和手动任务重叠
- 同时触发导致重复买卖

### 3.5 本地脚本已改为自适应项目目录

以下脚本不再写死本机路径：

- `run_daily.bat`
- `run_intraday.bat`
- `setup_scheduler.ps1`

现在可以直接根据脚本所在目录运行，更适合双机开发。

---

## 4. 日常开发（已有环境）

### 切换到这台电脑继续开发

```powershell
cd "你的项目目录\stock-AI-systems"
.\venv\Scripts\Activate.ps1

# 拉取最新代码
git fetch
git checkout 你正在开发的分支
git pull
```

### 离开这台电脑前

```powershell
git add .
git commit -m "feat: 进度描述"

# 推送到远程（需代理时加上）
$env:HTTPS_PROXY="http://127.0.0.1:17890"
git push
```

### 新功能开发

```powershell
git checkout master
git pull origin master
git checkout -b feature/新功能名字

# ... 开发 ...

git add .
git commit -m "feat: 功能描述"
git push -u origin feature/新功能名字
```

---

## 5. 项目结构

```
stock-AI-systems/
├── app.py                    # Streamlit 主界面入口
├── config.py                 # 全局配置（优先读 .env 环境变量）
├── daily_job.py              # 每日收盘任务（缓存更新 + AI扫描 + 邮件推送）
├── retrain_all.py            # 三层模型重训练
├── start.bat                 # 一键启动脚本
├── requirements.txt          # Python 依赖清单
├── DEV_RULES.md              # 多机开发与部署规范
├── SETUP.md                  # 本文档
│
├── .env.example              # 环境变量模板（提交 Git）
├── .env                      # 本机实际配置（不提交 Git）
│
├── src/                      # 源代码
│   ├── data/                 # 数据获取、缓存、情绪、板块
│   ├── strategy/             # AI策略、扫描、模式识别
│   ├── trading/              # 模拟交易、持仓监控、绩效、策略学习
│   ├── services/             # 统一服务层（扫描/自动交易/每日任务）
│   ├── ui/                   # 前端渲染辅助模块
│   ├── backtest/             # 回测系统
│   └── utils/                # 通知、原子写入、运行锁、DB工具
│
├── data/                     # 运行数据目录（不提交 Git，自动生成）
│   ├── trading.db            # 模拟交易数据库
│   ├── stock_cache.db        # K线缓存
│   ├── stock_pool.db         # 股票池
│   ├── ai_daily_scores.json  # AI评分
│   ├── market_sentiment.json # 市场情绪
│   ├── xgb_v2_model.json     # XGBoost 模型
│   ├── pattern_engine.pkl    # 形态识别模型
│   ├── transformer_model.pt  # Transformer 模型
│   └── logs/                 # 运行日志
│
└── venv/                     # Python 虚拟环境（不提交 Git）
```

---

## 6. 配置说明

### 配置层级

| 优先级 | 文件 | 提交 Git | 说明 |
|:---|:---|:---|:---|
| 高 | `.env` | 否 | 本机实际配置，每台电脑独立维护 |
| 低 | `config.py` | 是 | 项目默认值，所有参数的 fallback |
| 参考 | `.env.example` | 是 | 模板，列出所有可配置项 |

### 关键可配置参数

| 环境变量 | 默认值 | 说明 |
|:---|:---|:---|
| `INSTANCE_NAME` | default | 运行实例标识 |
| `DATA_ROOT` | data | 数据根目录 |
| `AUTO_ENABLED` | true | 自动交易总开关 |
| `AUTO_SCORE_THRESHOLD` | 75 | 自动买入最低AI评分 |
| `AUTO_MAX_POSITIONS` | 50 | 最大同时持仓数 |
| `AI_SELL_SCORE_DROP` | 15 | AI评分衰减卖出确认阈值 |
| `SCAN_LIMIT` | 0 | 扫描股票上限（0=全量） |
| `ENABLE_HEAVY_MODEL` | true | 是否启用重模型(Transformer等) |
| `HISTORY_DAYS` | 365 | 历史数据拉取天数 |
| `INITIAL_CAPITAL` | 1000000 | 模拟盘初始资金(元) |
| `EMAIL_ENABLE` | true | 是否启用邮件通知 |
| `SMTP_PASSWORD` | 空 | QQ邮箱授权码 |
| `BAILIAN_API_KEY` | 空 | 百炼大模型API Key |

### 股票观察池

在 `config.py` 的 `WATCHLIST` 字典中添加或删除。

---

## 7. 每日收盘任务

```powershell
.\venv\Scripts\Activate.ps1
python daily_job.py
```

功能：增量更新缓存 → AI三层策略扫描 → 自动交易校验/执行 → 持仓检查 → 邮件推送

可配合 Windows 任务计划程序每日自动执行。

---

## 8. 重要提醒

### 两台电脑的运行数据是独立的

`data/` 目录不进 Git。所以：
- A 电脑的模拟盘持仓、AI评分、缓存
- B 电脑的模拟盘持仓、AI评分、缓存

是**两套独立的实验环境**，互不影响。

### 不要直接改 config.py 来适配本机

如果某个参数在这台电脑上需要不同的值（比如低配机要关闭重模型），
应该在 `.env` 里设置，不要直接改 `config.py`。

详细规范参见 `DEV_RULES.md`。

### 升级后第一次自动买入被跳过，通常是正常现象

如果你刚拉取了这轮改造代码，旧版 `ai_daily_scores.json` 可能不符合新结构要求。

表现为：

- AI一键执行交易可以运行
- 但摘要里提示“已跳过自动买入”

这时先重新运行一次 AI 扫描或 `daily_job.py` 即可。

---

## 9. 常见问题

### Q: 启动后浏览器没有自动打开？
手动访问 http://localhost:8501。如果端口被占用，换一个：`--server.port 8502`

### Q: 数据获取失败？
检查网络连接。AkShare 从东方财富获取数据，需要能访问外网。

### Q: 如何清除缓存数据？
删除 `data/stock_cache.db` 文件，下次启动时会自动重新创建。

### Q: 如何重置模拟账户？
在界面「模拟交易」页面点击「重置模拟账户」，或删除 `data/trading.db`。

### Q: pip install 报编码错误？
在命令前加 `$env:PYTHONUTF8=1`。

### Q: git push 超时？
设置代理：`$env:HTTPS_PROXY="http://127.0.0.1:17890"`。

### Q: 为什么 AI 一键执行交易时没有自动买入？
先看执行摘要里是否出现“评分未通过新鲜度校验”或“已跳过自动买入”。
如果是，先重新运行一次 AI 扫描，生成当天新版 `ai_daily_scores.json`。
