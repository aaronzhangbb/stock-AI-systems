# 多机开发与部署规范

本项目在两台本地电脑上同时开发和运行，未来将部署到云服务器。
本文档定义了代码、配置、数据三层分离的管理规则。

---

## 一、三层分离原则

| 层 | 内容 | 管理方式 |
|---|---|---|
| 代码层 | 业务逻辑、页面、策略算法、脚本 | Git 管理，所有机器共享 |
| 配置层 | 本机密钥、性能参数、路径、开关 | `.env` 文件，每台机器独立，不提交 Git |
| 状态层 | 数据库、缓存、模型、日志、运行结果 | `data/` 目录，每台机器独立，不提交 Git |

### 判断标准

改动前先问自己：
- **"这是项目功能变化"** → 提交 Git
- **"这是为了这台机器能跑/跑得更快"** → 放 `.env`，不提交 Git

---

## 二、Git 分支规则

### 分支命名

| 类型 | 格式 | 示例 |
|---|---|---|
| 稳定版本 | `master` | 永远保持可运行 |
| 新功能 | `feature/功能描述` | `feature/ai-sell-confirm` |
| 修 Bug | `fix/问题描述` | `fix/auto-trade-timeout` |
| 实验 | `experiment/实验描述` | `experiment/lightweight-scan` |

### 禁止事项

- 不按电脑建长期分支（如 `pc-a`、`low-pc`）
- 不在 `master` 上直接开发
- 不提交 `.env`、`data/`、`venv/` 等本机文件

### 功能开发流程

```
git checkout master
git pull origin master
git checkout -b feature/新功能
# ... 开发 ...
git add .
git commit -m "feat: 功能描述"
git push -u origin feature/新功能
# 功能完成后合并回 master
```

---

## 三、双机协作规则

### 换电脑前（在当前电脑）

```
git add .
git commit -m "feat: 进度描述"
git push
```

### 换电脑后（在另一台电脑）

```
git fetch
git checkout feature/当前功能分支
git pull
```

### 硬规则

- 换电脑前**必须** commit + push
- 另一台电脑开始前**必须** pull
- 任何一台电脑都不长期保留"只属于自己"的代码版本

---

## 四、配置管理

### 配置文件层级

| 文件 | 用途 | 是否提交 Git |
|---|---|---|
| `config.py` | 项目默认值（所有参数的 fallback） | 是 |
| `.env.example` | 环境变量模板和说明 | 是 |
| `.env` | 每台机器的实际配置 | 否 |

### 参数读取优先级

`.env` 环境变量 > `config.py` 默认值

### 各环境推荐配置

#### 低配电脑 `.env`

```env
INSTANCE_NAME=dev-low
AUTO_ENABLED=false
SCAN_LIMIT=500
ENABLE_HEAVY_MODEL=false
```

#### 高配电脑 `.env`

```env
INSTANCE_NAME=dev-high
AUTO_ENABLED=true
SCAN_LIMIT=0
ENABLE_HEAVY_MODEL=true
```

#### 云服务器 `.env`（未来）

```env
INSTANCE_NAME=server-prod
AUTO_ENABLED=true
SCAN_LIMIT=0
ENABLE_HEAVY_MODEL=true
SERVER_PORT=8501
```

---

## 五、数据与状态管理

### 不进 Git 的文件（已在 .gitignore 中配置）

- `.env` — 本机密钥和配置
- `venv/` — Python 虚拟环境
- `data/*.db` — SQLite 数据库
- `data/*.json` — 运行结果、缓存
- `data/*.pkl` — 模型文件
- `data/*.pt` — PyTorch 模型
- `data/*.model` — 其他模型
- `*.log` — 日志

### 两台电脑的运行状态

两台电脑各自维护独立的运行状态，包括：

- `trading.db` — 模拟交易数据库
- `stock_cache.db` — K线缓存
- `market_sentiment.json` — 情绪数据
- `ai_daily_scores.json` — AI 评分
- `last_auto_result.json` — 最近执行结果
- 模型文件 (`.pkl`, `.pt`, `.json`)

两台电脑的 AI 策略和模拟交易结果默认是**独立的两套实验环境**。

### 以后上云服务器

- 服务器成为唯一"正式运行环境"
- 服务器只部署 `master` 分支
- 所有正式状态数据（持仓、交易记录、绩效）以服务器为准
- 本地两台电脑继续做开发和实验

---

## 六、代码中的路径规范

### 所有数据路径必须经过 config

```python
# 正确: 通过 config 统一管理
import config
path = os.path.join(config.DATA_ROOT, 'ai_daily_scores.json')

# 错误: 写死路径
path = os.path.join('data', 'ai_daily_scores.json')
```

### 添加新参数时的规范

```python
# 在 config.py 中:
NEW_PARAM = int(os.getenv("NEW_PARAM", "默认值"))

# 在 .env.example 中同步添加说明:
# NEW_PARAM=默认值  # 参数说明
```

---

## 七、提交检查清单

每次 `git commit` 前确认：

- [ ] 没有把 `.env` 加入暂存区
- [ ] 没有把 `data/` 下的文件加入暂存区
- [ ] 没有把写死的本机路径提交（如 `C:\xunqing\...`）
- [ ] 新增的配置参数已经用 `os.getenv()` 包裹
- [ ] `.env.example` 已同步更新
