# QuantX 项目优化与进阶建议书

> 创建时间: 2026-02-14
> 状态: 规划中
> 当前版本: v5.1 (AI三层融合策略已完成)

---

## 概述

本文档基于对 QuantX 现有架构的深度分析，提出从模拟走向实战的系统性优化建议。
项目当前已完成: AI 三层融合策略(XGBoost + Pattern + Transformer) + AkShare数据流 + Streamlit前端 + 模拟交易 + 邮件推送。

以下为 5 个维度的强化计划，按优先级排序。

---

## 一、风险控制升级 (Risk Control) ⭐ 最高优先级

**现状问题**: 目前风控仅限于单只股票的止盈止损(position_monitor.py)，缺乏组合层面的系统性风控。

### 1.1 大盘择时风控 (Market Timing)

- **逻辑**: 利用现有的 `market_sentiment.py` 情绪分，动态调节推荐仓位上限
- **规则**:
  | 情绪分区间 | 情绪等级 | 最大推荐仓位 |
  |-----------|---------|-------------|
  | 0 ~ 25    | 极度恐慌 | 20% (仅精选1-2只) |
  | 25 ~ 40   | 偏悲观   | 40% |
  | 40 ~ 60   | 中性     | 70% |
  | 60 ~ 80   | 偏乐观   | 90% |
  | 80 ~ 100  | 极度贪婪 | 50% (警惕过热) |
- **实现位置**: `app.py` 的"仓位建议"生成逻辑，读取 `market_sentiment.json` 的 `sentiment_score`
- **UI变化**: 仓位建议页面顶部增加风控提示条，如 "⚠️ 当前市场情绪【偏悲观】(22.6分)，风控模型建议总仓位不超过 40%"

### 1.2 行业集中度限制 (Sector Cap)

- **逻辑**: 防止AI集中推荐同一板块(如5只银行股)，导致板块风险暴露
- **规则**:
  - 单一申万一级行业持仓市值不超过总资产的 30%
  - 单只个股持仓不超过总资产的 15%
- **实现**: 在仓位配置生成时，按行业分组检查，超限则自动降权
- **数据来源**: `stock_pool.db` 中的 `board_name` 字段

### 1.3 账户级熔断机制 (Circuit Breaker)

- **规则**:
  - 当日账户净值回撤 > 5%: 暂停所有买入，仅允许卖出
  - 累计最大回撤 > 15%: 暂停所有交易 3 天(冷静期)
  - 冷静期结束后需在UI上手动确认恢复
- **实现**: 在 `position_monitor.py` 中增加账户级别检查逻辑
- **存储**: `trading.db` 中新增 `risk_events` 表记录熔断事件

### 1.4 止损策略增强

- **现状**: 固定百分比止损
- **优化方向**:
  - 引入 ATR(平均真实波幅) 动态止损: 止损位 = 买入价 - 2倍ATR
  - 移动止盈: 股价上涨后，止损线同步上移(保护利润)
  - 时间止损: 持仓超过 N 天仍未盈利，强制平仓(避免资金占用)

---

## 二、数据源增强 (Data Enrichment)

**现状问题**: 数据主要依赖量价(OHLCV)和基础财务(PE/PB/ROE)，缺少非量价因子和排雷机制。

### 2.1 基本面排雷指标

- **目的**: 在AI推荐前预先剔除"地雷股"
- **排雷规则**:
  | 指标 | 阈值 | 动作 |
  |------|------|------|
  | 商誉/净资产 | > 30% | 剔除 |
  | 大股东质押率 | > 50% | 剔除 |
  | 连续经营性现金流为负 | >= 3年 | 剔除 |
  | ST / *ST 标记 | 是 | 剔除 |
  | 上市不足1年 | 是 | 剔除(新股波动大) |
- **实现**: 新建 `src/strategy/risk_filter.py`，在AI扫描前过滤股票池
- **数据来源**: AkShare `stock_financial_abstract_ths` + `stock_em_zt_pool`

### 2.2 宏观特征引入

- **目的**: 为 Transformer 提供市场全局环境上下文
- **候选特征**:
  - Shibor 利率(资金面松紧)
  - 人民币汇率 USD/CNY(影响北向资金)
  - 大宗商品指数(原油、铜，领先于周期股)
  - VIX 恐慌指数(全球风险偏好)
- **实现**: 新建 `src/data/macro_data.py`
- **接入方式**: AkShare 提供了 `macro_china_*` 系列接口

### 2.3 龙虎榜数据

- **目的**: 捕捉游资/机构的异动信号，作为AI评分加分项
- **关注维度**:
  - 机构席位净买入金额
  - 知名游资席位(如量化私募)出现频率
- **实现**: `src/data/dragon_tiger.py`
- **数据来源**: AkShare `stock_lhb_detail_em`

---

## 三、实盘自动化 (Execution) ⭐ 高优先级

**现状问题**: 目前仅支持模拟交易 + 邮件通知手动买入，手动执行效率低且易受情绪干扰。

### 3.1 方案: MiniQMT

- **原理**: Python通过 `xtquant` SDK 与本地 QMT 交易客户端通信，客户端负责连接券商网关
- **券商**: 需确认信达证券是否支持 QMT/MiniQMT (详见下方调研)

### 3.2 核心模块设计

```
src/trading/real_trader.py     — QMT 连接管理、下单、查询
src/trading/order_manager.py   — 订单状态追踪、成交回报处理
src/trading/sync_positions.py  — 券商持仓 <-> 本地DB 双向同步
```

### 3.3 安全机制

- **双重确认**: 实盘模式下，每次下单前需在UI上二次确认
- **金额上限**: 单笔订单金额不超过总资产的 10%
- **白名单**: 只允许交易AI推荐列表中的股票
- **紧急停止**: UI上提供"一键撤销所有挂单"按钮

### 3.4 滑点与成本模拟

- **目的**: 在回测中引入真实交易摩擦，评估策略净收益
- **参数**:
  - 买入滑点: +0.2% (模拟冲击成本)
  - 卖出滑点: -0.2%
  - 佣金: 万分之2.5 (双向)
  - 印花税: 千分之0.5 (仅卖出)
- **实现**: 修改 `strategy_validator.py` 的回测逻辑

---

## 四、模型迭代 (Model Iteration)

**现状问题**: 模型训练为手动触发(retrain_all.py)，缺乏自动化重训练和特征监控。

### 4.1 滚动训练机制 (Walk-Forward)

- **频率**: 每月1号自动重训练
- **窗口**: 训练集 = 过去3年，验证集 = 最近3个月
- **流程**:
  1. 自动拉取最新数据
  2. 训练新模型
  3. 在验证集上计算胜率/夏普
  4. 如果新模型 > 旧模型，自动替换；否则保留旧模型并报警
- **实现**: 修改 `retrain_all.py`，增加模型对比逻辑

### 4.2 特征重要性监控

- **目的**: 及时发现特征失效(如MACD在震荡市中无用)
- **输出**: 每次训练后生成 `data/feature_importance_report.json`
- **报警**: 如果某核心特征权重环比下降 > 50%，邮件/UI告警

### 4.3 策略效果跟踪

- **指标**: 每日记录AI推荐股票的次日/5日/10日实际收益
- **目的**: 形成"策略绩效曲线"，直观判断模型是否在赚钱
- **存储**: `trading.db` 新增 `strategy_performance` 表
- **展示**: 在"策略概览"页面增加绩效走势图

---

## 五、架构优化 (Architecture)

**现状问题**: app.py 职责过重(UI+逻辑+数据)，SQLite 有并发写入限制。

### 5.1 数据库升级 (长期)

- **现状**: SQLite (`data/trading.db`, `data/stock_pool.db`)
- **问题**: 并发写入时容易 "database is locked"
- **方案**: 迁移至 MySQL 或 PostgreSQL (Docker部署)
- **时机**: 当数据量超过 100MB 或多进程并发时考虑

### 5.2 前后端分离 (长期)

- **现状**: Streamlit 同时承担 UI 和业务逻辑
- **方案**:
  ```
  [Streamlit 前端] <--HTTP--> [FastAPI 后端] <---> [数据库]
                                    |
                              [定时任务 Worker]
  ```
- **好处**:
  - 策略服务可独立于 UI 运行
  - 支持多终端访问(手机/网页)
  - 便于部署到云服务器

### 5.3 日志与监控

- **现状**: 日志散落在各模块的 print/logging 中
- **优化**:
  - 统一日志格式，按日期分文件存储 (`logs/2026-02-14.log`)
  - 增加关键操作审计日志(买卖操作、模型替换等)
  - 可选: 接入简单的监控面板(如 Grafana)

---

## 实施路线图

| 阶段 | 时间 | 内容 | 优先级 |
|------|------|------|--------|
| Phase 1 | 1~2周 | 风控: 大盘择时 + 行业分散 + 熔断机制 | ⭐⭐⭐ |
| Phase 2 | 1~2周 | 排雷: 基本面黑名单过滤 | ⭐⭐⭐ |
| Phase 3 | 2~3周 | 实盘: QMT接入 + 安全机制 | ⭐⭐ |
| Phase 4 | 1周   | 模型: 滚动训练 + 特征监控 | ⭐⭐ |
| Phase 5 | 长期  | 架构: 数据库升级 + 前后端分离 | ⭐ |

---

---

## 附录A: 信达证券实盘接口调研

> 调研日期: 2026-02-14

### 结论: 信达证券 **不支持** QMT / MiniQMT

经查询，支持 QMT/MiniQMT 的主流券商为:
华泰、国泰君安、国金、申万宏源、中信建投、国信、银河、海通、方正、广发等。
**信达证券不在此列。**

### 信达证券现有交易软件

| 软件名 | 类型 | 是否支持程序化 |
|--------|------|---------------|
| 通达信金融终端 (v9.07) | PC行情+交易 | ❌ 仅手动交易 |
| 同花顺网上交易 | PC行情+交易 | ❌ 仅手动交易 |
| 通达信独立下单程序 | PC下单 | ❌ 仅手动交易 |
| **金字塔决策交易系统** | PC程序化 | ✅ 支持策略编写+自动执行 |
| 信达天下 APP | 移动端 | ❌ |

### 可行方案 (按推荐优先级排序)

#### 方案一: 金字塔决策交易系统 (信达原生支持) ⭐推荐

- **优点**: 信达证券官方提供，无需换券商，合规性最好
- **缺点**: 使用自有脚本语言(类似通达信公式)，不是Python；运行环境封闭，无法直接调用外部Python库
- **对接思路**:
  1. QuantX 每日生成信号文件 (`data/ai_action_list.json`)
  2. 将信号转换为金字塔可读的格式(CSV/TXT)
  3. 金字塔读取信号文件，执行下单
  4. 通过文件交换实现 Python QuantX <-> 金字塔 的通信
- **难度**: ⭐⭐⭐ (中等，需学习金字塔脚本语言)

#### 方案二: 新开一个支持QMT的券商账户 ⭐⭐强烈推荐

- **推荐券商**: 国金证券 (门槛可协商至10万) 或 华泰证券(综合实力强)
- **优点**:
  - 原生Python支持 (`xtquant` 库)，与 QuantX 无缝对接
  - 生态成熟，社区资料丰富
  - 可以保留信达账户不动，用新账户跑量化
- **缺点**: 需要额外开户，资金分散在两个账户
- **对接思路**:
  1. 新券商开户 + 申请 MiniQMT 权限
  2. 安装 QMT 客户端 + `xtquant` Python库
  3. 编写 `src/trading/real_trader.py`，直接读取AI信号并下单
  4. 实时同步券商持仓到 QuantX 系统
- **难度**: ⭐⭐ (较低，Python原生支持)

#### 方案三: 通达信 + Python自动化 (灰色地带)

- **原理**: 用Python模拟键盘鼠标操作通达信客户端
- **工具**: `pywinauto` 或 `pyautogui` 库
- **优点**: 不需要换券商
- **缺点**:
  - ⚠️ 不稳定 (界面变动就会失效)
  - ⚠️ 灰色地带 (券商可能封号)
  - ⚠️ 速度慢 (模拟点击有延迟)
- **难度**: ⭐⭐⭐⭐ (高，且不推荐)
- **结论**: **不推荐**，仅作为了解

### 最终建议

**短期 (验证策略阶段)**: 继续使用当前的模拟交易 + 邮件通知 + 手动在信达下单。

**中期 (策略验证有效后)**: 选择"方案二"，新开一个国金证券或华泰证券账户，
专门用于量化交易。保留信达账户做日常手动交易。
MiniQMT 开通门槛约 10~50万，具体可和客户经理协商。

**QMT对接代码框架** (方案二的核心实现):

```python
# src/trading/real_trader.py  —— QMT实盘交易模块

from xtquant import xt_trader
from xtquant.xt_type import StockAccount
import xtquant.xt_constant as xt_constant
import json, time, logging

logger = logging.getLogger(__name__)


class RealTrader:
    """QMT实盘交易封装"""

    def __init__(self, qmt_path, account_id, session_id=None):
        """
        Args:
            qmt_path: QMT客户端安装目录下的 userdata_mini 路径
                       例如: 'D:\\国金QMT\\userdata_mini'
            account_id: 资金账号
        """
        self.session_id = session_id or int(time.time())
        self.account = StockAccount(account_id)

        # 创建交易对象并连接
        self.trader = xt_trader.XtQuantTrader(qmt_path, self.session_id)
        self.trader.start()

        result = self.trader.connect()
        if result == 0:
            logger.info("✅ QMT实盘接口连接成功")
            self.trader.subscribe(self.account)
            self.connected = True
        else:
            logger.error("❌ QMT连接失败，请检查客户端是否登录")
            self.connected = False

    def get_assets(self):
        """查询账户资金"""
        return self.trader.query_stock_asset(self.account)

    def get_positions(self):
        """查询真实持仓"""
        return self.trader.query_stock_positions(self.account)

    def buy(self, stock_code, amount, price=0, strategy="AI_V2"):
        """
        买入下单
        Args:
            stock_code: 股票代码，格式 '600519.SH' 或 '000001.SZ'
            amount: 买入股数 (必须是100的整数倍)
            price: 限价，0表示市价
        """
        if not self.connected:
            logger.error("未连接，无法下单")
            return None

        order_type = xt_constant.FIX_PRICE if price > 0 else xt_constant.LATEST_PRICE
        order_id = self.trader.order_stock(
            self.account, stock_code,
            xt_constant.STOCK_BUY, amount,
            order_type, price,
            strategy, "QuantX自动买入"
        )
        logger.info(f"买入委托: {stock_code} x {amount}股, 订单号={order_id}")
        return order_id

    def sell(self, stock_code, amount, price=0, strategy="AI_V2"):
        """卖出下单"""
        if not self.connected:
            return None

        order_type = xt_constant.FIX_PRICE if price > 0 else xt_constant.LATEST_PRICE
        order_id = self.trader.order_stock(
            self.account, stock_code,
            xt_constant.STOCK_SELL, amount,
            order_type, price,
            strategy, "QuantX自动卖出"
        )
        logger.info(f"卖出委托: {stock_code} x {amount}股, 订单号={order_id}")
        return order_id

    def cancel_all(self):
        """撤销所有挂单"""
        orders = self.trader.query_stock_orders(self.account)
        cancelled = 0
        for order in orders:
            if order.order_status in [48, 49, 50]:  # 待报/已报/部分成交
                self.trader.cancel_order_stock(self.account, order.order_id)
                cancelled += 1
        logger.info(f"已撤销 {cancelled} 笔挂单")
        return cancelled


def execute_ai_signals(trader, signal_file='data/ai_action_list.json',
                       total_capital=100000):
    """
    读取AI信号文件，自动执行下单

    Args:
        trader: RealTrader 实例
        signal_file: AI操作清单路径
        total_capital: 可用资金
    """
    with open(signal_file, 'r', encoding='utf-8') as f:
        actions = json.load(f)

    buy_actions = [a for a in actions if a.get('action') == '买入']
    if not buy_actions:
        logger.info("今日无买入信号")
        return

    # 按AI评分降序
    buy_actions.sort(key=lambda x: x.get('score', 0), reverse=True)

    # 计算每只股票分配金额 (等权分配，最多5只)
    top_n = min(len(buy_actions), 5)
    per_stock = total_capital / top_n

    for action in buy_actions[:top_n]:
        code = action['stock_code']
        price = action.get('current_price', 0)
        if price <= 0:
            continue

        # 计算买入手数 (向下取整到100股)
        shares = int(per_stock / price / 100) * 100
        if shares < 100:
            logger.warning(f"{code} 资金不足一手，跳过")
            continue

        # 限价委托 (在现价基础上加0.5%确保成交)
        limit_price = round(price * 1.005, 2)
        trader.buy(code, shares, limit_price)
        time.sleep(0.5)  # 避免下单过快

    logger.info(f"今日执行完毕: 共委托 {top_n} 只股票")
```

---

## 备注

- 所有优化均以"不破坏现有功能"为前提，增量式推进
- 每个 Phase 完成后，需在模拟盘验证 2 周再进入下一阶段
- 实盘接入采用"方案二"(新开QMT券商账户)，信达账户保留做手动交易
