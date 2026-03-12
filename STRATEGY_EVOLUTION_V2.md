# 策略进化 V2 — 闭环自适应优化系统

> 版本: 2.0  
> 日期: 2026-02-25  
> 状态: 实施中

---

## 一、背景与动机

V1 版策略学习模块 (`StrategyLearner`) 能够从已完成交易中提炼出基本规律并给出文字建议，
但存在以下 10 个核心问题：

| # | 问题 | 严重程度 |
|---|------|---------|
| 1 | `_derive_optimal_params` 用 `split('调整为')` 硬解析字符串，极脆弱 | 高 |
| 2 | 评分阈值搜索固定枚举 `[75,80,85,90]`，无法发现最优点 | 高 |
| 3 | 止损/止盈分析只看绝对触发率，不考虑全局盈亏比 | 中高 |
| 4 | 持有时间用固定 3 桶 (1-5/6-12/12+天)，不自适应 | 中 |
| 5 | 承诺的"仓位合理性"学习维度完全缺失 | 高 |
| 6 | 卖后分析 `analyze_post_sell_performance()` 被重复调用 2~3 次 | 中高 |
| 7 | 所有历史交易等权参与学习，无时效性衰减 | 中 |
| 8 | `hold_days` 用自然日而非交易日，节假日会失真 | 中 |
| 9 | 学习结果没有回灌给 AI 扫描和自动交易，闭环断裂 | 高 |
| 10 | 报告仅保留最新一份，无历史演化追踪 | 低 |

---

## 二、目标架构

```
┌──────────────────────────────────────────────────────────┐
│                    策略进化层 (StrategyLearner)            │
│                                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐   │
│  │ 动态评分 │ │ 止损精度 │ │ 止盈精度 │ │ 持有时间  │   │
│  │ 阈值学习 │ │ +盈亏比  │ │ +盈亏比  │ │ 效率分析  │   │
│  └──────────┘ └──────────┘ └──────────┘ └───────────┘   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐   │
│  │ 仓位合理 │ │ 卖出时机 │ │ 整体评价 │ │ 时效性    │   │
│  │ 性(新增) │ │ 学习     │ │          │ │ 加权      │   │
│  └──────────┘ └──────────┘ └──────────┘ └───────────┘   │
│                         │                                │
│                         ▼                                │
│              strategy_insights.json (V2)                 │
│              + insights_archive/{date}.json               │
└──────────────────────────┬───────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  AI扫描层    │
                    │             │
                    │ · 动态阈值  │
                    │ · 环境权重  │
                    │ · 学习仓位  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  自动交易    │
                    │             │
                    │ · 学习阈值  │
                    │ · 学习仓位  │
                    └──────┬──────┘
                           │
                    交易结果回流 ──→ 策略进化层
```

---

## 三、报告 Schema V2 完整定义

```json
{
  "schema_version": 2,
  "status": "ok | insufficient_data",
  "trade_count": 42,
  "reliability": "high | medium | low | none",
  "learning_window": {
    "start_date": "2025-11-01",
    "end_date": "2026-02-25",
    "total_calendar_days": 117,
    "recent_30d_trades": 15,
    "recent_90d_trades": 30
  },
  "metrics_snapshot": {
    "win_rate": 58.3,
    "rr_ratio": 1.8,
    "avg_pnl_pct": 1.2,
    "max_drawdown_pct": -5.3,
    "profit_factor": 2.1,
    "total_pnl": 12500.0
  },
  "insights": [
    {
      "category": "评分阈值 | 止损精度 | 止盈精度 | 持有时间 | 仓位合理性 | 卖出时机 | 整体评价",
      "finding": "人类可读的分析发现",
      "suggestion": "人类可读的建议",
      "confidence": 0.85,
      "metadata": {
        "// 结构化数值, 各类别不同, 示例:": "",
        "recommended_threshold": 82,
        "sample_count": 25,
        "expected_win_rate": 62.0
      }
    }
  ],
  "optimal_params": {
    "score_threshold": 82,
    "avg_hold_days": 6.5,
    "stop_trigger_rate": 35.0,
    "tp_trigger_rate": 20.0,
    "recommended_position_map": {
      "90+": 0.25,
      "85-90": 0.20,
      "80-85": 0.15,
      "75-80": 0.10
    },
    "recommended_stop_multi": null,
    "recommended_target_multi": null
  },
  "generated_at": "2026-02-25 14:30:00"
}
```

### 各类别 insight.metadata 定义

**评分阈值**:
```json
{
  "recommended_threshold": 82,
  "current_threshold": 75,
  "sample_count": 25,
  "expected_win_rate": 62.0,
  "search_range": [72, 92]
}
```

**止损精度**:
```json
{
  "stop_rate": 45.0,
  "avg_stop_loss_pct": -3.2,
  "global_rr_ratio": 1.8,
  "recommended_action": "maintain | loosen | tighten"
}
```

**止盈精度**:
```json
{
  "avg_tp_gain": 4.5,
  "avg_sl_loss": 2.8,
  "effective_rr": 1.6,
  "tp_count": 12,
  "recommended_action": "maintain | raise_target | lower_target"
}
```

**持有时间**:
```json
{
  "best_bucket": "4-8天",
  "best_daily_return": 0.35,
  "best_hold_range": [4, 8],
  "avg_hold_days": 6.5,
  "buckets": [
    {"range": "1-4天", "count": 10, "daily_return": 0.2, "win_rate": 50.0},
    {"range": "4-8天", "count": 15, "daily_return": 0.35, "win_rate": 60.0},
    {"range": "8天+", "count": 8, "daily_return": 0.15, "win_rate": 45.0}
  ]
}
```

**仓位合理性**:
```json
{
  "high_score_should_overweight": true,
  "recommended_position_map": {"90+": 0.25, "85-90": 0.20, "80-85": 0.15, "75-80": 0.10},
  "large_position_win_rate": 55.0,
  "small_position_win_rate": 60.0,
  "concentration_risk": "low | medium | high"
}
```

**卖出时机**:
```json
{
  "total_analyzed": 30,
  "right_pct": 50.0,
  "early_pct": 30.0,
  "late_pct": 20.0,
  "worst_reason": "止损",
  "worst_reason_early_pct": 45.0
}
```

**整体评价**:
```json
{
  "win_rate": 58.0,
  "rr_ratio": 1.8,
  "quality": "优秀 | 合格 | 需优化",
  "total_trades": 42,
  "profitable_trades": 24
}
```

---

## 四、各模块改造详细说明

### 4.1 strategy_learner.py (核心重构)

1. **insight metadata 升级**: 所有 `_analyze_*` 方法返回的 insight 字典新增 `metadata` 字段
2. **动态阈值**: `_analyze_score_threshold` 用数据驱动搜索替代固定枚举
3. **全局盈亏比修正**: `_analyze_stop_loss` 和 `_analyze_take_profit` 引入整体盈亏比
4. **时间效率**: `_analyze_hold_duration` 改为自适应分桶 + 日均收益率
5. **仓位分析**: 新增 `_analyze_position_sizing` 方法
6. **时效性加权**: 新增 `_apply_recency_weight` 为近期交易赋予更高权重
7. **`_derive_optimal_params`**: 从 metadata 直接取值，不再解析字符串
8. **报告归档**: `_save_report` 同时写最新文件和归档文件

### 4.2 performance.py (性能分析层)

1. **消除重复**: `analyze_post_sell_by_reason(post_df=None)` 支持外部传入
2. **行情缓存**: `analyze_post_sell_performance` 内部按 stock_code 缓存
3. **交易日字段**: `get_completed_trades` 新增 `hold_trading_days`
4. **资金字段**: 新增 `trade_amount` 和 `position_ratio`

### 4.3 ai_scan_service.py (扫描层接入)

1. **`_load_learned_params()`**: 读取 strategy_insights.json
2. **动态阈值**: `_build_action_list` 使用学习阈值
3. **环境权重**: 分数融合权重根据 market_sentiment 动态调整
4. **学习仓位**: 用学习结果的 position_map 替代引擎默认值

### 4.4 auto_trader.py (执行层)

1. 买入过滤优先使用学习阈值，fallback 到 config.AUTO_SCORE_THRESHOLD

### 4.5 app.py (前端)

1. 洞察卡片新增"仓位合理性"类别
2. 推荐参数区扩展
3. 卖后分析消除重复调用

### 4.6 config.py (配置)

新增 3 个配置项:
- `LEARNER_RECENCY_WEIGHTS`: 时效性权重 [1.0, 0.6, 0.3]
- `LEARNER_MIN_BUCKET_SIZE`: 分桶最小样本数 5
- `SCAN_USE_LEARNED_PARAMS`: 扫描是否使用学习参数 True

---

## 五、前端展示变更

### 策略进化标签页

| 区域 | V1 | V2 |
|------|----|----|
| 洞察卡片类别 | 5 类 (评分/止损/止盈/持有时间/卖出时机/整体) | 7 类 (新增仓位合理性) |
| 推荐参数 | 3 列 (阈值/止损率/止盈率) | 5 列 (+ 推荐仓位/推荐持有期) |
| 对比展示 | 无 | "上次 vs 本次"差异对比 |
| 卖后分析 | 重复调用 | 单次调用复用 |

---

## 六、测试与验证要点

### 单元测试

1. `_apply_recency_weight`: 验证 30/90/90+ 天的权重分配
2. `_analyze_score_threshold`: 验证动态搜索能找到正确阈值
3. `_analyze_position_sizing`: 验证分档仓位建议合理性
4. `_derive_optimal_params`: 验证从 metadata 取值而非字符串解析
5. `analyze_post_sell_by_reason(post_df=...)`: 验证传入已有数据时不重复调用

### 集成验证

1. 运行 `learn()` 后检查 `strategy_insights.json` 符合 schema V2
2. 运行 AI 扫描后检查是否使用了学习阈值
3. 运行自动交易后检查买入过滤是否使用学习阈值
4. 前端策略进化页正确展示所有新增类别和参数

### 回归验证

1. 无历史交易时 `learn()` 仍正常返回 `insufficient_data`
2. `strategy_insights.json` 不存在时扫描和交易 fallback 到默认值
3. 已有的前端功能不受影响
