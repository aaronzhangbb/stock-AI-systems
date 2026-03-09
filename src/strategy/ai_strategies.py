# -*- coding: utf-8 -*-
"""
AI挖掘策略集成模块
将AI从历史数据中发现的策略规则集成到交易系统中

=== 版本V1: 初始发现（基于500只随机样本回测） ===
  - 均值回归是A股最强规律
  - 最佳持有周期10天
  - 超跌（偏离均线/RSI极低/布林带下轨）后反弹概率60-75%

=== 版本V2: 全市场验证（5008只可交易A股 × 5维度 × 35分组） ===
  新发现:
  1. 「布林带底部放量」从均衡级逆袭成为全市场最强策略 (35个分组中24次最佳)
  2. 原精选级策略(超跌MA30/MA60)在全市场验证中胜率下降约15%, 说明原500只样本偏乐观
  3. 策略表现高度依赖行业: 科技成长>制造装备>电子>周期资源>大盘金融>消费医药
  4. 医药生物行业是几乎所有策略的"黑洞"(胜率比平均低10%+)
  5. 高波动股(50-80%年化)是策略的"温床"(所有策略在此表现最好)
  6. 低价股(<10元)策略效果最差,应避开

策略分为3档(V2修订):
  1. 精选型 (3套): V1胜率>70%, 但V2全市场验证降至58-59%, 需限定行业使用
  2. 均衡型 (4套): 其中「布林带底部放量」在V2中表现最优,建议升级
  3. 广谱型 (2套): 覆盖面广但胜率偏低
"""

import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config

REPORT_PATH = os.path.join(config.DATA_ROOT, 'ai_strategy_report.json')


# ============================================================
# AI发现的核心策略 (硬编码版，不依赖JSON文件)
# V1来源: 2023-02-08 ~ 2026-02-06 全市场500只随机股票回测
# V2验证: 2026-02-08 全市场5008只可交易A股 × 5维度(行业/市值/波动率/价格/趋势)
# ============================================================
AI_STRATEGIES = [
    # === 精选型: 胜率>70%, 夏普>2.0 ===
    {
        'id': 'ai_core_01',
        'name': '超跌MA30均值回归',
        'tier': '精选',
        'type': '均值回归',
        'description': '股价大幅低于30日均线（偏离-9.6%~-55%），10天内反弹概率极高',
        'conditions': [
            ('ma30_diff', '>=', -0.5531),
            ('ma30_diff', '<=', -0.0962),
        ],
        'hold_days': 10,
        'backtest': {
            'win_rate': 74.97, 'sharpe': 2.50, 'avg_return': 4.96,
            'profit_loss_ratio': 1.52, 'max_drawdown': -77.05,
            'avg_win': 8.47, 'avg_lose': -5.55, 'trades': 887,
        },
        'v2_fullmarket': {
            'win_rate': 58.3, 'sharpe': 1.05, 'avg_return': 2.75,
            'profit_loss_ratio': 1.29, 'trades': 257814,
            'best_group': '行业-电网设备(胜率64.4%)',
            'worst_group': '行业-医药生物(胜率50.8%)',
            'note': 'V2全市场胜率下降16.6%,建议限定科技/制造板块使用',
        },
        'signal_strength': 90,
    },
    {
        'id': 'ai_core_02',
        'name': '超跌MA60均值回归',
        'tier': '精选',
        'type': '均值回归',
        'description': '股价大幅低于60日均线（偏离-13.2%~-58.5%），中长期均值回归力量更强',
        'conditions': [
            ('ma60_diff', '>=', -0.5846),
            ('ma60_diff', '<=', -0.1323),
        ],
        'hold_days': 10,
        'backtest': {
            'win_rate': 73.13, 'sharpe': 2.56, 'avg_return': 4.99,
            'profit_loss_ratio': 1.59, 'max_drawdown': -75.96,
            'avg_win': 8.87, 'avg_lose': -5.57, 'trades': 536,
        },
        'v2_fullmarket': {
            'win_rate': 58.7, 'sharpe': 1.17, 'avg_return': 3.10,
            'profit_loss_ratio': 1.37, 'trades': 243828,
            'best_group': '行业-电网设备(胜率67.9%)',
            'worst_group': '行业-医药生物(胜率50.8%)',
            'note': 'V2全市场胜率下降14.4%,在电网设备行业仍表现优秀',
        },
        'signal_strength': 88,
    },
    {
        'id': 'ai_core_03',
        'name': '均线下行企稳',
        'tier': '精选',
        'type': '趋势企稳',
        'description': 'MA30斜率处于极低区间（-14.4%~-2.8%），均线大幅下跌后放缓=底部企稳信号',
        'conditions': [
            ('ma30_slope', '>=', -0.1444),
            ('ma30_slope', '<=', -0.0284),
        ],
        'hold_days': 10,
        'backtest': {
            'win_rate': 70.46, 'sharpe': 2.39, 'avg_return': 4.37,
            'profit_loss_ratio': 1.68, 'max_drawdown': -65.83,
            'avg_win': 8.27, 'avg_lose': -4.91, 'trades': 738,
        },
        'v2_fullmarket': {
            'win_rate': 58.0, 'sharpe': 1.26, 'avg_return': 3.14,
            'profit_loss_ratio': 1.54, 'trades': 260057,
            'best_group': '行业-电网设备(胜率67.0%)',
            'worst_group': '行业-医药生物(胜率49.2%)',
            'note': 'V2在横盘震荡市中仍有效(胜率57.5%,夏普1.57)',
        },
        'signal_strength': 85,
    },

    # === 均衡型: 胜率>60%, 夏普>1.5 ===
    {
        'id': 'ai_balanced_01',
        'name': 'RSI+布林带超卖反弹',
        'tier': '均衡',
        'type': '均值回归',
        'description': 'RSI(14)≤30 且 布林带位置≤0.2，双重超卖确认',
        'conditions': [
            ('rsi14', '<=', 30.0),
            ('bb_pos', '<=', 0.2),
        ],
        'hold_days': 10,
        'backtest': {
            'win_rate': 65.93, 'sharpe': 1.85, 'avg_return': 3.05,
            'profit_loss_ratio': 1.77, 'max_drawdown': -78.55,
            'avg_win': 6.54, 'avg_lose': -3.69, 'trades': 1541,
        },
        'v2_fullmarket': {
            'win_rate': 51.4, 'sharpe': 0.44, 'avg_return': 1.05,
            'profit_loss_ratio': 1.22, 'trades': 278253,
            'best_group': '高波动(50-80%)(胜率55.2%)',
            'worst_group': '行业-医药生物(胜率44.2%,夏普-0.12)',
            'note': '⚠️ V2大幅衰退,全市场几乎失效,建议降级或弃用',
        },
        'signal_strength': 80,
    },
    {
        'id': 'ai_balanced_02',
        'name': '深度超卖三重确认',
        'tier': '均衡',
        'type': '均值回归',
        'description': 'RSI(6)≤20 + 连跌≥3天 + 接近20日新低，三重超卖共振',
        'conditions': [
            ('rsi6', '<=', 20.0),
            ('consec_down', '>=', 3.0),
            ('dist_low20', '<=', 0.02),
        ],
        'hold_days': 10,
        'backtest': {
            'win_rate': 62.85, 'sharpe': 1.71, 'avg_return': 2.71,
            'profit_loss_ratio': 1.88, 'max_drawdown': -50.59,
            'avg_win': 6.29, 'avg_lose': -3.34, 'trades': 1058,
        },
        'v2_fullmarket': {
            'win_rate': 53.4, 'sharpe': 0.56, 'avg_return': 1.33,
            'profit_loss_ratio': 1.23, 'trades': 96001,
            'best_group': '行业-轻工制造(胜率59.7%)',
            'worst_group': '行业-电力设备(胜率49.1%,夏普-0.18)',
            'note': '⚠️ V2表现大幅下降,条件过于严苛导致信号少且质量差',
        },
        'signal_strength': 78,
    },
    {
        'id': 'ai_balanced_03',
        'name': '布林带底部放量',
        'tier': '均衡',
        'type': '量价共振',
        'description': '布林带位置≤0.1（极端下轨）+ 量比≥1.5（资金抄底信号）',
        'conditions': [
            ('bb_pos', '<=', 0.1),
            ('vol_ratio', '>=', 1.5),
        ],
        'hold_days': 10,
        'backtest': {
            'win_rate': 59.81, 'sharpe': 1.41, 'avg_return': 2.57,
            'profit_loss_ratio': 1.59, 'max_drawdown': -48.78,
            'avg_win': 7.43, 'avg_lose': -4.67, 'trades': 321,
        },
        'v2_fullmarket': {
            'win_rate': 59.5, 'sharpe': 1.51, 'avg_return': 4.10,
            'profit_loss_ratio': 1.63, 'trades': 26895,
            'best_group': '风格-B-科技成长(胜率70.7%,夏普2.78)',
            'worst_group': '低波动<25%(胜率51.2%)',
            'note': '🏆 V2逆袭!全市场综合评分第1,35个分组中24次最佳,建议升级为精选',
        },
        'signal_strength': 75,
    },
    {
        'id': 'ai_balanced_04',
        'name': 'MA60斜率探底',
        'tier': '均衡',
        'type': '趋势企稳',
        'description': 'MA60斜率处于极低区间（-8.1%~-2.1%），长期均线下行接近平缓',
        'conditions': [
            ('ma60_slope', '>=', -0.0807),
            ('ma60_slope', '<=', -0.0205),
        ],
        'hold_days': 10,
        'backtest': {
            'win_rate': 65.55, 'sharpe': 1.92, 'avg_return': 3.66,
            'profit_loss_ratio': 1.62, 'max_drawdown': -68.30,
            'avg_win': 8.28, 'avg_lose': -5.11, 'trades': 357,
        },
        'v2_fullmarket': {
            'win_rate': 56.4, 'sharpe': 1.32, 'avg_return': 3.31,
            'profit_loss_ratio': 1.69, 'trades': 238840,
            'best_group': '行业-电网设备(胜率67.9%,夏普2.44)',
            'worst_group': '行业-医药生物(胜率49.2%)',
            'note': 'V2综合排名第2,在电网设备行业表现卓越,适合行业择时',
        },
        'signal_strength': 77,
    },

    # === 广谱型: 胜率>55%, 夏普>0.8 ===
    {
        'id': 'ai_wide_01',
        'name': '高波动区间捕捉',
        'tier': '广谱',
        'type': '波动率',
        'description': '布林带宽处于高位（>0.27），高波动环境中超跌反弹幅度大',
        'conditions': [
            ('bb_width', '>=', 0.2698),
            ('bb_width', '<=', 2.1271),
        ],
        'hold_days': 10,
        'backtest': {
            'win_rate': 55.66, 'sharpe': 1.00, 'avg_return': 2.64,
            'profit_loss_ratio': 1.42, 'max_drawdown': -72.57,
            'avg_win': 10.84, 'avg_lose': -7.66, 'trades': 1942,
        },
        'v2_fullmarket': {
            'win_rate': 53.9, 'sharpe': 0.82, 'avg_return': 2.28,
            'profit_loss_ratio': 1.39, 'trades': 574763,
            'best_group': '高价股>80元(胜率55.2%)',
            'worst_group': '行业-医药生物(胜率49.3%)',
            'note': 'V2基本持平,胜率刚过50%,风险收益比不佳',
        },
        'signal_strength': 65,
    },
    {
        'id': 'ai_wide_02',
        'name': '极高波动率反转',
        'tier': '广谱',
        'type': '波动率',
        'description': '20日波动率处于极高位（>78%年化），极端波动后均值回归',
        'conditions': [
            ('vol_20', '>=', 0.7807),
            ('vol_20', '<=', 1.9810),
        ],
        'hold_days': 10,
        'backtest': {
            'win_rate': 56.18, 'sharpe': 0.98, 'avg_return': 2.68,
            'profit_loss_ratio': 1.35, 'max_drawdown': -89.88,
            'avg_win': 11.26, 'avg_lose': -8.32, 'trades': 947,
        },
        'v2_fullmarket': {
            'win_rate': 53.6, 'sharpe': 0.78, 'avg_return': 2.33,
            'profit_loss_ratio': 1.38, 'trades': 288566,
            'best_group': '行业-化学制品(胜率57.4%)',
            'worst_group': '低价股<10元(胜率50.5%)',
            'note': 'V2基本持平,全市场验证无明显退化',
        },
        'signal_strength': 60,
    },
]

# ============================================================
# V2 全市场验证总结（2026-02-08 5008只可交易A股 × 5维度 × 35分组）
# ============================================================
V3_FULL_MARKET_RESULT = {
    'date': '2026-02-08',
    'scan_type': '全量回测(无采样)',
    'total_stocks': 5008,
    'hold_days': 10,
    'cost_rate': 0.002,

    # === 核心结论 ===
    'conclusion': (
        '经过5008只全部可交易A股的完整回测验证（无采样偏差），'
        '最优策略为「布林带底部放量 + MA60斜率探底」组合，'
        '胜率79.0%，夏普4.24，每笔收益+14.92%，覆盖2524只股票。'
    ),

    # === 最优策略：组合 ===
    'best_combo': {
        'name': '布林带底部放量 + MA60斜率探底',
        'id': 'v3_combo_best',
        'logic': 'AND（同时满足两个条件）',
        'conditions': '布林带位置≤0.1 且 量比≥1.5 且 MA60斜率在-8.1%~-2.1%区间',
        'sub_strategies': ['ai_balanced_03', 'ai_balanced_04'],
        'win_rate': 79.0, 'sharpe': 4.24, 'avg_return': 14.92,
        'profit_loss_ratio': 3.37, 'max_drawdown': None,
        'trades': 4913, 'stocks_hit': 2524,
    },

    # === 最优策略：单个 ===
    'best_single': {
        'name': '布林带底部放量',
        'id': 'ai_balanced_03',
        'win_rate': 58.6, 'sharpe': 1.46, 'avg_return': 3.93,
        'profit_loss_ratio': 1.68,
        'trades': 27724, 'stocks_hit': 4862,
        'note': '覆盖面最广(4862只)、表现最稳定的单策略',
    },

    # === 全量排行（前5） ===
    'ranking': [
        {'rank': 1, 'name': '布林带底部放量 + MA60斜率探底', 'type': '组合AND',
         'win_rate': 79.0, 'sharpe': 4.24, 'avg_return': 14.92, 'trades': 4913, 'score': 125.0},
        {'rank': 2, 'name': '布林带底部放量 + 均线下行企稳', 'type': '组合AND',
         'win_rate': 74.9, 'sharpe': 3.54, 'avg_return': 13.30, 'trades': 5527, 'score': 109.7},
        {'rank': 3, 'name': '布林带底部放量 + 超跌MA60均值回归', 'type': '组合AND',
         'win_rate': 71.1, 'sharpe': 2.69, 'avg_return': 8.87, 'trades': 11543, 'score': 86.5},
        {'rank': 4, 'name': 'MA60斜率探底 + 超跌MA60均值回归', 'type': '组合AND',
         'win_rate': 63.8, 'sharpe': 2.13, 'avg_return': 5.76, 'trades': 118442, 'score': 69.0},
        {'rank': 5, 'name': '布林带底部放量(单策略)', 'type': '单策略',
         'win_rate': 58.6, 'sharpe': 1.46, 'avg_return': 3.93, 'trades': 27724, 'score': 53.2},
    ],

    # === 单策略全量排行 ===
    'single_ranking': [
        {'rank': 1, 'id': 'ai_balanced_03', 'name': '布林带底部放量',
         'win_rate': 58.6, 'sharpe': 1.46, 'avg_return': 3.93, 'trades': 27724, 'stocks_hit': 4862},
        {'rank': 2, 'id': 'ai_core_02', 'name': '超跌MA60均值回归',
         'win_rate': 59.1, 'sharpe': 1.20, 'avg_return': 3.18, 'trades': 242649, 'stocks_hit': 4790},
        {'rank': 3, 'id': 'ai_balanced_04', 'name': 'MA60斜率探底',
         'win_rate': 56.4, 'sharpe': 1.33, 'avg_return': 3.32, 'trades': 237512, 'stocks_hit': 4474},
        {'rank': 4, 'id': 'ai_core_03', 'name': '均线下行企稳',
         'win_rate': 58.4, 'sharpe': 1.27, 'avg_return': 3.18, 'trades': 260159, 'stocks_hit': 4682},
        {'rank': 5, 'id': 'ai_core_01', 'name': '超跌MA30均值回归',
         'win_rate': 58.5, 'sharpe': 1.05, 'avg_return': 2.78, 'trades': 258117, 'stocks_hit': 4877},
    ],

    # === 投资建议（V3最终版） ===
    'investment_advice': [
        '首选「布林带底部放量 + MA60斜率探底」组合策略，胜率79%，夏普4.24',
        '组合信号出现时果断买入，持有10天卖出，每笔预期收益+14.92%',
        '信号较少时(组合条件严格)，可退而使用「布林带底部放量」单策略',
        '单策略覆盖4862只股票，交易27724次，胜率58.6%，稳定可靠',
        '避开医药生物行业，优选科技成长/制造装备板块',
        '优选股价10-80元、年化波动率50-80%的标的',
    ],
}

# ============================================================
# V3 全量验证组合策略（5008只A股无采样回测）
# ============================================================
AI_COMBO_STRATEGIES = [
    # === V3新增：全量验证最优组合 ===
    {
        'id': 'v3_combo_01',
        'name': '布林带底部放量+MA60斜率探底',
        'tier': 'V3最优',
        'type': '组合',
        'description': '布林带极端下轨放量 + MA60长期均线企稳=量价时空四维共振，全量回测最优',
        'sub_strategies': ['ai_balanced_03', 'ai_balanced_04'],
        'backtest': {
            'win_rate': 79.0, 'sharpe': 4.24, 'avg_return': 14.92,
            'profit_loss_ratio': 3.37, 'max_drawdown': None,
            'trades': 4913,
        },
        'v3_fullmarket': True,
        'stocks_hit': 2524,
        'signal_strength': 98,
    },
    {
        'id': 'v3_combo_02',
        'name': '布林带底部放量+均线下行企稳',
        'tier': 'V3次优',
        'type': '组合',
        'description': '布林带底部放量 + MA30斜率企稳，短中期趋势反转确认',
        'sub_strategies': ['ai_balanced_03', 'ai_core_03'],
        'backtest': {
            'win_rate': 74.9, 'sharpe': 3.54, 'avg_return': 13.30,
            'profit_loss_ratio': 2.36, 'max_drawdown': None,
            'trades': 5527,
        },
        'v3_fullmarket': True,
        'stocks_hit': 2888,
        'signal_strength': 95,
    },
    {
        'id': 'v3_combo_03',
        'name': '布林带底部放量+超跌MA60均值回归',
        'tier': 'V3精选',
        'type': '组合',
        'description': '布林带底部放量 + 股价偏离MA60，量价超跌双重确认',
        'sub_strategies': ['ai_balanced_03', 'ai_core_02'],
        'backtest': {
            'win_rate': 71.1, 'sharpe': 2.69, 'avg_return': 8.87,
            'profit_loss_ratio': 1.83, 'max_drawdown': None,
            'trades': 11543,
        },
        'v3_fullmarket': True,
        'stocks_hit': 4074,
        'signal_strength': 92,
    },
    {
        'id': 'v3_combo_04',
        'name': 'MA60斜率探底+超跌MA60均值回归',
        'tier': 'V3均衡',
        'type': '组合',
        'description': 'MA60斜率企稳 + MA60偏离回归，纯均线趋势+均值回归',
        'sub_strategies': ['ai_balanced_04', 'ai_core_02'],
        'backtest': {
            'win_rate': 63.8, 'sharpe': 2.13, 'avg_return': 5.76,
            'profit_loss_ratio': 2.07, 'max_drawdown': None,
            'trades': 118442,
        },
        'v3_fullmarket': True,
        'stocks_hit': 4243,
        'signal_strength': 85,
    },
    # === V1旧组合（保留参考） ===
    {
        'id': 'ai_combo_01',
        'name': '超跌MA30+MA60双均线组合',
        'tier': 'V1精选',
        'type': '组合',
        'description': '同时满足MA30和MA60超跌条件，双重确认底部（V1-500只样本）',
        'sub_strategies': ['ai_core_01', 'ai_core_02'],
        'backtest': {
            'win_rate': 78.72, 'sharpe': 3.03, 'avg_return': 6.26,
            'profit_loss_ratio': 1.63, 'max_drawdown': -68.37,
            'trades': 230,
        },
        'signal_strength': 70,
    },
    {
        'id': 'ai_combo_02',
        'name': '超跌MA30+均线企稳组合',
        'tier': 'V1精选',
        'type': '组合',
        'description': '价格偏离MA30 + MA30斜率开始企稳（V1-500只样本）',
        'sub_strategies': ['ai_core_01', 'ai_core_03'],
        'backtest': {
            'win_rate': 77.54, 'sharpe': 2.98, 'avg_return': 5.87,
            'profit_loss_ratio': 1.65, 'max_drawdown': -70.22,
            'trades': 428,
        },
        'signal_strength': 68,
    },
    {
        'id': 'ai_combo_03',
        'name': 'AI挖掘+超卖反弹组合',
        'tier': '均衡组合',
        'type': '组合',
        'description': '均线超跌 + RSI/布林带超卖信号的多重共振',
        'sub_strategies': ['ai_core_01', 'ai_balanced_01'],
        'backtest': {
            'win_rate': 77.61, 'sharpe': 2.69, 'avg_return': 5.32,
            'profit_loss_ratio': 1.57, 'max_drawdown': -70.87,
            'trades': 585,
        },
        'signal_strength': 90,
    },
]


# ============================================================
# 特征计算（与 run_strategy_analysis.py 保持一致）
# ============================================================
def compute_ai_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算AI策略所需的技术指标特征
    输入: 含有 date/open/high/low/close/volume 的 DataFrame
    输出: 增加了 AI 特征列的 DataFrame
    """
    data = df.copy()
    data = data.sort_values('date').reset_index(drop=True)
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']

    # MA偏离
    for p in [30, 60]:
        ma = close.rolling(p).mean()
        data[f'ma{p}_diff'] = (close - ma) / ma
        data[f'ma{p}_slope'] = ma.pct_change(5)

    # RSI
    for period in [6, 14]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss_ = (-delta).where(delta < 0, 0.0)
        ag = gain.rolling(period, min_periods=period).mean()
        al = loss_.rolling(period, min_periods=period).mean()
        rs = ag / al
        data[f'rsi{period}'] = 100 - (100 / (1 + rs))

    # 布林带
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    data['bb_pos'] = (close - bb_lower) / (bb_upper - bb_lower)
    data['bb_width'] = (bb_upper - bb_lower) / bb_mid

    # 波动率
    ret1 = close.pct_change()
    data['vol_20'] = ret1.rolling(20).std() * np.sqrt(252)

    # 量比
    data['vol_ratio'] = volume / volume.rolling(5).mean()

    # 连跌天数
    down = (close < close.shift(1)).astype(int)
    data['consec_down'] = down.groupby((down != down.shift()).cumsum()).cumsum()

    # 距20日新低
    data['dist_low20'] = close / low.rolling(20).min() - 1

    return data


def check_strategy_signal(row: pd.Series, strategy: dict) -> bool:
    """检查单行数据是否满足策略条件"""
    for feat, op, thresh in strategy['conditions']:
        val = row.get(feat)
        if pd.isna(val):
            return False
        if op == '<=' and val > thresh:
            return False
        if op == '>=' and val < thresh:
            return False
        if op == '>' and val <= thresh:
            return False
        if op == '<' and val >= thresh:
            return False
    return True


def scan_stock_signals(df: pd.DataFrame, tiers=None) -> list:
    """
    扫描单只股票是否触发AI策略信号

    参数:
        df: 历史OHLCV DataFrame (需至少60行)
        tiers: 只检查指定档次 ['精选', '均衡', '广谱'], None=全部

    返回:
        list[dict]: 触发的信号列表
    """
    if df.empty or len(df) < 65:
        return []

    data = compute_ai_features(df)
    if data.empty:
        return []

    last_row = data.iloc[-1]
    signals = []

    for strat in AI_STRATEGIES:
        if tiers and strat['tier'] not in tiers:
            continue

        if check_strategy_signal(last_row, strat):
            bt = strat['backtest']
            signals.append({
                'signal': 'buy',
                'strategy_id': strat['id'],
                'strategy': f"AI-{strat['name']}",
                'tier': strat['tier'],
                'type': strat['type'],
                'strength': strat['signal_strength'],
                'hold_days': strat['hold_days'],
                'reason': (
                    f"{strat['description']} | "
                    f"胜率{bt['win_rate']:.0f}% 夏普{bt['sharpe']:.1f} "
                    f"盈亏比{bt['profit_loss_ratio']:.1f}"
                ),
                'backtest': bt,
            })

    # 检查组合策略
    triggered_ids = {s['strategy_id'] for s in signals}
    for combo in AI_COMBO_STRATEGIES:
        if all(sid in triggered_ids for sid in combo['sub_strategies']):
            bt = combo['backtest']
            signals.append({
                'signal': 'buy',
                'strategy_id': combo['id'],
                'strategy': f"AI组合-{combo['name']}",
                'tier': combo['tier'],
                'type': combo['type'],
                'strength': combo['signal_strength'],
                'hold_days': 10,
                'reason': (
                    f"{combo['description']} | "
                    f"胜率{bt['win_rate']:.0f}% 夏普{bt['sharpe']:.1f}"
                ),
                'backtest': bt,
            })

    signals.sort(key=lambda x: x['strength'], reverse=True)
    return signals


def get_strategy_summary() -> dict:
    """获取AI策略总结信息"""
    # V3全量验证后的最优数据
    v3 = V3_FULL_MARKET_RESULT
    v3_best = v3['best_combo']
    v3_combos = [c for c in AI_COMBO_STRATEGIES if c.get('v3_fullmarket')]
    return {
        'total_strategies': len(AI_STRATEGIES),
        'combo_strategies': len(AI_COMBO_STRATEGIES),
        'tiers': {
            '精选': len([s for s in AI_STRATEGIES if s['tier'] == '精选']),
            '均衡': len([s for s in AI_STRATEGIES if s['tier'] == '均衡']),
            '广谱': len([s for s in AI_STRATEGIES if s['tier'] == '广谱']),
        },
        'best_strategy': v3_best['name'],
        'best_win_rate': v3_best['win_rate'],
        'best_sharpe': v3_best['sharpe'],
        'data_source': f'{v3["total_stocks"]}只可交易A股全量回测',
        'validation': 'V3全量验证(无采样偏差)',
        'hold_days': v3['hold_days'],
        'core_finding': v3['conclusion'],
        'v3_combos': len(v3_combos),
    }
