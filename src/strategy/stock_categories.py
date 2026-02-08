# -*- coding: utf-8 -*-
"""
股票分类模块
基于申万行业 + 交易所板块 + 股价区间，将A股分为策略适用组

分类维度：
  1. 大类风格: A-大盘稳健 / B-科技成长 / C-消费医药 / D-周期制造 / E-制造装备
  2. 市场板块: 沪市主板 / 深市主板 / 中小板 / 创业板 / 科创板
  3. 股价区间: 低价(<10元) / 中价(10-50元) / 高价(>50元)

不同分类适用不同策略参数和信号权重
"""

import sqlite3
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

POOL_DB = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'stock_pool.db')
CACHE_DB = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'stock_cache.db')

# ============================================================
# 申万行业 → 大类风格映射
# ============================================================
INDUSTRY_TO_STYLE = {
    # A-大盘稳健: 金融、公用事业、地产、交通等低波动板块
    '银行': 'A-大盘稳健', '非银金融': 'A-大盘稳健', '公用事业': 'A-大盘稳健',
    '房地产': 'A-大盘稳健', '交通运输': 'A-大盘稳健', '电力行业': 'A-大盘稳健',
    '建筑装饰': 'A-大盘稳健', '综合': 'A-大盘稳健',

    # B-科技成长: 电子、计算机、通信、军工、新能源等高成长板块
    '电子': 'B-科技成长', '计算机': 'B-科技成长', '通信': 'B-科技成长',
    '电力设备': 'B-科技成长', '传媒': 'B-科技成长', '国防军工': 'B-科技成长',
    '电网设备': 'B-科技成长', '光伏设备': 'B-科技成长', '电池': 'B-科技成长',
    '风电设备': 'B-科技成长', '电机': 'B-科技成长',

    # C-消费医药: 食品、医药、家电、零售等消费板块
    '食品饮料': 'C-消费医药', '医药生物': 'C-消费医药', '家用电器': 'C-消费医药',
    '化学制药': 'C-消费医药', '中药': 'C-消费医药', '农林牧渔': 'C-消费医药',
    '商贸零售': 'C-消费医药', '社会服务': 'C-消费医药', '美容护理': 'C-消费医药',
    '纺织服饰': 'C-消费医药', '珠宝首饰': 'C-消费医药',

    # D-周期制造: 化工、有色、钢铁、煤炭、石油等周期板块
    '基础化工': 'D-周期制造', '有色金属': 'D-周期制造', '钢铁': 'D-周期制造',
    '煤炭': 'D-周期制造', '石油石化': 'D-周期制造', '化学制品': 'D-周期制造',
    '化学原料': 'D-周期制造', '建筑材料': 'D-周期制造', '小金属': 'D-周期制造',
    '石油行业': 'D-周期制造', '采掘行业': 'D-周期制造', '能源金属': 'D-周期制造',
    '化纤行业': 'D-周期制造', '化肥行业': 'D-周期制造', '农药兽药': 'D-周期制造',

    # E-制造装备: 机械、专用设备、汽车、轻工等制造板块
    '机械设备': 'E-制造装备', '专用设备': 'E-制造装备', '汽车': 'E-制造装备',
    '轻工制造': 'E-制造装备', '环保': 'E-制造装备', '塑料制品': 'E-制造装备',
    '造纸印刷': 'E-制造装备', '包装材料': 'E-制造装备', '燃气': 'E-制造装备',
}

# ============================================================
# 各风格的策略参数偏好（基于分类回测验证数据）
# 验证日期: 2026-02-07, 每类采样80只股票
# ============================================================
STYLE_STRATEGY_CONFIG = {
    'A-大盘稳健': {
        'description': '银行/金融/公用事业等低波动蓝筹',
        'stock_count': 640,
        'best_strategies': ['ai_core_02', 'ai_core_01'],  # MA60>MA30
        'best_strategy_names': ['超跌MA60均值回归', '超跌MA30均值回归'],
        'hold_days': 10,
        'stop_loss': 0.05,      # 蓝筹止损窄
        'take_profit': 0.12,
        'position_ratio': 0.35,  # 仓位可以高
        'verified_performance': {
            '超跌MA60': {'win_rate': 75.6, 'avg_return': 5.41, 'sharpe': 3.05, 'plr': 1.87},
            '超跌MA30': {'win_rate': 67.5, 'avg_return': 4.50, 'sharpe': 2.18, 'plr': 1.78},
            'RSI超卖':  {'win_rate': 60.2, 'avg_return': 2.26, 'sharpe': 1.35, 'plr': 1.75},
        },
        'note': 'MA60超跌策略最佳(胜率75.6%,夏普3.05), 波动小盈亏比高',
    },
    'B-科技成长': {
        'description': '电子/计算机/新能源等高成长高波动',
        'stock_count': 1569,
        'best_strategies': ['ai_core_02', 'ai_core_03', 'ai_balanced_02'],
        'best_strategy_names': ['超跌MA60均值回归', '均线下行企稳', '深度超卖三重确认'],
        'hold_days': 10,
        'stop_loss': 0.10,      # 高波动宽止损
        'take_profit': 0.25,
        'position_ratio': 0.20,  # 仓位控制
        'verified_performance': {
            '超跌MA60': {'win_rate': 73.1, 'avg_return': 5.88, 'sharpe': 2.43, 'plr': 1.43},
            '均线企稳': {'win_rate': 71.8, 'avg_return': 3.35, 'sharpe': 1.74, 'plr': 1.09},
            '深度超卖': {'win_rate': 65.9, 'avg_return': 2.92, 'sharpe': 1.64, 'plr': 1.49},
        },
        'note': 'MA60超跌效果好(均收益5.88%), 均线企稳胜率也高(71.8%)',
    },
    'C-消费医药': {
        'description': '食品饮料/医药/家电等消费白马',
        'stock_count': 1117,
        'best_strategies': ['ai_core_02', 'ai_core_01', 'ai_balanced_02'],
        'best_strategy_names': ['超跌MA60均值回归', '超跌MA30均值回归', '深度超卖三重确认'],
        'hold_days': 10,
        'stop_loss': 0.06,
        'take_profit': 0.15,
        'position_ratio': 0.30,
        'verified_performance': {
            '超跌MA60': {'win_rate': 70.6, 'avg_return': 5.16, 'sharpe': 2.60, 'plr': 1.69},
            '超跌MA30': {'win_rate': 69.5, 'avg_return': 4.51, 'sharpe': 2.20, 'plr': 1.56},
            '深度超卖': {'win_rate': 61.9, 'avg_return': 2.53, 'sharpe': 1.86, 'plr': 1.85},
        },
        'note': '消费白马超跌后回归稳定, 深度超卖盈亏比最优(1.85)',
    },
    'D-周期制造': {
        'description': '化工/有色/钢铁/煤炭等周期股',
        'stock_count': 750,
        'best_strategies': ['ai_core_02', 'ai_core_01', 'ai_core_03'],
        'best_strategy_names': ['超跌MA60均值回归', '超跌MA30均值回归', '均线下行企稳'],
        'hold_days': 10,
        'stop_loss': 0.08,
        'take_profit': 0.20,
        'position_ratio': 0.25,
        'verified_performance': {
            '超跌MA60': {'win_rate': 85.5, 'avg_return': 8.65, 'sharpe': 3.66, 'plr': 2.15},
            '超跌MA30': {'win_rate': 79.5, 'avg_return': 6.72, 'sharpe': 3.44, 'plr': 2.78},
            '均线企稳': {'win_rate': 69.6, 'avg_return': 4.42, 'sharpe': 2.14, 'plr': 1.82},
        },
        'note': '★★★全场最佳! 周期股超跌反弹最强(MA60胜率85.5%,均收益8.65%)',
    },
    'E-制造装备': {
        'description': '机械/专用设备/汽车等制造业',
        'stock_count': 1106,
        'best_strategies': ['ai_core_02', 'ai_core_01', 'ai_core_03'],
        'best_strategy_names': ['超跌MA60均值回归', '超跌MA30均值回归', '均线下行企稳'],
        'hold_days': 10,
        'stop_loss': 0.08,
        'take_profit': 0.18,
        'position_ratio': 0.25,
        'verified_performance': {
            '超跌MA60': {'win_rate': 77.0, 'avg_return': 6.27, 'sharpe': 3.61, 'plr': 1.73},
            '超跌MA30': {'win_rate': 75.3, 'avg_return': 5.97, 'sharpe': 2.50, 'plr': 1.76},
            '均线企稳': {'win_rate': 74.7, 'avg_return': 4.39, 'sharpe': 2.65, 'plr': 1.38},
        },
        'note': '三大策略均表现优秀, 胜率均>74%, 适合全面配置',
    },
}


def classify_market_board(stock_code: str) -> str:
    """按交易所/板块分类"""
    if stock_code.startswith(('600', '601', '603', '605')):
        return '沪市主板'
    elif stock_code.startswith(('000', '001', '003')):
        return '深市主板'
    elif stock_code.startswith('002'):
        return '中小板'
    elif stock_code.startswith(('300', '301')):
        return '创业板'
    elif stock_code.startswith(('688', '689')):
        return '科创板'
    else:
        return '其他'


def classify_price_range(price: float) -> str:
    """按股价区间分类"""
    if price < 5:
        return '低价股(<5元)'
    elif price < 10:
        return '中低价(5-10元)'
    elif price < 20:
        return '中价(10-20元)'
    elif price < 50:
        return '中高价(20-50元)'
    elif price < 100:
        return '高价(50-100元)'
    else:
        return '超高价(>100元)'


def get_stock_style(stock_code: str) -> str:
    """获取单只股票的风格分类"""
    if not os.path.exists(POOL_DB):
        return 'F-未分类'

    conn = sqlite3.connect(POOL_DB)
    cursor = conn.cursor()
    cursor.execute('SELECT board_name FROM all_stocks WHERE stock_code = ?', (stock_code,))
    row = cursor.fetchone()
    conn.close()

    if row and row[0]:
        return INDUSTRY_TO_STYLE.get(row[0], 'F-未分类')
    return 'F-未分类'


def get_stock_full_category(stock_code: str, latest_price: float = None) -> dict:
    """获取股票的完整分类信息"""
    style = get_stock_style(stock_code)
    market = classify_market_board(stock_code)
    price_range = classify_price_range(latest_price) if latest_price else '未知'

    config = STYLE_STRATEGY_CONFIG.get(style, {})

    return {
        'stock_code': stock_code,
        'style': style,
        'market_board': market,
        'price_range': price_range,
        'strategy_config': config,
    }


def get_style_strategy_config(style: str) -> dict:
    """获取风格对应的策略参数"""
    return STYLE_STRATEGY_CONFIG.get(style, STYLE_STRATEGY_CONFIG.get('E-制造装备', {}))


def get_all_stock_categories() -> pd.DataFrame:
    """获取所有缓存股票的分类"""
    if not os.path.exists(POOL_DB) or not os.path.exists(CACHE_DB):
        return pd.DataFrame()

    conn_pool = sqlite3.connect(POOL_DB)
    conn_cache = sqlite3.connect(CACHE_DB)

    meta = pd.read_sql_query('SELECT stock_code FROM cache_meta', conn_cache)
    pool = pd.read_sql_query('SELECT stock_code, stock_name, board_name FROM all_stocks', conn_pool)

    merged = meta.merge(pool, on='stock_code', how='left')
    merged['style'] = merged['board_name'].map(INDUSTRY_TO_STYLE).fillna('F-未分类')
    merged['market_board'] = merged['stock_code'].apply(classify_market_board)

    conn_pool.close()
    conn_cache.close()
    return merged


def get_category_stats() -> dict:
    """获取分类统计信息"""
    df = get_all_stock_categories()
    if df.empty:
        return {}

    return {
        'total_stocks': len(df),
        'style_distribution': df['style'].value_counts().to_dict(),
        'market_distribution': df['market_board'].value_counts().to_dict(),
        'industry_count': df['board_name'].nunique(),
    }
