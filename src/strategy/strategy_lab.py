# -*- coding: utf-8 -*-
"""
策略发现实验室 - 多维度策略分析引擎
Strategy Discovery Lab

核心功能:
  1. 多维度股票分组 (行业/市值/波动率/价格/趋势/大盘环境)
  2. 批量策略回测 (对每个分组运行所有AI策略)
  3. 性能矩阵生成 (维度 × 策略 的回测矩阵)
  4. 参数优化 (自动寻找最优策略参数)
  5. ML模式发现 (按分组挖掘新规则)
  6. 大盘环境识别 (牛/熊/震荡市划分)
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config

from src.data.data_cache import DataCache
from src.data.stock_pool import StockPool
from src.strategy.ai_strategies import AI_STRATEGIES, compute_ai_features

# ============================================================
# 常量
# ============================================================
LAB_DB_PATH = os.path.join(config.DATA_ROOT, 'strategy_lab.db')

DIMENSIONS = {
    'industry': {'name': '行业板块', 'icon': '🏭', 'desc': '按申万行业板块分组，发现不同行业的最优策略'},
    'market_cap': {'name': '市值规模', 'icon': '📊', 'desc': '按市值分为大盘/中盘/小盘，分析规模效应'},
    'volatility': {'name': '波动率', 'icon': '📈', 'desc': '按历史波动率分组，高波动vs低波动策略表现'},
    'price_range': {'name': '价格区间', 'icon': '💰', 'desc': '按股价分组，不同价位策略效果差异'},
    'trend': {'name': '趋势状态', 'icon': '📐', 'desc': '上升/下降/震荡趋势下的策略表现'},
    'market_regime': {'name': '大盘环境', 'icon': '🌍', 'desc': '牛市/熊市/震荡市中策略效果对比'},
}

# 行业→风格映射（精简版，用于快速分类）
INDUSTRY_STYLE_MAP = {
    '银行': 'A-大盘金融', '非银金融': 'A-大盘金融', '房地产': 'A-大盘金融',
    '公用事业': 'A-大盘金融', '交通运输': 'A-大盘金融', '建筑装饰': 'A-大盘金融',
    '电子': 'B-科技成长', '计算机': 'B-科技成长', '通信': 'B-科技成长',
    '传媒': 'B-科技成长', '国防军工': 'B-科技成长',
    '食品饮料': 'C-消费医药', '医药生物': 'C-消费医药', '家用电器': 'C-消费医药',
    '美容护理': 'C-消费医药', '商贸零售': 'C-消费医药', '社会服务': 'C-消费医药',
    '纺织服饰': 'C-消费医药', '轻工制造': 'C-消费医药', '农林牧渔': 'C-消费医药',
    '基础化工': 'D-周期资源', '有色金属': 'D-周期资源', '钢铁': 'D-周期资源',
    '煤炭': 'D-周期资源', '石油石化': 'D-周期资源', '建筑材料': 'D-周期资源',
    '机械设备': 'E-制造装备', '电力设备': 'E-制造装备', '汽车': 'E-制造装备',
    '环保': 'E-制造装备', '综合': 'E-制造装备',
}


# ============================================================
# 策略实验室主类
# ============================================================
class StrategyLab:
    """多维度策略发现实验室"""

    def __init__(self, lab_db_path=None):
        self.cache = DataCache()
        self.pool = StockPool()
        self.lab_db = lab_db_path or LAB_DB_PATH
        self._init_lab_db()

    def _init_lab_db(self):
        """初始化实验室结果数据库"""
        os.makedirs(os.path.dirname(self.lab_db), exist_ok=True)
        conn = sqlite3.connect(self.lab_db)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS lab_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dimension TEXT NOT NULL,
                group_name TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                win_rate REAL, avg_return REAL, sharpe REAL,
                max_drawdown REAL, profit_loss_ratio REAL,
                trades INTEGER, avg_win REAL, avg_lose REAL,
                stock_count INTEGER, sample_size INTEGER,
                created_at TEXT NOT NULL,
                UNIQUE(dimension, group_name, strategy_id, created_at)
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS lab_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dimension TEXT NOT NULL,
                status TEXT DEFAULT 'running',
                total_groups INTEGER, completed_groups INTEGER DEFAULT 0,
                total_stocks INTEGER DEFAULT 0, sample_per_group INTEGER,
                started_at TEXT NOT NULL, completed_at TEXT,
                params TEXT
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS param_optimization (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dimension TEXT, group_name TEXT,
                strategy_base TEXT, param_name TEXT,
                param_value REAL, win_rate REAL, sharpe REAL,
                avg_return REAL, trades INTEGER,
                created_at TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    # ============================================================
    # 核心: 股票分组
    # ============================================================
    def get_stock_groups(self, dimension: str, max_per_group: int = 50) -> dict:
        """
        按指定维度将股票分组

        参数:
            dimension: 维度名称 (industry/market_cap/volatility/price_range/trend/market_regime)
            max_per_group: 每组最大采样数

        返回:
            dict: {group_name: [(stock_code, stock_name), ...]}
        """
        # 获取所有已缓存的股票，并过滤掉不可交易的
        cached = self.cache.get_all_cached_stocks()
        if cached.empty:
            return {}

        # 从股票池获取可交易列表，过滤掉ST/B股/北交所等
        try:
            tradeable_df = self.pool.get_tradeable_stocks()
            if not tradeable_df.empty:
                tradeable_codes = set(tradeable_df['stock_code'].values)
                cached = cached[cached['stock_code'].isin(tradeable_codes)]
                if cached.empty:
                    return {}
        except (AttributeError, Exception):
            pass  # 如果方法不存在，使用全部缓存股票

        if dimension == 'industry':
            return self._group_by_industry(cached, max_per_group)
        elif dimension == 'market_cap':
            return self._group_by_market_cap(cached, max_per_group)
        elif dimension == 'volatility':
            return self._group_by_volatility(cached, max_per_group)
        elif dimension == 'price_range':
            return self._group_by_price_range(cached, max_per_group)
        elif dimension == 'trend':
            return self._group_by_trend(cached, max_per_group)
        elif dimension == 'market_regime':
            return self._group_by_market_regime(cached, max_per_group)
        else:
            return {}

    def _group_by_industry(self, cached_df, max_per_group):
        """按行业板块分组"""
        groups = {}
        boards = self.pool.get_industry_boards()
        if boards.empty:
            return groups

        # 用风格分类代替过多的细分行业
        style_stocks = defaultdict(list)
        for _, board in boards.iterrows():
            bname = board['board_name']
            style = INDUSTRY_STYLE_MAP.get(bname, 'F-其他')
            try:
                stocks = self.pool.get_tradeable_stocks_by_board(bname)
            except AttributeError:
                stocks = self.pool.get_stocks_by_board(bname)
            for _, stk in stocks.iterrows():
                code = stk['stock_code']
                if code in cached_df['stock_code'].values:
                    style_stocks[style].append((code, stk.get('stock_name', '')))

        # 同时提供细分行业分组（取前15个大行业）
        top_boards = boards.nlargest(15, 'stock_count')
        for _, board in top_boards.iterrows():
            bname = board['board_name']
            try:
                stocks = self.pool.get_tradeable_stocks_by_board(bname)
            except AttributeError:
                stocks = self.pool.get_stocks_by_board(bname)
            stock_list = []
            for _, stk in stocks.iterrows():
                code = stk['stock_code']
                if code in cached_df['stock_code'].values:
                    stock_list.append((code, stk.get('stock_name', '')))
            if len(stock_list) >= 10:
                key = f'行业-{bname}'
                if len(stock_list) > max_per_group:
                    stock_list = self._random_sample(stock_list, max_per_group)
                groups[key] = stock_list

        # 风格分组
        for style, stock_list in style_stocks.items():
            if len(stock_list) >= 5:
                key = f'风格-{style}'
                if len(stock_list) > max_per_group:
                    stock_list = self._random_sample(stock_list, max_per_group)
                groups[key] = stock_list

        return groups

    def _group_by_market_cap(self, cached_df, max_per_group):
        """按市值规模分组（用最新价格×日均成交量近似）"""
        groups = {'大盘股(高流动性)': [], '中盘股(中流动性)': [], '小盘股(低流动性)': []}
        stock_scores = []

        for _, row in cached_df.iterrows():
            code = row['stock_code']
            name = row.get('stock_name', '')
            try:
                df = self.cache.load_kline(code)
                if df is not None and len(df) >= 30:
                    recent = df.tail(30)
                    avg_amount = (recent['close'] * recent['volume']).mean()
                    stock_scores.append((code, name, avg_amount))
            except Exception:
                continue

        if not stock_scores:
            return groups

        stock_scores.sort(key=lambda x: x[2], reverse=True)
        n = len(stock_scores)
        t1 = n // 3
        t2 = 2 * n // 3

        large = [(s[0], s[1]) for s in stock_scores[:t1]]
        mid = [(s[0], s[1]) for s in stock_scores[t1:t2]]
        small = [(s[0], s[1]) for s in stock_scores[t2:]]

        groups['大盘股(高流动性)'] = self._random_sample(large, max_per_group) if len(large) > max_per_group else large
        groups['中盘股(中流动性)'] = self._random_sample(mid, max_per_group) if len(mid) > max_per_group else mid
        groups['小盘股(低流动性)'] = self._random_sample(small, max_per_group) if len(small) > max_per_group else small

        return {k: v for k, v in groups.items() if v}

    def _group_by_volatility(self, cached_df, max_per_group):
        """按60日年化波动率分组"""
        groups = {'低波动(<25%)': [], '中波动(25-50%)': [], '高波动(50-80%)': [], '极高波动(>80%)': []}
        for _, row in cached_df.iterrows():
            code = row['stock_code']
            name = row.get('stock_name', '')
            try:
                df = self.cache.load_kline(code)
                if df is not None and len(df) >= 70:
                    ret = df['close'].pct_change().dropna()
                    vol = ret.tail(60).std() * np.sqrt(252) * 100
                    item = (code, name)
                    if vol < 25:
                        groups['低波动(<25%)'].append(item)
                    elif vol < 50:
                        groups['中波动(25-50%)'].append(item)
                    elif vol < 80:
                        groups['高波动(50-80%)'].append(item)
                    else:
                        groups['极高波动(>80%)'].append(item)
            except Exception:
                continue

        for k in groups:
            if len(groups[k]) > max_per_group:
                groups[k] = self._random_sample(groups[k], max_per_group)
        return {k: v for k, v in groups.items() if v}

    def _group_by_price_range(self, cached_df, max_per_group):
        """按最新收盘价分组"""
        groups = {'低价股(<10元)': [], '中低价(10-30元)': [], '中高价(30-80元)': [], '高价股(>80元)': []}
        for _, row in cached_df.iterrows():
            code = row['stock_code']
            name = row.get('stock_name', '')
            try:
                df = self.cache.load_kline(code)
                if df is not None and len(df) >= 10:
                    price = float(df.iloc[-1]['close'])
                    item = (code, name)
                    if price < 10:
                        groups['低价股(<10元)'].append(item)
                    elif price < 30:
                        groups['中低价(10-30元)'].append(item)
                    elif price < 80:
                        groups['中高价(30-80元)'].append(item)
                    else:
                        groups['高价股(>80元)'].append(item)
            except Exception:
                continue

        for k in groups:
            if len(groups[k]) > max_per_group:
                groups[k] = self._random_sample(groups[k], max_per_group)
        return {k: v for k, v in groups.items() if v}

    def _group_by_trend(self, cached_df, max_per_group):
        """按当前趋势状态分组（基于MA20/MA60关系）"""
        groups = {'上升趋势': [], '下降趋势': [], '横盘震荡': []}
        for _, row in cached_df.iterrows():
            code = row['stock_code']
            name = row.get('stock_name', '')
            try:
                df = self.cache.load_kline(code)
                if df is not None and len(df) >= 65:
                    close = df['close']
                    ma20 = close.rolling(20).mean().iloc[-1]
                    ma60 = close.rolling(60).mean().iloc[-1]
                    ma20_slope = (close.rolling(20).mean().iloc[-1] - close.rolling(20).mean().iloc[-6]) / close.rolling(20).mean().iloc[-6]
                    last_price = float(close.iloc[-1])
                    item = (code, name)
                    if last_price > ma20 > ma60 and ma20_slope > 0.01:
                        groups['上升趋势'].append(item)
                    elif last_price < ma20 < ma60 or ma20_slope < -0.02:
                        groups['下降趋势'].append(item)
                    else:
                        groups['横盘震荡'].append(item)
            except Exception:
                continue

        for k in groups:
            if len(groups[k]) > max_per_group:
                groups[k] = self._random_sample(groups[k], max_per_group)
        return {k: v for k, v in groups.items() if v}

    def _group_by_market_regime(self, cached_df, max_per_group):
        """
        按大盘环境分组 — 不分股票，而是把时间段分为牛/熊/震荡
        对所有股票在不同时间段运行策略
        """
        # 用沪深300/上证指数的表现来判断大盘环境
        # 简单方法：把历史数据按季度划分，看每个季度指数涨跌
        # 由于我们没有指数数据缓存，用所有缓存股票的中位数涨跌来近似
        groups = {
            '牛市阶段(大盘上涨>10%)': [],
            '熊市阶段(大盘下跌>10%)': [],
            '震荡阶段(大盘±10%)': [],
        }
        # 所有缓存股票即为分析对象，按时间段拆分在回测中处理
        all_stocks = [(row['stock_code'], row.get('stock_name', '')) for _, row in cached_df.iterrows()]
        if len(all_stocks) > max_per_group:
            all_stocks = self._random_sample(all_stocks, max_per_group)
        # 所有股票放在同一组，但回测时按时段拆分
        groups['牛市阶段(大盘上涨>10%)'] = all_stocks
        groups['熊市阶段(大盘下跌>10%)'] = all_stocks
        groups['震荡阶段(大盘±10%)'] = all_stocks
        return groups

    @staticmethod
    def _random_sample(lst, n):
        """随机采样"""
        np.random.seed(42)
        indices = np.random.choice(len(lst), size=min(n, len(lst)), replace=False)
        return [lst[i] for i in indices]

    # ============================================================
    # 核心: 策略回测
    # ============================================================
    def backtest_strategy_on_group(self, stock_list, strategy, hold_days=10,
                                   regime=None, cost_rate=0.002):
        """
        在一组股票上回测单个策略

        参数:
            stock_list: [(code, name), ...]
            strategy: AI_STRATEGIES中的策略dict
            hold_days: 持有天数
            regime: 市场环境过滤 ('bull'/'bear'/'sideways'/None)
            cost_rate: 交易成本率

        返回:
            dict: 回测指标
        """
        all_returns = []

        for code, name in stock_list:
            try:
                df = self.cache.load_kline(code)
                if df is None or len(df) < 80:
                    continue

                data = compute_ai_features(df)
                if data.empty:
                    continue

                # 市场环境过滤
                if regime:
                    data = self._filter_by_regime(data, regime)
                    if len(data) < 80:
                        continue

                # 检查策略触发条件
                for i in range(60, len(data) - hold_days):
                    row = data.iloc[i]
                    triggered = True
                    for feat, op, val in strategy['conditions']:
                        if feat not in data.columns:
                            triggered = False
                            break
                        fv = row[feat]
                        if pd.isna(fv):
                            triggered = False
                            break
                        if op == '>=' and not (fv >= val):
                            triggered = False
                        elif op == '<=' and not (fv <= val):
                            triggered = False
                        elif op == '>' and not (fv > val):
                            triggered = False
                        elif op == '<' and not (fv < val):
                            triggered = False
                    if triggered:
                        buy_price = float(data.iloc[i]['close'])
                        sell_price = float(data.iloc[i + hold_days]['close'])
                        ret = (sell_price - buy_price) / buy_price - cost_rate
                        all_returns.append(ret)
            except Exception:
                continue

        return self._compute_metrics(all_returns, stock_list)

    def _filter_by_regime(self, data, regime):
        """按市场环境过滤数据行"""
        if 'close' not in data.columns:
            return data

        # 计算60日滚动收益来判断大盘环境
        data = data.copy()
        data['rolling_ret_60'] = data['close'].pct_change(60)

        if regime == 'bull':
            return data[data['rolling_ret_60'] > 0.10].copy()
        elif regime == 'bear':
            return data[data['rolling_ret_60'] < -0.10].copy()
        elif regime == 'sideways':
            return data[(data['rolling_ret_60'] >= -0.10) & (data['rolling_ret_60'] <= 0.10)].copy()
        return data

    @staticmethod
    def _compute_metrics(returns_list, stock_list):
        """从收益列表计算回测指标"""
        if not returns_list or len(returns_list) < 5:
            return {
                'win_rate': 0, 'avg_return': 0, 'sharpe': 0,
                'max_drawdown': 0, 'profit_loss_ratio': 0,
                'trades': len(returns_list), 'avg_win': 0, 'avg_lose': 0,
                'stock_count': len(stock_list), 'sample_size': len(returns_list),
            }

        arr = np.array(returns_list) * 100  # 转为百分比
        wins = arr[arr > 0]
        losses = arr[arr <= 0]

        win_rate = len(wins) / len(arr) * 100 if len(arr) > 0 else 0
        avg_ret = float(np.mean(arr))
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
        avg_lose = float(np.mean(losses)) if len(losses) > 0 else 0
        pl_ratio = abs(avg_win / avg_lose) if avg_lose != 0 else 0

        # 夏普比率 (年化)
        if np.std(arr) > 0:
            sharpe = float(np.mean(arr) / np.std(arr) * np.sqrt(252 / 10))
        else:
            sharpe = 0

        # 最大回撤
        cum = np.cumsum(arr)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        max_dd = float(np.min(dd)) if len(dd) > 0 else 0

        return {
            'win_rate': round(win_rate, 2),
            'avg_return': round(avg_ret, 2),
            'sharpe': round(sharpe, 2),
            'max_drawdown': round(max_dd, 2),
            'profit_loss_ratio': round(pl_ratio, 2),
            'trades': len(arr),
            'avg_win': round(avg_win, 2),
            'avg_lose': round(avg_lose, 2),
            'stock_count': len(stock_list),
            'sample_size': len(arr),
        }

    # ============================================================
    # 核心: 维度分析（批量运行）
    # ============================================================
    def run_dimension_analysis(self, dimension: str, max_per_group: int = 40,
                                hold_days: int = 10, strategies=None,
                                progress_callback=None):
        """
        对指定维度进行全面策略分析

        参数:
            dimension: 维度名
            max_per_group: 每组采样股票数
            hold_days: 持有天数
            strategies: 策略列表 (默认用AI_STRATEGIES)
            progress_callback: fn(current, total, group_name, strategy_name)

        返回:
            dict: {
                'dimension': str,
                'groups': {group_name: group_info},
                'matrix': pd.DataFrame (group × strategy → metrics),
                'best_by_group': {group_name: best_strategy_info},
                'insights': [str, ...],
                'run_id': int,
            }
        """
        if strategies is None:
            strategies = AI_STRATEGIES

        # 获取分组
        groups = self.get_stock_groups(dimension, max_per_group)
        if not groups:
            return {'error': '无法获取分组数据，请确保已缓存历史数据'}

        # 记录运行
        run_id = self._log_run_start(dimension, len(groups), max_per_group)

        total_tasks = len(groups) * len(strategies)
        current = 0
        results = {}
        matrix_rows = []

        is_regime = (dimension == 'market_regime')

        for gname, stock_list in groups.items():
            group_results = {}

            # 确定大盘环境标记
            regime = None
            if is_regime:
                if '牛市' in gname:
                    regime = 'bull'
                elif '熊市' in gname:
                    regime = 'bear'
                elif '震荡' in gname:
                    regime = 'sideways'

            for strat in strategies:
                current += 1
                if progress_callback:
                    progress_callback(current, total_tasks, gname, strat['name'])

                metrics = self.backtest_strategy_on_group(
                    stock_list, strat, hold_days=hold_days, regime=regime
                )
                group_results[strat['id']] = {
                    'name': strat['name'],
                    'tier': strat['tier'],
                    **metrics,
                }

                # 保存到DB
                self._save_result(dimension, gname, strat, metrics, run_id)

                matrix_rows.append({
                    '分组': gname,
                    '策略': strat['name'],
                    '策略ID': strat['id'],
                    '胜率': metrics['win_rate'],
                    '收益': metrics['avg_return'],
                    '夏普': metrics['sharpe'],
                    '回撤': metrics['max_drawdown'],
                    '盈亏比': metrics['profit_loss_ratio'],
                    '交易数': metrics['trades'],
                })

            results[gname] = {
                'stock_count': len(stock_list),
                'strategies': group_results,
            }

        # 构建矩阵DataFrame
        matrix_df = pd.DataFrame(matrix_rows)

        # 找出每个分组的最佳策略
        best_by_group = {}
        for gname, ginfo in results.items():
            best_strat = None
            best_score = -999
            for sid, sinfo in ginfo['strategies'].items():
                # 综合评分 = 胜率×0.4 + 夏普×15 + 每笔收益×2
                score = sinfo['win_rate'] * 0.4 + sinfo['sharpe'] * 15 + sinfo['avg_return'] * 2
                if sinfo['trades'] < 10:
                    score *= 0.3  # 样本不足惩罚
                if score > best_score:
                    best_score = score
                    best_strat = sinfo
            best_by_group[gname] = best_strat

        # 生成洞察
        insights = self._generate_insights(dimension, results, best_by_group, matrix_df)

        # 更新运行状态
        self._log_run_complete(run_id, len(groups))

        return {
            'dimension': dimension,
            'dimension_name': DIMENSIONS.get(dimension, {}).get('name', dimension),
            'groups': results,
            'matrix': matrix_df,
            'best_by_group': best_by_group,
            'insights': insights,
            'run_id': run_id,
        }

    # ============================================================
    # 参数优化
    # ============================================================
    def optimize_parameters(self, stock_list, base_strategy_id='ai_core_01',
                             param_name='threshold', param_range=None,
                             hold_days=10, progress_callback=None):
        """
        对单个策略进行参数优化

        返回:
            list[dict]: 每个参数值的回测结果
        """
        base = None
        for s in AI_STRATEGIES:
            if s['id'] == base_strategy_id:
                base = s.copy()
                break
        if base is None:
            return []

        # 根据策略类型确定参数范围
        if param_range is None:
            conditions = base['conditions']
            # 找到主要条件的下限（如ma30_diff <= -0.0962）
            param_range = []
            for feat, op, val in conditions:
                if op == '<=':
                    # 在原始值附近搜索
                    for factor in [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8, 2.0]:
                        param_range.append((feat, op, val * factor))

        results = []
        total = len(param_range)
        for i, (feat, op, new_val) in enumerate(param_range):
            if progress_callback:
                progress_callback(i + 1, total, feat, f'{new_val:.4f}')

            # 修改条件
            modified = base.copy()
            modified['conditions'] = []
            for f, o, v in base['conditions']:
                if f == feat and o == op:
                    modified['conditions'].append((f, o, new_val))
                else:
                    modified['conditions'].append((f, o, v))

            metrics = self.backtest_strategy_on_group(stock_list, modified, hold_days)
            results.append({
                'param_name': feat,
                'param_op': op,
                'param_value': new_val,
                'original_value': next(v for f, o, v in base['conditions'] if f == feat and o == op),
                **metrics,
            })

        return sorted(results, key=lambda x: x['sharpe'], reverse=True)

    # ============================================================
    # 综合分析（一键全维度）
    # ============================================================
    def run_full_analysis(self, dimensions=None, max_per_group=30,
                           hold_days=10, progress_callback=None):
        """
        运行所有维度的分析

        返回:
            dict: {dimension: analysis_result}
        """
        if dimensions is None:
            dimensions = ['industry', 'market_cap', 'volatility', 'price_range', 'trend']

        all_results = {}
        total_dims = len(dimensions)

        for idx, dim in enumerate(dimensions):
            if progress_callback:
                progress_callback(idx, total_dims, dim, '开始分析...')

            result = self.run_dimension_analysis(
                dim, max_per_group=max_per_group, hold_days=hold_days,
                progress_callback=progress_callback
            )
            all_results[dim] = result

        return all_results

    # ============================================================
    # 获取历史分析结果
    # ============================================================
    def get_latest_results(self, dimension: str = None):
        """获取最近一次分析结果"""
        conn = sqlite3.connect(self.lab_db)
        if dimension:
            runs = pd.read_sql_query(
                'SELECT * FROM lab_runs WHERE dimension=? AND status="completed" ORDER BY id DESC LIMIT 1',
                conn, params=[dimension])
        else:
            runs = pd.read_sql_query(
                'SELECT * FROM lab_runs WHERE status="completed" ORDER BY id DESC LIMIT 1', conn)

        if runs.empty:
            conn.close()
            return None

        run = runs.iloc[0]
        dim = run['dimension']
        created_at = run['started_at']

        results_df = pd.read_sql_query(
            'SELECT * FROM lab_results WHERE dimension=? AND created_at >= ?',
            conn, params=[dim, created_at])
        conn.close()

        if results_df.empty:
            return None

        # 重建matrix
        matrix_rows = []
        for _, r in results_df.iterrows():
            matrix_rows.append({
                '分组': r['group_name'],
                '策略': r['strategy_name'],
                '策略ID': r['strategy_id'],
                '胜率': r['win_rate'],
                '收益': r['avg_return'],
                '夏普': r['sharpe'],
                '回撤': r['max_drawdown'],
                '盈亏比': r['profit_loss_ratio'],
                '交易数': r['trades'],
            })

        matrix_df = pd.DataFrame(matrix_rows)

        # 找最佳
        best_by_group = {}
        for gname in matrix_df['分组'].unique():
            gdf = matrix_df[matrix_df['分组'] == gname]
            if not gdf.empty:
                best_idx = (gdf['胜率'] * 0.4 + gdf['夏普'] * 15 + gdf['收益'] * 2).idxmax()
                best_row = gdf.loc[best_idx]
                best_by_group[gname] = {
                    'name': best_row['策略'],
                    'win_rate': best_row['胜率'],
                    'sharpe': best_row['夏普'],
                    'avg_return': best_row['收益'],
                    'trades': best_row['交易数'],
                }

        insights = self._generate_insights_from_matrix(dim, matrix_df, best_by_group)

        return {
            'dimension': dim,
            'dimension_name': DIMENSIONS.get(dim, {}).get('name', dim),
            'matrix': matrix_df,
            'best_by_group': best_by_group,
            'insights': insights,
            'run_info': run.to_dict(),
        }

    def get_all_run_history(self):
        """获取所有运行历史"""
        conn = sqlite3.connect(self.lab_db)
        df = pd.read_sql_query(
            'SELECT * FROM lab_runs ORDER BY id DESC LIMIT 50', conn)
        conn.close()
        return df

    # ============================================================
    # 洞察生成
    # ============================================================
    def _generate_insights(self, dimension, results, best_by_group, matrix_df):
        """根据分析结果生成洞察"""
        insights = []

        if matrix_df.empty:
            return ['数据不足，无法生成洞察']

        # 1. 整体最佳策略
        if not matrix_df.empty and matrix_df['交易数'].sum() > 0:
            valid = matrix_df[matrix_df['交易数'] >= 10]
            if not valid.empty:
                best_idx = valid['夏普'].idxmax()
                best = valid.loc[best_idx]
                insights.append(
                    f"🏆 整体最优：「{best['策略']}」在「{best['分组']}」中表现最佳 "
                    f"(胜率{best['胜率']:.1f}%, 夏普{best['夏普']:.2f}, 收益{best['收益']:.2f}%)"
                )

        # 2. 按分组找差异
        group_best_rates = [(g, info['name'], info.get('win_rate', 0))
                            for g, info in best_by_group.items() if info]
        if len(group_best_rates) >= 2:
            group_best_rates.sort(key=lambda x: x[2], reverse=True)
            top = group_best_rates[0]
            bot = group_best_rates[-1]
            if top[2] - bot[2] > 5:
                insights.append(
                    f"📊 分组差异：「{top[0]}」最佳胜率{top[2]:.1f}% vs 「{bot[0]}」最低{bot[2]:.1f}%，"
                    f"差距{top[2]-bot[2]:.1f}个百分点"
                )

        # 3. 策略普适性
        strat_counts = defaultdict(list)
        for g, info in best_by_group.items():
            if info:
                strat_counts[info['name']].append(g)
        if strat_counts:
            most_common = max(strat_counts.items(), key=lambda x: len(x[1]))
            insights.append(
                f"🔄 最普适策略：「{most_common[0]}」在 {len(most_common[1])}/{len(best_by_group)} 个分组中表现最好"
            )

        # 4. 样本充足性
        low_sample = matrix_df[matrix_df['交易数'] < 20]
        if len(low_sample) > 0:
            insights.append(
                f"⚠️ 注意：{len(low_sample)} 项测试交易次数 < 20，结果可靠性有限"
            )

        # 5. 高胜率发现
        high_wr = matrix_df[(matrix_df['胜率'] > 70) & (matrix_df['交易数'] >= 20)]
        if not high_wr.empty:
            insights.append(
                f"🎯 高胜率发现：{len(high_wr)} 项组合胜率超过70%（且交易>=20次）"
            )

        return insights

    def _generate_insights_from_matrix(self, dimension, matrix_df, best_by_group):
        """从矩阵数据生成洞察（用于缓存结果）"""
        return self._generate_insights(dimension, {}, best_by_group, matrix_df)

    # ============================================================
    # 数据库操作
    # ============================================================
    def _log_run_start(self, dimension, total_groups, sample_per_group):
        conn = sqlite3.connect(self.lab_db)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO lab_runs (dimension, status, total_groups, sample_per_group, started_at) VALUES (?,?,?,?,?)',
            (dimension, 'running', total_groups, sample_per_group, datetime.now().isoformat())
        )
        run_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return run_id

    def _log_run_complete(self, run_id, completed_groups):
        conn = sqlite3.connect(self.lab_db)
        conn.execute(
            'UPDATE lab_runs SET status=?, completed_groups=?, completed_at=? WHERE id=?',
            ('completed', completed_groups, datetime.now().isoformat(), run_id)
        )
        conn.commit()
        conn.close()

    def _save_result(self, dimension, group_name, strategy, metrics, run_id):
        conn = sqlite3.connect(self.lab_db)
        try:
            conn.execute('''
                INSERT OR REPLACE INTO lab_results
                (dimension, group_name, strategy_id, strategy_name,
                 win_rate, avg_return, sharpe, max_drawdown, profit_loss_ratio,
                 trades, avg_win, avg_lose, stock_count, sample_size, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                dimension, group_name, strategy['id'], strategy['name'],
                metrics['win_rate'], metrics['avg_return'], metrics['sharpe'],
                metrics['max_drawdown'], metrics['profit_loss_ratio'],
                metrics['trades'], metrics['avg_win'], metrics['avg_lose'],
                metrics['stock_count'], metrics['sample_size'],
                datetime.now().isoformat()
            ))
            conn.commit()
        except Exception:
            pass
        finally:
            conn.close()
