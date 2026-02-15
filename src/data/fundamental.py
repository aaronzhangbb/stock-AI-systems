# -*- coding: utf-8 -*-
"""
基本面数据模块

提供两层数据:
1. 批量层: 一次性获取全市场PE/PB/总市值 (from stock_zh_a_spot_em)
2. 详细层: 为候选股票逐只获取ROE/利润增速/营收增速 (from stock_financial_abstract_ths)

设计原则:
- 批量层做为特征输入或软性筛选(不硬淘汰)
- 详细层做为推荐标签展示(给用户看的辅助信息)
"""

import os
import sys
import time
import logging
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


def _safe_float(val, default=None):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


# ============================================================
# 1. 批量数据 - 全市场 PE / PB / 市值
# ============================================================

def fetch_bulk_valuation(cache_minutes=60):
    """一次性获取全市场PE/PB/市值等估值数据(带文件缓存)"""
    cache_path = os.path.join(DATA_DIR, '_cache_bulk_valuation.pkl')
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(cache_path):
        try:
            mtime = os.path.getmtime(cache_path)
            age_min = (time.time() - mtime) / 60
            if age_min < cache_minutes:
                df = pd.read_pickle(cache_path)
                logger.info(f"[基本面] 使用缓存估值数据 ({len(df)} 只, {age_min:.0f}分钟前)")
                return df
        except Exception:
            pass

    import akshare as ak
    logger.info("[基本面] 正在获取全市场估值数据...")
    t0 = time.time()

    try:
        raw = ak.stock_zh_a_spot_em()
    except Exception as e:
        logger.warning(f"[基本面] 获取全市场数据失败: {e}")
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    df = pd.DataFrame({
        'stock_code': raw['代码'].astype(str),
        'stock_name': raw['名称'].astype(str),
        'close': pd.to_numeric(raw['最新价'], errors='coerce'),
        'pct_change': pd.to_numeric(raw['涨跌幅'], errors='coerce'),
        'pe': pd.to_numeric(raw['市盈率-动态'], errors='coerce'),
        'pb': pd.to_numeric(raw['市净率'], errors='coerce'),
        'total_mv': pd.to_numeric(raw['总市值'], errors='coerce') / 1e8,
        'circ_mv': pd.to_numeric(raw['流通市值'], errors='coerce') / 1e8,
        'turnover': pd.to_numeric(raw['换手率'], errors='coerce'),
        'volume_ratio': pd.to_numeric(raw['量比'], errors='coerce'),
    })

    try:
        df.to_pickle(cache_path)
    except Exception:
        pass

    elapsed = time.time() - t0
    logger.info(f"[基本面] 获取完成: {len(df)} 只股票, 耗时 {elapsed:.1f}s")
    return df


def get_valuation_for_stocks(stock_codes):
    """获取指定股票的估值指标"""
    bulk = fetch_bulk_valuation()
    if bulk.empty:
        return {}

    bulk = bulk[bulk['stock_code'].isin(stock_codes)]
    result = {}
    for _, row in bulk.iterrows():
        code = row['stock_code']
        result[code] = {
            'pe': _safe_float(row.get('pe')),
            'pb': _safe_float(row.get('pb')),
            'total_mv': _safe_float(row.get('total_mv')),
            'circ_mv': _safe_float(row.get('circ_mv')),
            'turnover': _safe_float(row.get('turnover')),
            'volume_ratio': _safe_float(row.get('volume_ratio')),
        }
    return result


# ============================================================
# 2. 详细层 - 单只股票ROE/增速 (仅候选股票)
# ============================================================

def fetch_financial_detail(stock_code):
    """获取单只股票的详细财务指标(ROE/利润增速/营收增速等)"""
    import akshare as ak

    try:
        df = ak.stock_financial_abstract_ths(symbol=stock_code, indicator="按年度")
        if df.empty or len(df) < 1:
            return {}

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else None

        def parse_pct(val):
            if val is None or val is False or str(val) == 'False':
                return None
            s = str(val).replace('%', '').strip()
            try:
                return float(s)
            except (ValueError, TypeError):
                return None

        result = {
            'report_period': str(latest.get('报告期', '')),
            'roe': parse_pct(latest.get('净资产收益率')),
            'net_profit_growth': parse_pct(latest.get('净利润同比增长率')),
            'revenue_growth': parse_pct(latest.get('营业总收入同比增长率')),
            'gross_margin': parse_pct(latest.get('销售毛利率')),
            'net_margin': parse_pct(latest.get('销售净利率')),
            'debt_ratio': parse_pct(latest.get('资产负债率')),
        }

        if prev is not None:
            result['roe_prev'] = parse_pct(prev.get('净资产收益率'))

        return result

    except Exception as e:
        logger.debug(f"[基本面] 获取 {stock_code} 财务详情失败: {e}")
        return {}


# ============================================================
# 3. 一键丰富推荐列表
# ============================================================

def enrich_recommendations_with_fundamentals(recommendations, max_detail=30):
    """为推荐列表添加基本面标签(PE/PB + ROE/增速)"""
    if not recommendations:
        return recommendations

    t0 = time.time()
    codes = [r['stock_code'] for r in recommendations]

    # Step 1: 批量估值
    valuation_map = get_valuation_for_stocks(codes)

    # Step 2: 详细财务(仅前 max_detail 只)
    detail_map = {}
    detail_codes = codes[:max_detail]
    for i, code in enumerate(detail_codes):
        try:
            detail = fetch_financial_detail(code)
            if detail:
                detail_map[code] = detail
        except Exception:
            pass

        if i > 0 and i % 10 == 0:
            time.sleep(0.5)
            logger.info(f"[基本面] 详细数据进度: {i}/{len(detail_codes)}")

    # Step 3: 合并到推荐列表
    for rec in recommendations:
        code = rec['stock_code']
        fund = {}

        val = valuation_map.get(code, {})
        fund['pe'] = val.get('pe')
        fund['pb'] = val.get('pb')
        fund['total_mv'] = val.get('total_mv')

        detail = detail_map.get(code, {})
        fund['roe'] = detail.get('roe')
        fund['net_profit_growth'] = detail.get('net_profit_growth')
        fund['revenue_growth'] = detail.get('revenue_growth')
        fund['gross_margin'] = detail.get('gross_margin')
        fund['debt_ratio'] = detail.get('debt_ratio')

        board = rec.get('board_name', '')
        fund['label'] = _build_fundamental_label(fund, board_name=board)
        fund['quality'] = get_fundamental_quality_tag(fund)

        rec['fundamental'] = fund

    elapsed = time.time() - t0
    logger.info(f"[基本面] 推荐标签丰富完成: {len(recommendations)} 只, "
                f"详细数据 {len(detail_map)} 只, 耗时 {elapsed:.1f}s")

    return recommendations


def _build_fundamental_label(fund, board_name=''):
    """根据基本面数据生成一句话标签(含行业相对估值)"""
    parts = []

    pe = fund.get('pe')
    if pe is not None:
        if pe < 0:
            parts.append("PE 亏损")
        elif pe > 500:
            parts.append(f"PE {pe:.0f}(偏高)")
        else:
            try:
                pe_eval, _ = evaluate_pe(pe, board_name)
                parts.append(f"PE {pe:.1f}({pe_eval})")
            except Exception:
                parts.append(f"PE {pe:.1f}")

    pb = fund.get('pb')
    if pb is not None:
        try:
            pb_eval, _ = evaluate_pb(pb, board_name)
            parts.append(f"PB {pb:.2f}({pb_eval})")
        except Exception:
            parts.append(f"PB {pb:.2f}")

    roe = fund.get('roe')
    if roe is not None:
        parts.append(f"ROE {roe:.1f}%")

    growth = fund.get('net_profit_growth')
    if growth is not None:
        parts.append(f"利润{growth:+.1f}%")

    mv = fund.get('total_mv')
    if mv is not None:
        if mv >= 10000:
            parts.append(f"市值{mv/10000:.0f}万亿")
        elif mv >= 100:
            parts.append(f"市值{mv:.0f}亿")
        else:
            parts.append(f"市值{mv:.1f}亿")

    return " | ".join(parts) if parts else "基本面暂缺"


def get_fundamental_quality_tag(fund):
    """
    根据基本面数据给出质量标签(不用来过滤, 仅供展示)
    返回: 'A+' / 'A' / 'B' / 'C' / 'D'
    """
    score = 0
    count = 0

    roe = fund.get('roe')
    if roe is not None:
        count += 1
        if roe >= 15:
            score += 3
        elif roe >= 10:
            score += 2
        elif roe >= 5:
            score += 1
        elif roe < 0:
            score -= 1

    growth = fund.get('net_profit_growth')
    if growth is not None:
        count += 1
        if growth >= 30:
            score += 3
        elif growth >= 10:
            score += 2
        elif growth >= 0:
            score += 1
        elif growth < -20:
            score -= 1

    pe = fund.get('pe')
    if pe is not None:
        count += 1
        if pe < 0:
            score -= 2
        elif pe <= 20:
            score += 2
        elif pe <= 40:
            score += 1

    debt = fund.get('debt_ratio')
    if debt is not None:
        count += 1
        if debt >= 80:
            score -= 1
        elif debt <= 40:
            score += 1

    if count == 0:
        return '-'

    avg = score / count
    if avg >= 2.0:
        return 'A+'
    elif avg >= 1.2:
        return 'A'
    elif avg >= 0.5:
        return 'B'
    elif avg >= -0.3:
        return 'C'
    else:
        return 'D'


# ============================================================
# 4. 行业估值基准 - PE/PB 行业中位数
# ============================================================

_industry_benchmark_cache = {}


def get_industry_benchmarks(cache_minutes=120):
    """计算各行业PE/PB中位数和分位数"""
    global _industry_benchmark_cache
    if _industry_benchmark_cache:
        cache_time = _industry_benchmark_cache.get('_time', 0)
        if (time.time() - cache_time) < cache_minutes * 60:
            return _industry_benchmark_cache

    cache_path = os.path.join(DATA_DIR, '_cache_industry_benchmarks.json')
    if os.path.exists(cache_path):
        try:
            import json
            mtime = os.path.getmtime(cache_path)
            if (time.time() - mtime) / 60 < cache_minutes:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                cached['_time'] = mtime
                _industry_benchmark_cache = cached
                return cached
        except Exception:
            pass

    logger.info("[基本面] 计算行业估值基准...")
    bulk = fetch_bulk_valuation()
    if bulk.empty:
        return {}

    import sqlite3
    pool_db = os.path.join(DATA_DIR, 'stock_pool.db')
    if not os.path.exists(pool_db):
        return {}
    try:
        conn = sqlite3.connect(pool_db)
        pool_df = pd.read_sql_query(
            "SELECT stock_code, board_name FROM all_stocks WHERE board_name != ''", conn)
        conn.close()
    except Exception:
        return {}

    merged = bulk.merge(pool_df, on='stock_code', how='inner')
    valid_pe = merged[merged['pe'].notna() & (merged['pe'] > 0) & (merged['pe'] < 2000)]
    valid_pb = merged[merged['pb'].notna() & (merged['pb'] > 0) & (merged['pb'] < 200)]

    benchmarks = {}
    for board, group in valid_pe.groupby('board_name'):
        if len(group) < 5:
            continue
        pe_vals = group['pe'].dropna()
        entry = benchmarks.get(board, {})
        entry['pe_median'] = round(float(pe_vals.median()), 1)
        entry['pe_q25'] = round(float(pe_vals.quantile(0.25)), 1)
        entry['pe_q75'] = round(float(pe_vals.quantile(0.75)), 1)
        benchmarks[board] = entry

    for board, group in valid_pb.groupby('board_name'):
        if len(group) < 5:
            continue
        pb_vals = group['pb'].dropna()
        entry = benchmarks.get(board, {})
        entry['pb_median'] = round(float(pb_vals.median()), 2)
        entry['pb_q25'] = round(float(pb_vals.quantile(0.25)), 2)
        entry['pb_q75'] = round(float(pb_vals.quantile(0.75)), 2)
        benchmarks[board] = entry

    try:
        import json
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(benchmarks, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    benchmarks['_time'] = time.time()
    _industry_benchmark_cache = benchmarks
    return benchmarks


def evaluate_pe(pe_value, board_name, benchmarks=None):
    """根据行业基准评价PE: 返回 (评价文字, 颜色)"""
    if pe_value is None:
        return ('--', '#7a869a')
    if pe_value < 0:
        return ('亏损', '#ef4444')
    if benchmarks is None:
        benchmarks = get_industry_benchmarks()
    ind = benchmarks.get(board_name, {})
    q25 = ind.get('pe_q25')
    q75 = ind.get('pe_q75')
    if q25 is None:
        if pe_value <= 15:
            return ('偏低', '#4ade80')
        elif pe_value <= 35:
            return ('合理', '#94a3b8')
        elif pe_value <= 60:
            return ('偏高', '#f97316')
        else:
            return ('很高', '#ef4444')
    if pe_value <= q25:
        return ('偏低', '#4ade80')
    elif pe_value <= q75:
        return ('合理', '#94a3b8')
    elif pe_value <= q75 * 1.5:
        return ('偏高', '#f97316')
    else:
        return ('很高', '#ef4444')


def evaluate_pb(pb_value, board_name, benchmarks=None):
    """根据行业基准评价PB: 返回 (评价文字, 颜色)"""
    if pb_value is None:
        return ('--', '#7a869a')
    if pb_value < 1:
        return ('破净', '#60a5fa')
    if benchmarks is None:
        benchmarks = get_industry_benchmarks()
    ind = benchmarks.get(board_name, {})
    q25 = ind.get('pb_q25')
    q75 = ind.get('pb_q75')
    if q25 is None:
        if pb_value <= 2:
            return ('偏低', '#4ade80')
        elif pb_value <= 5:
            return ('合理', '#94a3b8')
        elif pb_value <= 10:
            return ('偏高', '#f97316')
        else:
            return ('很高', '#ef4444')
    if pb_value <= q25:
        return ('偏低', '#4ade80')
    elif pb_value <= q75:
        return ('合理', '#94a3b8')
    elif pb_value <= q75 * 1.5:
        return ('偏高', '#f97316')
    else:
        return ('很高', '#ef4444')
