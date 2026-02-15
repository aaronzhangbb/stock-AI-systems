# -*- coding: utf-8 -*-
"""
板块轮动热度模块

获取行业板块的涨跌排名和资金流向, 为个股推荐提供板块维度参考信息.

功能:
1. fetch_sector_ranking()  - 板块涨跌幅排名
2. fetch_sector_fund_flow() - 板块资金净流入排名
3. get_sector_heat_map()   - 综合板块热度(涨幅+资金)
4. enrich_recommendations_with_sector() - 为推荐标注板块热度

设计原则: 板块数据仅做展示标签, 不做硬性过滤
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


# ============================================================
# 1. 板块涨跌幅排名
# ============================================================

def fetch_sector_ranking(cache_minutes=60):
    """
    获取行业板块涨跌幅排名

    数据来源: stock_board_industry_name_em (东方财富行业板块)

    返回 DataFrame:
        board_name, board_code, close, pct_change, total_mv,
        turnover, up_count, down_count, leader_name, leader_pct
    """
    cache_path = os.path.join(DATA_DIR, '_cache_sector_ranking.pkl')
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(cache_path):
        try:
            mtime = os.path.getmtime(cache_path)
            age_min = (time.time() - mtime) / 60
            if age_min < cache_minutes:
                df = pd.read_pickle(cache_path)
                logger.info(f"[板块] 使用缓存板块排名 ({len(df)} 个, {age_min:.0f}分钟前)")
                return df
        except Exception:
            pass

    import akshare as ak
    logger.info("[板块] 正在获取行业板块排名...")
    t0 = time.time()

    try:
        raw = ak.stock_board_industry_name_em()
    except Exception as e:
        logger.warning(f"[板块] 获取板块排名失败: {e}")
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    df = pd.DataFrame({
        'rank': raw['排名'],
        'board_name': raw['板块名称'].astype(str),
        'board_code': raw['板块代码'].astype(str),
        'close': pd.to_numeric(raw['最新价'], errors='coerce'),
        'pct_change': pd.to_numeric(raw['涨跌幅'], errors='coerce'),
        'total_mv': pd.to_numeric(raw['总市值'], errors='coerce') / 1e8,
        'turnover': pd.to_numeric(raw['换手率'], errors='coerce'),
        'up_count': pd.to_numeric(raw['上涨家数'], errors='coerce'),
        'down_count': pd.to_numeric(raw['下跌家数'], errors='coerce'),
        'leader_name': raw['领涨股票'].astype(str),
        'leader_pct': pd.to_numeric(raw['领涨股票-涨跌幅'], errors='coerce'),
    })

    df = df.sort_values('pct_change', ascending=False).reset_index(drop=True)

    try:
        df.to_pickle(cache_path)
    except Exception:
        pass

    elapsed = time.time() - t0
    logger.info(f"[板块] 板块排名获取完成: {len(df)} 个板块, 耗时 {elapsed:.1f}s")
    return df


# ============================================================
# 2. 板块资金流排名
# ============================================================

def fetch_sector_fund_flow(cache_minutes=60):
    """
    获取行业板块资金净流入排名

    数据来源: stock_sector_fund_flow_rank (东方财富)
    """
    cache_path = os.path.join(DATA_DIR, '_cache_sector_fund_flow.pkl')
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(cache_path):
        try:
            mtime = os.path.getmtime(cache_path)
            age_min = (time.time() - mtime) / 60
            if age_min < cache_minutes:
                df = pd.read_pickle(cache_path)
                logger.info(f"[板块] 使用缓存板块资金流 ({len(df)} 个)")
                return df
        except Exception:
            pass

    import akshare as ak
    logger.info("[板块] 正在获取板块资金流向...")
    t0 = time.time()

    try:
        raw = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
    except Exception as e:
        logger.warning(f"[板块] 获取板块资金流失败: {e}")
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    df = pd.DataFrame({
        'board_name': raw['名称'].astype(str),
        'pct_change': pd.to_numeric(raw['今日涨跌幅'], errors='coerce'),
        'main_net_flow': pd.to_numeric(raw['今日主力净流入-净额'], errors='coerce') / 1e8,
        'main_net_pct': pd.to_numeric(raw['今日主力净流入-净占比'], errors='coerce'),
    })

    df = df.sort_values('main_net_flow', ascending=False).reset_index(drop=True)

    try:
        df.to_pickle(cache_path)
    except Exception:
        pass

    elapsed = time.time() - t0
    logger.info(f"[板块] 板块资金流获取完成: {len(df)} 个板块, 耗时 {elapsed:.1f}s")
    return df


# ============================================================
# 3. 综合板块热度
# ============================================================

def get_sector_heat_map():
    """
    综合板块涨幅和资金流计算板块热度

    返回:
        {
            'hot_sectors': Top10热门板块列表,
            'cold_sectors': Top10冷门板块列表,
            'sector_map': {板块名: {pct_change, fund_flow, heat_score}},
            'fetch_time': 获取时间,
        }
    """
    ranking = fetch_sector_ranking()
    fund_flow = fetch_sector_fund_flow()

    sector_map = {}

    # 从涨幅排名获取数据
    if not ranking.empty:
        for _, row in ranking.iterrows():
            name = row['board_name']
            sector_map[name] = {
                'pct_change': float(row.get('pct_change', 0) or 0),
                'up_count': int(row.get('up_count', 0) or 0),
                'down_count': int(row.get('down_count', 0) or 0),
                'leader_name': str(row.get('leader_name', '')),
                'main_net_flow': 0.0,
                'main_net_pct': 0.0,
            }

    # 合并资金流数据
    if not fund_flow.empty:
        for _, row in fund_flow.iterrows():
            name = row['board_name']
            if name in sector_map:
                sector_map[name]['main_net_flow'] = float(row.get('main_net_flow', 0) or 0)
                sector_map[name]['main_net_pct'] = float(row.get('main_net_pct', 0) or 0)

    # 计算热度分 (涨幅权重60% + 资金流权重40%)
    for name, info in sector_map.items():
        pct = info['pct_change']
        flow_pct = info['main_net_pct']

        # 涨幅分: -5%~+5% 映射到 0~100
        pct_score = max(0, min(100, 50 + pct * 10))

        # 资金分: -5%~+5% 映射到 0~100
        flow_score = max(0, min(100, 50 + flow_pct * 10))

        heat = pct_score * 0.6 + flow_score * 0.4
        info['heat_score'] = round(heat, 1)

    # 排序
    sorted_sectors = sorted(sector_map.items(), key=lambda x: x[1]['heat_score'], reverse=True)

    hot_sectors = []
    for name, info in sorted_sectors[:10]:
        hot_sectors.append({
            'name': name,
            'pct_change': info['pct_change'],
            'main_net_flow': info['main_net_flow'],
            'heat_score': info['heat_score'],
            'leader': info.get('leader_name', ''),
        })

    cold_sectors = []
    for name, info in sorted_sectors[-10:]:
        cold_sectors.append({
            'name': name,
            'pct_change': info['pct_change'],
            'main_net_flow': info['main_net_flow'],
            'heat_score': info['heat_score'],
        })

    return {
        'hot_sectors': hot_sectors,
        'cold_sectors': cold_sectors,
        'sector_map': sector_map,
        'total_sectors': len(sector_map),
        'fetch_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


# ============================================================
# 4. 为推荐列表标注板块热度
# ============================================================

def enrich_recommendations_with_sector(recommendations, heat_data=None):
    """
    为推荐列表添加板块热度标签

    利用推荐中已有的 board_name 字段匹配板块热度
    """
    if not recommendations:
        return recommendations

    if heat_data is None:
        try:
            heat_data = get_sector_heat_map()
        except Exception as e:
            logger.warning(f"[板块] 获取板块热度失败: {e}")
            return recommendations

    sector_map = heat_data.get('sector_map', {})
    total = heat_data.get('total_sectors', 1)

    for rec in recommendations:
        board_name = rec.get('board_name', '')
        sector_info = {}

        if board_name and board_name in sector_map:
            s = sector_map[board_name]
            sector_info = {
                'board_name': board_name,
                'board_pct': s.get('pct_change', 0),
                'board_fund_flow': s.get('main_net_flow', 0),
                'board_heat': s.get('heat_score', 50),
            }

            # 板块热度等级
            heat = s.get('heat_score', 50)
            if heat >= 75:
                sector_info['heat_level'] = '热门'
            elif heat >= 55:
                sector_info['heat_level'] = '偏暖'
            elif heat >= 45:
                sector_info['heat_level'] = '中性'
            elif heat >= 30:
                sector_info['heat_level'] = '偏冷'
            else:
                sector_info['heat_level'] = '冰点'

            # 板块标签
            pct_str = f"{s.get('pct_change', 0):+.2f}%"
            flow = s.get('main_net_flow', 0)
            flow_str = f"资金{flow:+.1f}亿" if flow != 0 else ""
            sector_info['label'] = f"{board_name}({pct_str} {flow_str})".strip()
        else:
            sector_info = {
                'board_name': board_name or '未知',
                'board_heat': 50,
                'heat_level': '未知',
                'label': board_name or '板块未知',
            }

        rec['sector'] = sector_info

    return recommendations


def format_sector_summary(heat_data):
    """格式化板块热度摘要(用于邮件)"""
    lines = []
    lines.append("")
    lines.append("=" * 45)
    lines.append("  板块热度概览")
    lines.append("=" * 45)

    hot = heat_data.get('hot_sectors', [])
    cold = heat_data.get('cold_sectors', [])

    if hot:
        lines.append("  [热门板块 Top5]")
        for s in hot[:5]:
            name = s['name']
            pct = s.get('pct_change', 0)
            flow = s.get('main_net_flow', 0)
            leader = s.get('leader', '')
            lines.append(f"    {name}: {pct:+.2f}%  资金{flow:+.1f}亿  领涨:{leader}")

    if cold:
        lines.append("  [冷门板块 Bottom5]")
        for s in cold[-5:]:
            name = s['name']
            pct = s.get('pct_change', 0)
            flow = s.get('main_net_flow', 0)
            lines.append(f"    {name}: {pct:+.2f}%  资金{flow:+.1f}亿")

    lines.append("")
    return "\n".join(lines)
