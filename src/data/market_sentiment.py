# -*- coding: utf-8 -*-
"""
大盘情绪指标模块

采集并计算以下维度的市场情绪：
1. 涨跌家数比 / 涨停跌停数 — 市场活跃度
2. 两市成交额 / 成交额变化 — 资金活跃度
3. 主力资金流向 — 大单/超大单净流入
4. 北向资金 — 外资态度（"聪明钱"）
5. 融资余额变化 — 杠杆情绪

综合计算 0~100 的市场情绪分数：
  0~20  = 极度恐慌（可能是抄底机会）
  20~40 = 偏悲观
  40~60 = 中性
  60~80 = 偏乐观
  80~100 = 极度贪婪（注意风险）
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


def _safe_float(val, default=0.0):
    """安全转换为浮点数"""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


# ============================================================
# 1. 市场活跃度（涨跌家数、涨停跌停）
# ============================================================

def fetch_market_activity() -> dict:
    """
    获取当日市场涨跌活跃度

    返回:
        {
            'up_count': 上涨家数,
            'down_count': 下跌家数,
            'flat_count': 平盘家数,
            'limit_up': 涨停数,
            'limit_down': 跌停数,
            'up_down_ratio': 涨跌比,
            'activity_pct': 活跃度百分比,
            'date': 统计日期,
        }
    """
    import akshare as ak

    try:
        df = ak.stock_market_activity_legu()
        data = {}
        for _, row in df.iterrows():
            item = str(row['item']).strip()
            val = row['value']
            if item == '上涨':
                data['up_count'] = int(_safe_float(val))
            elif item == '下跌':
                data['down_count'] = int(_safe_float(val))
            elif item == '平盘':
                data['flat_count'] = int(_safe_float(val))
            elif item == '涨停':
                data['limit_up'] = int(_safe_float(val))
            elif item == '跌停':
                data['limit_down'] = int(_safe_float(val))
            elif item == '活跃度':
                s = str(val).replace('%', '')
                data['activity_pct'] = _safe_float(s)
            elif item == '统计日期':
                data['date'] = str(val)[:10]

        # 计算涨跌比
        up = data.get('up_count', 0)
        down = data.get('down_count', 1)
        data['up_down_ratio'] = round(up / max(down, 1), 2)

        return data
    except Exception as e:
        logger.warning(f"[情绪] 获取市场活跃度失败: {e}")
        return {}


# ============================================================
# 2. 两市成交额（从上证指数日K获取）
# ============================================================

def fetch_market_volume(days: int = 30) -> dict:
    """
    获取近N天两市成交额及变化趋势

    返回:
        {
            'today_amount': 今日成交额（亿）,
            'avg_5d': 5日均成交额,
            'avg_10d': 10日均成交额,
            'avg_20d': 20日均成交额,
            'vol_ratio_5d': 今日/5日均量比,
            'vol_ratio_20d': 今日/20日均量比,
            'trend': '放量'/'缩量'/'正常',
        }
    """
    import akshare as ak

    try:
        end = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=days + 15)).strftime('%Y%m%d')

        # 上证指数成交额
        sh = ak.index_zh_a_hist(symbol='000001', period='daily',
                                start_date=start, end_date=end)
        # 深证成指成交额
        sz = ak.index_zh_a_hist(symbol='399001', period='daily',
                                start_date=start, end_date=end)

        if sh.empty or sz.empty:
            return {}

        # 合并两市成交额
        sh = sh.sort_values('日期').tail(days)
        sz = sz.sort_values('日期').tail(days)

        # 对齐日期
        sh = sh.set_index('日期')
        sz = sz.set_index('日期')
        common = sh.index.intersection(sz.index)
        sh = sh.loc[common]
        sz = sz.loc[common]

        # 两市总成交额（转亿元）
        total_amount = (sh['成交额'].astype(float) + sz['成交额'].astype(float)) / 1e8

        if len(total_amount) == 0:
            return {}

        today_amt = float(total_amount.iloc[-1])
        avg_5d = float(total_amount.tail(5).mean())
        avg_10d = float(total_amount.tail(10).mean())
        avg_20d = float(total_amount.tail(20).mean())

        vol_ratio_5d = today_amt / avg_5d if avg_5d > 0 else 1.0
        vol_ratio_20d = today_amt / avg_20d if avg_20d > 0 else 1.0

        if vol_ratio_5d >= 1.3:
            trend = '显著放量'
        elif vol_ratio_5d >= 1.1:
            trend = '温和放量'
        elif vol_ratio_5d <= 0.7:
            trend = '显著缩量'
        elif vol_ratio_5d <= 0.9:
            trend = '温和缩量'
        else:
            trend = '正常'

        return {
            'today_amount': round(today_amt, 1),
            'avg_5d': round(avg_5d, 1),
            'avg_10d': round(avg_10d, 1),
            'avg_20d': round(avg_20d, 1),
            'vol_ratio_5d': round(vol_ratio_5d, 2),
            'vol_ratio_20d': round(vol_ratio_20d, 2),
            'trend': trend,
            'amount_series': [round(float(x), 1) for x in total_amount.tail(10).values],
        }
    except Exception as e:
        logger.warning(f"[情绪] 获取成交额失败: {e}")
        return {}


# ============================================================
# 3. 主力资金流向
# ============================================================

def fetch_main_fund_flow() -> dict:
    """
    获取主力资金净流入/流出

    返回:
        {
            'main_net_flow': 主力净流入（亿元）,
            'main_net_pct': 主力净流入占比(%),
            'super_large_net': 超大单净流入（亿元）,
            'large_net': 大单净流入（亿元）,
            'trend_3d': 近3日主力资金趋势,
        }
    """
    import akshare as ak

    try:
        df = ak.stock_market_fund_flow()
        if df.empty:
            return {}

        df = df.sort_values('日期').tail(10)

        # 最新一天
        latest = df.iloc[-1]
        main_net = _safe_float(latest.get('主力净流入-净额')) / 1e8  # 转亿
        main_pct = _safe_float(latest.get('主力净流入-净占比'))
        super_net = _safe_float(latest.get('超大单净流入-净额')) / 1e8
        large_net = _safe_float(latest.get('大单净流入-净额')) / 1e8

        # 近3日趋势
        recent_3 = df.tail(3)
        main_3d = sum(_safe_float(r.get('主力净流入-净额')) for _, r in recent_3.iterrows()) / 1e8

        if main_3d > 50:
            trend_3d = '大幅净流入'
        elif main_3d > 0:
            trend_3d = '小幅净流入'
        elif main_3d > -50:
            trend_3d = '小幅净流出'
        else:
            trend_3d = '大幅净流出'

        return {
            'main_net_flow': round(main_net, 1),
            'main_net_pct': round(main_pct, 2),
            'super_large_net': round(super_net, 1),
            'large_net': round(large_net, 1),
            'main_3d_total': round(main_3d, 1),
            'trend_3d': trend_3d,
        }
    except Exception as e:
        logger.warning(f"[情绪] 获取主力资金流向失败: {e}")
        return {}


# ============================================================
# 4. 北向资金
# ============================================================

def fetch_northbound_flow() -> dict:
    """
    获取北向资金（沪股通+深股通）净买入

    返回:
        {
            'today_net': 今日北向净买入（亿元）,
            'sh_connect': 沪股通净买入,
            'sz_connect': 深股通净买入,
            'status': 交易状态描述,
        }
    """
    import akshare as ak

    try:
        df = ak.stock_hsgt_fund_flow_summary_em()
        if df.empty:
            return {}

        result = {'status': '正常'}

        for _, row in df.iterrows():
            board = str(row.get('板块', ''))
            direction = str(row.get('资金方向', ''))
            net = _safe_float(row.get('成交净买额'))

            if direction == '北向':
                if '沪股通' in board:
                    result['sh_connect'] = round(net, 2)
                elif '深股通' in board:
                    result['sz_connect'] = round(net, 2)

                # 检查交易状态
                trade_status = row.get('交易状态', '')
                if str(trade_status) == '3':
                    result['status'] = '已收盘'

        sh = result.get('sh_connect', 0)
        sz = result.get('sz_connect', 0)
        result['today_net'] = round(sh + sz, 2)

        return result
    except Exception as e:
        logger.warning(f"[情绪] 获取北向资金失败: {e}")
        return {}


# ============================================================
# 5. 融资余额变化
# ============================================================

def fetch_margin_data() -> dict:
    """
    获取融资余额及变化（上交所）

    返回:
        {
            'margin_balance': 融资余额（亿元）,
            'margin_buy': 融资买入额（亿元）,
            'margin_change_1d': 日环比变化（亿元）,
            'margin_change_5d': 5日变化（亿元）,
            'trend': 趋势描述,
        }
    """
    import akshare as ak

    try:
        end = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')

        df = ak.stock_margin_sse(start_date=start, end_date=end)
        if df.empty or len(df) < 2:
            return {}

        df = df.sort_values('信用交易日期')

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        balance = _safe_float(latest.get('融资余额')) / 1e8
        buy_amt = _safe_float(latest.get('融资买入额')) / 1e8
        prev_balance = _safe_float(prev.get('融资余额')) / 1e8

        change_1d = balance - prev_balance

        # 5日变化
        if len(df) >= 6:
            balance_5ago = _safe_float(df.iloc[-6].get('融资余额')) / 1e8
            change_5d = balance - balance_5ago
        else:
            change_5d = change_1d

        if change_5d > 50:
            trend = '融资大幅流入'
        elif change_5d > 0:
            trend = '融资小幅流入'
        elif change_5d > -50:
            trend = '融资小幅流出'
        else:
            trend = '融资大幅流出'

        return {
            'margin_balance': round(balance, 1),
            'margin_buy': round(buy_amt, 1),
            'margin_change_1d': round(change_1d, 1),
            'margin_change_5d': round(change_5d, 1),
            'trend': trend,
        }
    except Exception as e:
        logger.warning(f"[情绪] 获取融资数据失败: {e}")
        return {}


# ============================================================
# 综合情绪评分
# ============================================================

def compute_sentiment_score(activity: dict, volume: dict,
                            fund_flow: dict, northbound: dict,
                            margin: dict) -> dict:
    """
    综合5个维度计算 0~100 的市场情绪分数

    维度权重:
        涨跌活跃度: 25%
        成交额:     20%
        主力资金:   25%
        北向资金:   15%
        融资余额:   15%

    返回:
        {
            'score': 综合情绪分数 (0~100),
            'level': '极度恐慌'/'偏悲观'/'中性'/'偏乐观'/'极度贪婪',
            'advice': 操作建议,
            'sub_scores': 各维度分数,
        }
    """
    sub_scores = {}

    # ---- 维度1: 涨跌活跃度 (25%) ----
    if activity:
        ratio = activity.get('up_down_ratio', 1.0)
        lu = activity.get('limit_up', 0)
        ld = activity.get('limit_down', 0)

        # 涨跌比: 0.3(极度悲观) ~ 3.0(极度乐观)，1.0为中性
        if ratio >= 2.5:
            s1 = 90
        elif ratio >= 1.5:
            s1 = 70 + (ratio - 1.5) * 20
        elif ratio >= 1.0:
            s1 = 50 + (ratio - 1.0) * 40
        elif ratio >= 0.5:
            s1 = 20 + (ratio - 0.5) * 60
        else:
            s1 = max(5, ratio * 40)

        # 涨停跌停修正
        if lu >= 50 and ld <= 5:
            s1 = min(s1 + 10, 100)
        elif ld >= 30 and lu <= 10:
            s1 = max(s1 - 15, 0)

        sub_scores['activity'] = round(min(max(s1, 0), 100), 1)
    else:
        sub_scores['activity'] = 50

    # ---- 维度2: 成交额 (20%) ----
    if volume:
        vr5 = volume.get('vol_ratio_5d', 1.0)
        vr20 = volume.get('vol_ratio_20d', 1.0)

        # 量比 > 1.3 放量乐观，< 0.7 缩量悲观
        base = 50
        base += (vr5 - 1.0) * 50  # 量比偏离1.0的部分
        base += (vr20 - 1.0) * 25  # 20日量比辅助

        sub_scores['volume'] = round(min(max(base, 5), 95), 1)
    else:
        sub_scores['volume'] = 50

    # ---- 维度3: 主力资金 (25%) ----
    if fund_flow:
        main_pct = fund_flow.get('main_net_pct', 0)

        # 主力净占比: -5%(大幅流出) ~ +5%(大幅流入)
        # 映射到 0~100
        s3 = 50 + main_pct * 10  # 1% 对应 10分

        # 3日趋势修正
        total_3d = fund_flow.get('main_3d_total', 0)
        if total_3d > 100:
            s3 += 10
        elif total_3d < -100:
            s3 -= 10

        sub_scores['fund_flow'] = round(min(max(s3, 0), 100), 1)
    else:
        sub_scores['fund_flow'] = 50

    # ---- 维度4: 北向资金 (15%) ----
    if northbound:
        net = northbound.get('today_net', 0)

        # 净买入: -100亿(恐慌) ~ +100亿(乐观)
        s4 = 50 + net * 0.3  # 1亿 对应 0.3分
        s4 = min(max(s4, 5), 95)

        sub_scores['northbound'] = round(s4, 1)
    else:
        sub_scores['northbound'] = 50

    # ---- 维度5: 融资余额 (15%) ----
    if margin:
        change_5d = margin.get('margin_change_5d', 0)

        # 5日融资变化: -200亿(悲观) ~ +200亿(乐观)
        s5 = 50 + change_5d * 0.15
        s5 = min(max(s5, 10), 90)

        sub_scores['margin'] = round(s5, 1)
    else:
        sub_scores['margin'] = 50

    # ---- 加权综合 ----
    weights = {
        'activity': 0.25,
        'volume': 0.20,
        'fund_flow': 0.25,
        'northbound': 0.15,
        'margin': 0.15,
    }

    total_score = sum(sub_scores[k] * weights[k] for k in weights)
    total_score = round(min(max(total_score, 0), 100), 1)

    # ---- 情绪等级 & 操作建议 ----
    if total_score <= 20:
        level = '极度恐慌'
        advice = '市场极度悲观，可能是左侧抄底机会，但需严控仓位'
    elif total_score <= 35:
        level = '偏悲观'
        advice = '市场情绪低迷，降低买入推荐数量，已持仓注意止损'
    elif total_score <= 50:
        level = '偏谨慎'
        advice = '市场情绪偏弱，精选个股，控制仓位在50%以内'
    elif total_score <= 65:
        level = '中性偏暖'
        advice = '市场情绪正常偏暖，可正常操作'
    elif total_score <= 80:
        level = '偏乐观'
        advice = '市场情绪良好，适合顺势做多，但注意追高风险'
    else:
        level = '极度贪婪'
        advice = '市场过热，注意获利了结，新仓位谨慎'

    return {
        'score': total_score,
        'level': level,
        'advice': advice,
        'sub_scores': sub_scores,
    }


# ============================================================
# 主入口
# ============================================================

def get_market_sentiment(verbose: bool = True) -> dict:
    """
    一键获取完整市场情绪数据

    返回:
        {
            'sentiment_score': 综合情绪分 (0~100),
            'sentiment_level': 情绪等级文字,
            'sentiment_advice': 操作建议,
            'sub_scores': 各维度分数,
            'activity': 涨跌活跃度原始数据,
            'volume': 成交额数据,
            'fund_flow': 主力资金数据,
            'northbound': 北向资金数据,
            'margin': 融资数据,
            'fetch_time': 获取时间,
        }
    """
    t0 = time.time()

    if verbose:
        print("[情绪] 正在获取大盘情绪数据...")

    # 1. 涨跌活跃度
    if verbose:
        print("[情绪]   1/5 涨跌活跃度...")
    activity = fetch_market_activity()

    # 2. 成交额
    if verbose:
        print("[情绪]   2/5 两市成交额...")
    volume = fetch_market_volume()

    # 3. 主力资金
    if verbose:
        print("[情绪]   3/5 主力资金流向...")
    fund_flow = fetch_main_fund_flow()

    # 4. 北向资金
    if verbose:
        print("[情绪]   4/5 北向资金...")
    northbound = fetch_northbound_flow()

    # 5. 融资余额
    if verbose:
        print("[情绪]   5/5 融资余额...")
    margin = fetch_margin_data()

    # 综合评分
    sentiment = compute_sentiment_score(activity, volume, fund_flow, northbound, margin)

    elapsed = time.time() - t0

    result = {
        'sentiment_score': sentiment['score'],
        'sentiment_level': sentiment['level'],
        'sentiment_advice': sentiment['advice'],
        'sub_scores': sentiment['sub_scores'],
        'activity': activity,
        'volume': volume,
        'fund_flow': fund_flow,
        'northbound': northbound,
        'margin': margin,
        'fetch_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': round(elapsed, 1),
    }

    # 保存到文件
    save_path = os.path.join(DATA_DIR, 'market_sentiment.json')
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    if verbose:
        print(f"[情绪] 完成! 情绪分: {sentiment['score']} ({sentiment['level']}) 耗时{elapsed:.1f}s")
        print(f"[情绪]   涨跌:{sentiment['sub_scores'].get('activity', '-')} "
              f"成交:{sentiment['sub_scores'].get('volume', '-')} "
              f"主力:{sentiment['sub_scores'].get('fund_flow', '-')} "
              f"北向:{sentiment['sub_scores'].get('northbound', '-')} "
              f"融资:{sentiment['sub_scores'].get('margin', '-')}")

    return result


def format_sentiment_text(data: dict) -> str:
    """将情绪数据格式化为邮件/文本段落"""
    lines = []
    lines.append("=" * 45)
    lines.append("  大盘情绪温度计")
    lines.append("=" * 45)

    score = data.get('sentiment_score', 50)
    level = data.get('sentiment_level', '未知')
    advice = data.get('sentiment_advice', '')

    # 情绪温度条
    bar_len = 20
    filled = int(score / 100 * bar_len)
    bar = '#' * filled + '-' * (bar_len - filled)
    lines.append(f"  情绪评分: [{bar}] {score}/100 ({level})")
    lines.append(f"  操作建议: {advice}")
    lines.append("")

    # 各维度详情
    sub = data.get('sub_scores', {})
    lines.append("  维度分析:")

    # 涨跌活跃度
    act = data.get('activity', {})
    if act:
        lines.append(f"    涨跌({sub.get('activity', '-')}分): "
                     f"涨{act.get('up_count', 0)}家 跌{act.get('down_count', 0)}家 "
                     f"涨跌比{act.get('up_down_ratio', '-')} "
                     f"涨停{act.get('limit_up', 0)} 跌停{act.get('limit_down', 0)}")

    # 成交额
    vol = data.get('volume', {})
    if vol:
        lines.append(f"    成交({sub.get('volume', '-')}分): "
                     f"今日{vol.get('today_amount', 0)}亿 "
                     f"5日均{vol.get('avg_5d', 0)}亿 "
                     f"量比{vol.get('vol_ratio_5d', '-')} {vol.get('trend', '')}")

    # 主力资金
    ff = data.get('fund_flow', {})
    if ff:
        sign = '+' if ff.get('main_net_flow', 0) >= 0 else ''
        lines.append(f"    主力({sub.get('fund_flow', '-')}分): "
                     f"净流{sign}{ff.get('main_net_flow', 0)}亿 "
                     f"占比{ff.get('main_net_pct', 0)}% "
                     f"{ff.get('trend_3d', '')}")

    # 北向资金
    nb = data.get('northbound', {})
    if nb:
        sign = '+' if nb.get('today_net', 0) >= 0 else ''
        lines.append(f"    北向({sub.get('northbound', '-')}分): "
                     f"净买{sign}{nb.get('today_net', 0)}亿 "
                     f"(沪{nb.get('sh_connect', 0)} 深{nb.get('sz_connect', 0)})")

    # 融资
    mg = data.get('margin', {})
    if mg:
        lines.append(f"    融资({sub.get('margin', '-')}分): "
                     f"余额{mg.get('margin_balance', 0)}亿 "
                     f"5日变化{mg.get('margin_change_5d', 0):+.1f}亿 "
                     f"{mg.get('trend', '')}")

    lines.append("")
    return "\n".join(lines)


# ============================================================
# 直接运行测试
# ============================================================

if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    result = get_market_sentiment(verbose=True)
    print()
    print(format_sentiment_text(result))
