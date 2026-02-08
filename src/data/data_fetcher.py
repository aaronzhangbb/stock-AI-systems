"""
数据获取模块
多数据源策略：腾讯 → 新浪 → 东方财富(AkShare)
自动使用本地 SQLite 缓存，避免重复请求 API

数据存储位置: data/stock_cache.db
"""

import akshare as ak
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import os
import sys
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from src.data.data_cache import DataCache

# 全局缓存实例
_cache = None

# ===== 全局速率控制 =====
_rate_lock = threading.Lock()
_last_request_time = 0
_MIN_REQUEST_INTERVAL = 0.12  # 每次请求最少间隔（秒）

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://finance.qq.com/",
}


def _get_cache() -> DataCache:
    """获取全局缓存实例（懒加载）"""
    global _cache
    if _cache is None:
        _cache = DataCache()
    return _cache


def _clean_stock_code(code: str) -> str:
    """
    清理股票代码，提取纯数字部分
    输入: '600519' 或 'sh600519' 或 'sh.600519'
    输出: '600519'
    """
    code = code.strip()
    for prefix in ['sh.', 'sz.', 'sh', 'sz']:
        if code.lower().startswith(prefix):
            code = code[len(prefix):]
            break
    return code


def _get_market_prefix(code: str) -> str:
    """根据股票代码判断市场前缀 (sh / sz)"""
    if code.startswith(('6', '9')):
        return 'sh'
    else:
        return 'sz'


def _throttle():
    """全局请求限流"""
    global _last_request_time
    with _rate_lock:
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        _last_request_time = time.time()


# ===================================================================
#  数据源 1: 腾讯日K（稳定、快速、无需认证）
# ===================================================================
def _fetch_tencent_single(symbol: str, sd: str, ed: str) -> list:
    """
    腾讯单次请求（最多640条）
    参数 sd/ed 格式: 'YYYY-MM-DD'
    返回: kline 原始列表
    """
    _throttle()
    url = (
        f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        f"?param={symbol},day,{sd},{ed},640,qfq"
    )
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    klines = data.get("data", {}).get(symbol, {}).get("qfqday", [])
    if not klines:
        klines = data.get("data", {}).get(symbol, {}).get("day", [])
    return klines or []


def _fetch_from_tencent(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从腾讯行情接口获取前复权日K数据（支持分段拼接超过640条）

    参数:
        stock_code: 纯数字，如 '600519'
        start_date: 'YYYYMMDD'
        end_date: 'YYYYMMDD'
    返回:
        标准 DataFrame 或空 DataFrame
    """
    try:
        prefix = _get_market_prefix(stock_code)
        symbol = f"{prefix}{stock_code}"

        sd_dt = datetime.strptime(start_date, '%Y%m%d')
        ed_dt = datetime.strptime(end_date, '%Y%m%d')
        total_days = (ed_dt - sd_dt).days

        all_klines = []

        if total_days > 900:
            # 超过~640个交易日，分两段请求
            mid_dt = sd_dt + timedelta(days=total_days // 2)
            sd1 = sd_dt.strftime('%Y-%m-%d')
            ed1 = mid_dt.strftime('%Y-%m-%d')
            sd2 = (mid_dt + timedelta(days=1)).strftime('%Y-%m-%d')
            ed2 = ed_dt.strftime('%Y-%m-%d')

            klines1 = _fetch_tencent_single(symbol, sd1, ed1)
            all_klines.extend(klines1)

            klines2 = _fetch_tencent_single(symbol, sd2, ed2)
            all_klines.extend(klines2)
        else:
            sd = sd_dt.strftime('%Y-%m-%d')
            ed = ed_dt.strftime('%Y-%m-%d')
            all_klines = _fetch_tencent_single(symbol, sd, ed)

        if not all_klines:
            return pd.DataFrame()

        # 格式: ['2026-01-02', '1500.000', '1510.000', '1520.000', '1490.000', '80000.000']
        #        日期,        开盘,       收盘,       最高,       最低,       成交量
        rows = []
        seen_dates = set()
        for k in all_klines:
            if len(k) < 6:
                continue
            d = k[0]
            if d in seen_dates:
                continue
            seen_dates.add(d)
            rows.append({
                'date': d,
                'open': float(k[1]),
                'close': float(k[2]),
                'high': float(k[3]),
                'low': float(k[4]),
                'volume': float(k[5]),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df['amount'] = 0.0
        df['pctChg'] = df['close'].pct_change() * 100
        df = df.sort_values('date').reset_index(drop=True)
        return df

    except Exception as e:
        # 静默失败，让调用方尝试下一个数据源
        return pd.DataFrame()


# ===================================================================
#  数据源 2: 新浪日K（备选）
# ===================================================================
def _fetch_from_sina(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从新浪财经接口获取日K数据（不支持前复权，需后续处理）

    新浪接口一次最多返回 datalen 条记录
    """
    try:
        _throttle()
        prefix = _get_market_prefix(stock_code)
        symbol = f"{prefix}{stock_code}"

        # 计算需要多少交易日
        sd = datetime.strptime(start_date, '%Y%m%d')
        ed = datetime.strptime(end_date, '%Y%m%d')
        trade_days = max((ed - sd).days, 30)
        datalen = min(trade_days + 50, 1023)  # 额外 buffer

        url = (
            f"https://money.finance.sina.com.cn/quotes_service/api/"
            f"json_v2.php/CN_MarketData.getKLineData"
            f"?symbol={symbol}&scale=240&ma=no&datalen={datalen}"
        )
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return pd.DataFrame()

        rows = []
        for item in data:
            d = item.get('day', '')
            if not d:
                continue
            rows.append({
                'date': d,
                'open': float(item.get('open', 0)),
                'high': float(item.get('high', 0)),
                'low': float(item.get('low', 0)),
                'close': float(item.get('close', 0)),
                'volume': float(item.get('volume', 0)),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df['amount'] = 0.0
        df['pctChg'] = df['close'].pct_change() * 100

        # 按日期范围过滤
        sd_dt = pd.to_datetime(start_date)
        ed_dt = pd.to_datetime(end_date)
        df = df[(df['date'] >= sd_dt) & (df['date'] <= ed_dt)]
        df = df.sort_values('date').reset_index(drop=True)
        return df

    except Exception as e:
        return pd.DataFrame()


# ===================================================================
#  数据源 3: 东方财富 AkShare（最后备选）
# ===================================================================
def _fetch_from_akshare(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从 AkShare (东方财富 push2) 获取前复权日K数据
    注意: 此接口在部分网络环境下可能被封
    """
    try:
        _throttle()
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            '日期': 'date', '开盘': 'open', '收盘': 'close',
            '最高': 'high', '最低': 'low', '成交量': 'volume',
            '成交额': 'amount', '振幅': 'amplitude',
            '涨跌幅': 'pctChg', '涨跌额': 'change', '换手率': 'turnover',
        })

        df['date'] = pd.to_datetime(df['date'])
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.sort_values('date').reset_index(drop=True)
        return df

    except Exception as e:
        return pd.DataFrame()


# ===================================================================
#  统一入口: 自动降级
# ===================================================================
def _fetch_from_api(stock_code: str, start_date: str, end_date: str,
                    retries: int = 2) -> pd.DataFrame:
    """
    从多个数据源依次尝试获取数据（腾讯 → 新浪 → 东方财富）

    如果一个源失败，自动切换到下一个源。
    """
    sources = [
        ("腾讯", _fetch_from_tencent),
        ("新浪", _fetch_from_sina),
        ("东方财富", _fetch_from_akshare),
    ]

    for source_name, fetch_func in sources:
        for attempt in range(retries):
            df = fetch_func(stock_code, start_date, end_date)
            if not df.empty and len(df) >= 1:
                return df
            # 短暂等待后重试同一个源
            if attempt < retries - 1:
                time.sleep(0.5)
        # 当前源多次失败，切换下一个源（不打印，减少日志噪音）

    # 所有数据源都失败了
    print(f"[数据] 所有数据源均无法获取 {stock_code} 的数据")
    return pd.DataFrame()


# ===================================================================
#  公开 API
# ===================================================================
def get_history_data(stock_code: str, days: int = None, start_date: str = None,
                     end_date: str = None, use_cache: bool = True) -> pd.DataFrame:
    """
    获取股票历史日线数据（优先从本地缓存读取）

    逻辑:
        1. 先检查本地缓存是否有数据
        2. 如果缓存够新（今天已更新），直接返回缓存
        3. 如果缓存不够新，只从 API 获取缺失的部分，然后合并

    参数:
        stock_code: 股票代码，如 '600519'
        days: 获取最近多少天的数据
        start_date: 开始日期，格式 'YYYYMMDD' 或 'YYYY-MM-DD'
        end_date: 结束日期，默认今天
        use_cache: 是否使用缓存（默认 True）

    返回:
        DataFrame: 包含日期、开高低收、成交量等字段
    """
    code = _clean_stock_code(stock_code)
    cache = _get_cache()

    # 计算日期范围
    if end_date is None:
        end_date_str = datetime.now().strftime('%Y%m%d')
        end_date_dash = datetime.now().strftime('%Y-%m-%d')
    else:
        end_date_str = end_date.replace('-', '')
        end_date_dash = datetime.strptime(end_date_str, '%Y%m%d').strftime('%Y-%m-%d')

    if start_date is None:
        if days is None:
            days = config.HISTORY_DAYS
        start_dt = datetime.now() - timedelta(days=days)
        start_date_str = start_dt.strftime('%Y%m%d')
        start_date_dash = start_dt.strftime('%Y-%m-%d')
    else:
        start_date_str = start_date.replace('-', '')
        start_date_dash = datetime.strptime(start_date_str, '%Y%m%d').strftime('%Y-%m-%d')

    # ===== 缓存策略 =====
    if use_cache:
        cache_info = cache.get_cache_info(code)

        if cache_info and cache.is_cache_fresh(code):
            cached_df = cache.load_kline(code, start_date_dash, end_date_dash)
            if not cached_df.empty:
                return cached_df

        if cache_info:
            cached_last_date = cache_info['last_date']
            next_day = (datetime.strptime(cached_last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y%m%d')

            if next_day <= end_date_str:
                new_df = _fetch_from_api(code, next_day, end_date_str)
                if not new_df.empty:
                    cache.save_kline(code, new_df)

            cached_df = cache.load_kline(code, start_date_dash, end_date_dash)
            if not cached_df.empty:
                return cached_df

    # ===== 无缓存或缓存失效，全量从 API 获取 =====
    df = _fetch_from_api(code, start_date_str, end_date_str)

    if not df.empty and use_cache:
        cache.save_kline(code, df)

    return df


def get_stock_name(stock_code: str) -> str:
    """
    获取股票名称（优先从缓存读取）
    """
    code = _clean_stock_code(stock_code)
    cache = _get_cache()

    if code in config.WATCHLIST:
        name = config.WATCHLIST[code]
        cache.save_stock_name(code, name)
        return name

    cached_name = cache.load_stock_name(code)
    if cached_name:
        return cached_name

    try:
        df = ak.stock_zh_a_spot_em()
        row = df[df['代码'] == code]
        if not row.empty:
            name = row.iloc[0]['名称']
            cache.save_stock_name(code, name)
            return name
    except Exception:
        pass

    return code


def get_realtime_quotes(stock_codes: list = None) -> pd.DataFrame:
    """获取实时行情数据"""
    try:
        df = ak.stock_zh_a_spot_em()
        df = df.rename(columns={
            '代码': 'code', '名称': 'name', '最新价': 'close',
            '涨跌幅': 'pctChg', '涨跌额': 'change', '成交量': 'volume',
            '成交额': 'amount', '今开': 'open', '最高': 'high',
            '最低': 'low', '昨收': 'pre_close', '换手率': 'turnover',
        })

        if stock_codes:
            codes = [_clean_stock_code(c) for c in stock_codes]
            df = df[df['code'].isin(codes)]

        return df

    except Exception as e:
        print(f"获取实时行情失败: {e}")
        return pd.DataFrame()


def get_realtime_price(stock_code: str) -> dict:
    """获取单只股票最新价格信息"""
    code = _clean_stock_code(stock_code)
    try:
        df = get_realtime_quotes([code])
        if df.empty:
            return {}
        row = df.iloc[0]
        return {
            'code': code,
            'name': str(row.get('name', code)),
            'close': float(row['close']) if pd.notna(row['close']) else 0.0,
            'open': float(row['open']) if pd.notna(row['open']) else 0.0,
            'high': float(row['high']) if pd.notna(row['high']) else 0.0,
            'low': float(row['low']) if pd.notna(row['low']) else 0.0,
            'volume': float(row['volume']) if pd.notna(row['volume']) else 0.0,
            'pctChg': float(row['pctChg']) if pd.notna(row['pctChg']) else 0.0,
        }
    except Exception as e:
        print(f"获取 {stock_code} 实时价格失败: {e}")
        return {}


def get_cache_stats() -> pd.DataFrame:
    """获取缓存统计信息"""
    return _get_cache().get_all_cached_stocks()


def clear_stock_cache(stock_code: str = None):
    """清除缓存"""
    _get_cache().clear_cache(stock_code)


if __name__ == '__main__':
    print("=" * 60)
    print("  数据获取模块测试（多数据源 + 本地缓存）")
    print("=" * 60)

    # 测试获取
    print("\n--- 获取 600519 ---")
    df = get_history_data('600519', days=30)
    if not df.empty:
        print(f"获取到 {len(df)} 条数据")
        print(df.tail(3).to_string(index=False))
    else:
        print("获取失败！")

    # 第二次应从缓存
    print("\n--- 再次获取（应从缓存）---")
    df2 = get_history_data('600519', days=30)
    if not df2.empty:
        print(f"获取到 {len(df2)} 条数据")

    print(f"\n缓存数据库位置: {_get_cache().db_path}")
