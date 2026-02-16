# -*- coding: utf-8 -*-
"""
AI策略引擎 V2.0
基于 XGBoost GPU 的数据驱动策略系统

核心理念: 不再手工设定阈值，让AI自己从数据中发现规律
"""
import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ============================================================
# 特征工程 V2.0 — 100+ 高阶特征
# ============================================================

def compute_v2_features(df: pd.DataFrame, extra_features: dict = None) -> pd.DataFrame:
    """
    计算V2高阶特征集 (~120+ 个特征)
    
    输入: 含有 date/open/high/low/close/volume/pctChg 的 DataFrame
          extra_features: 可选的外部特征字典, 键值对会被广播为列
                          (基本面/板块/情绪等非K线来源特征)
    输出: 增加了特征列的 DataFrame
    """
    data = df.copy()
    data = data.sort_values('date').reset_index(drop=True)
    
    close = data['close'].astype(float)
    high = data['high'].astype(float)
    low = data['low'].astype(float)
    open_ = data['open'].astype(float)
    volume = data['volume'].astype(float)
    
    # 确保 pctChg 存在
    if 'pctChg' not in data.columns or data['pctChg'].isna().all():
        data['pctChg'] = close.pct_change() * 100
    pctChg = data['pctChg'].astype(float)
    
    # ============================================================
    # 1. 多周期收益率 (动量)
    # ============================================================
    for p in [1, 2, 3, 5, 10, 20, 60]:
        data[f'ret_{p}d'] = close.pct_change(p)
    
    # ============================================================
    # 2. 动量加速度 (收益率的变化率 — 二阶导数)
    # ============================================================
    ret_1 = close.pct_change()
    data['momentum_accel_5'] = ret_1.diff(5)    # 5日动量加速度
    data['momentum_accel_10'] = ret_1.diff(10)   # 10日动量加速度
    data['momentum_accel_20'] = ret_1.diff(20)   # 20日动量加速度
    
    # ============================================================
    # 3. 均线系统 (多周期)
    # ============================================================
    for p in [5, 10, 20, 30, 60]:
        ma = close.rolling(p).mean()
        data[f'ma{p}_diff'] = (close - ma) / ma                # 偏离度
        data[f'ma{p}_slope'] = ma.pct_change(5)                 # 5日斜率
    
    # 均线交叉信号
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    data['ma5_10_cross'] = (ma5 - ma10) / close     # 短期vs中期
    data['ma10_20_cross'] = (ma10 - ma20) / close    # 中期vs长期
    data['ma20_60_cross'] = (ma20 - ma60) / close    # 长期趋势
    
    # 均线多头排列指标 (ma5>ma10>ma20>ma60 → 值越大越多头)
    data['ma_alignment'] = (
        (ma5 > ma10).astype(int) + 
        (ma10 > ma20).astype(int) + 
        (ma20 > ma60).astype(int)
    )
    
    # ============================================================
    # 4. RSI 多周期
    # ============================================================
    for period in [6, 14, 28]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss_ = (-delta).where(delta < 0, 0.0)
        ag = gain.rolling(period, min_periods=period).mean()
        al = loss_.rolling(period, min_periods=period).mean()
        rs = ag / (al + 1e-10)
        data[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # RSI背离 (价格创新低但RSI没有)
    data['rsi_diverge'] = data['rsi_14'] - data['rsi_14'].rolling(20).min()
    
    # ============================================================
    # 5. 布林带
    # ============================================================
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    data['bb_pos'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    data['bb_width'] = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    data['bb_squeeze'] = bb_std / (bb_mid + 1e-10)  # 布林带收窄度
    
    # ============================================================
    # 6. 波动率相关
    # ============================================================
    ret = close.pct_change()
    for p in [5, 10, 20, 60]:
        data[f'volatility_{p}d'] = ret.rolling(p).std() * np.sqrt(252)
    
    # 波动率变化率 (暴风雨前的宁静)
    vol_20 = ret.rolling(20).std()
    vol_60 = ret.rolling(60).std()
    data['vol_change_ratio'] = vol_20 / (vol_60 + 1e-10)
    
    # ATR (Average True Range)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    data['atr_14'] = tr.rolling(14).mean()
    data['atr_ratio'] = tr / (data['atr_14'] + 1e-10)  # 当日TR vs 平均
    
    # ============================================================
    # 7. 成交量特征
    # ============================================================
    for p in [5, 10, 20]:
        data[f'vol_ratio_{p}d'] = volume / (volume.rolling(p).mean() + 1)
    
    # 量价关系
    data['vol_price_corr_10'] = ret.rolling(10).corr(volume.pct_change())
    data['vol_price_corr_20'] = ret.rolling(20).corr(volume.pct_change())
    
    # OBV (On-Balance Volume) 变化率
    obv = (np.sign(close.diff()) * volume).cumsum()
    data['obv_slope_10'] = obv.pct_change(10)
    data['obv_slope_20'] = obv.pct_change(20)
    
    # 量能萎缩度
    data['vol_shrink'] = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1)
    
    # ============================================================
    # 8. K线形态 (微观结构)
    # ============================================================
    body = close - open_
    upper_shadow = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower_shadow = pd.concat([close, open_], axis=1).min(axis=1) - low
    range_ = high - low + 1e-10
    
    data['body_ratio'] = body / range_            # 实体/振幅比
    data['upper_shadow_ratio'] = upper_shadow / range_
    data['lower_shadow_ratio'] = lower_shadow / range_
    data['body_abs_ratio'] = body.abs() / range_  # 实体绝对比率
    
    # 十字星 (实体很小)
    data['is_doji'] = (body.abs() / range_ < 0.1).astype(int)
    
    # 长下影线 (可能底部信号)
    data['long_lower_shadow'] = (lower_shadow / range_ > 0.6).astype(int)
    
    # ============================================================
    # 9. VWAP 相关
    # ============================================================
    typical_price = (high + low + close) / 3
    data['vwap_diff'] = (close - typical_price) / (typical_price + 1e-10)
    
    # 累计VWAP偏离
    cum_tp_vol = (typical_price * volume).rolling(20).sum()
    cum_vol = volume.rolling(20).sum()
    vwap_20 = cum_tp_vol / (cum_vol + 1)
    data['vwap20_diff'] = (close - vwap_20) / (vwap_20 + 1e-10)
    
    # ============================================================
    # 10. MFI (Money Flow Index)
    # ============================================================
    mf = typical_price * volume
    mf_pos = mf.where(typical_price > typical_price.shift(1), 0)
    mf_neg = mf.where(typical_price <= typical_price.shift(1), 0)
    mfr = mf_pos.rolling(14).sum() / (mf_neg.rolling(14).sum() + 1e-10)
    data['mfi_14'] = 100 - (100 / (1 + mfr))
    
    # ============================================================
    # 11. MACD
    # ============================================================
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    data['macd'] = macd / (close + 1e-10)           # 归一化
    data['macd_signal'] = signal / (close + 1e-10)
    data['macd_hist'] = (macd - signal) / (close + 1e-10)
    data['macd_cross'] = np.sign(macd - signal)  # 金/死叉
    
    # ============================================================
    # 12. KDJ
    # ============================================================
    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9 + 1e-10) * 100
    data['kdj_k'] = rsv.ewm(alpha=1/3, adjust=False).mean()
    data['kdj_d'] = data['kdj_k'].ewm(alpha=1/3, adjust=False).mean()
    data['kdj_j'] = 3 * data['kdj_k'] - 2 * data['kdj_d']
    
    # ============================================================
    # 13. 价格位置特征
    # ============================================================
    for p in [5, 10, 20, 60]:
        data[f'dist_high_{p}d'] = close / (high.rolling(p).max() + 1e-10) - 1
        data[f'dist_low_{p}d'] = close / (low.rolling(p).min() + 1e-10) - 1
    
    # 距离52周(250日)高/低点
    data['dist_high_250d'] = close / (high.rolling(min(250, len(close))).max() + 1e-10) - 1
    data['dist_low_250d'] = close / (low.rolling(min(250, len(close))).min() + 1e-10) - 1
    
    # 价格区间位置 (0=最低, 1=最高)
    for p in [20, 60]:
        h = high.rolling(p).max()
        l = low.rolling(p).min()
        data[f'price_pos_{p}d'] = (close - l) / (h - l + 1e-10)
    
    # ============================================================
    # 14. 连续涨跌统计
    # ============================================================
    up = (close > close.shift(1)).astype(int)
    down = (close < close.shift(1)).astype(int)
    data['consec_up'] = up.groupby((up != up.shift()).cumsum()).cumsum()
    data['consec_down'] = down.groupby((down != down.shift()).cumsum()).cumsum()
    
    # 近N日涨跌天数比
    data['up_ratio_10d'] = up.rolling(10).sum() / 10
    data['up_ratio_20d'] = up.rolling(20).sum() / 20
    
    # ============================================================
    # 15. 统计分布特征
    # ============================================================
    data['ret_skew_20'] = ret.rolling(20).skew()   # 偏度
    data['ret_kurt_20'] = ret.rolling(20).kurt()   # 峰度
    data['ret_skew_60'] = ret.rolling(60).skew()
    
    # ============================================================
    # 16. 时间特征
    # ============================================================
    if 'date' in data.columns:
        dt = pd.to_datetime(data['date'])
        data['month'] = dt.dt.month
        data['weekday'] = dt.dt.dayofweek            # 0=周一
        data['month_pos'] = dt.dt.day / 31.0          # 月内位置
        data['quarter'] = dt.dt.quarter
    
    # ============================================================
    # 17. 高级动量指标
    # ============================================================
    # Stochastic RSI
    rsi14 = data['rsi_14']
    rsi_min = rsi14.rolling(14).min()
    rsi_max = rsi14.rolling(14).max()
    data['stoch_rsi'] = (rsi14 - rsi_min) / (rsi_max - rsi_min + 1e-10)
    
    # Williams %R
    data['williams_r'] = (high.rolling(14).max() - close) / (high.rolling(14).max() - low.rolling(14).min() + 1e-10)
    
    # CCI (Commodity Channel Index)
    tp = (high + low + close) / 3
    tp_ma = tp.rolling(20).mean()
    tp_md = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    data['cci_20'] = (tp - tp_ma) / (0.015 * tp_md + 1e-10)
    
    # ============================================================
    # 18. 价格效率 (Fractal Efficiency)
    # ============================================================
    for p in [10, 20]:
        net_move = (close - close.shift(p)).abs()
        gross_move = close.diff().abs().rolling(p).sum()
        data[f'efficiency_{p}d'] = net_move / (gross_move + 1e-10)
    
    # ============================================================
    # 19. 波动率深度特征 (V3新增 — 阶段2核心)
    # ============================================================
    # === 19.1 波动率分位数 (当前波动率在历史中的位置) ===
    vol_20_ann = ret.rolling(20).std() * np.sqrt(252)
    for lookback in [60, 120, 250]:
        lb = min(lookback, len(close) - 1)
        data[f'vol_percentile_{lb}d'] = vol_20_ann.rolling(lb, min_periods=20).rank(pct=True)
    
    # === 19.2 波动率均值回归信号 ===
    # 波动率偏离长期均值的程度 (Z-Score)
    vol_mean_120 = vol_20_ann.rolling(120, min_periods=30).mean()
    vol_std_120 = vol_20_ann.rolling(120, min_periods=30).std()
    data['vol_zscore'] = (vol_20_ann - vol_mean_120) / (vol_std_120 + 1e-10)
    
    # === 19.3 波动率收缩/扩张信号 (关键！高波→低波常伴随方向性突破) ===
    vol_5 = ret.rolling(5).std() * np.sqrt(252)
    vol_10 = ret.rolling(10).std() * np.sqrt(252)
    data['vol_contraction_5_20'] = vol_5 / (vol_20_ann + 1e-10)    # <1 = 短期波动收缩
    data['vol_contraction_10_60'] = vol_10 / (data.get('volatility_60d', vol_20_ann) + 1e-10)
    
    # 波动率收缩速度 (连续几天在收缩)
    vol_shrinking = (vol_20_ann < vol_20_ann.shift(1)).astype(int)
    data['vol_shrink_streak'] = vol_shrinking.groupby(
        (vol_shrinking != vol_shrinking.shift()).cumsum()
    ).cumsum()
    
    # === 19.4 波动率斜率 (波动率变化的方向和速度) ===
    data['vol_slope_5d'] = vol_20_ann.pct_change(5)
    data['vol_slope_10d'] = vol_20_ann.pct_change(10)
    data['vol_accel'] = data['vol_slope_5d'] - data['vol_slope_5d'].shift(5)  # 波动率加速度
    
    # === 19.5 实现波动率 vs 隐含波动率代理 ===
    # 用 ATR 相对值作为"隐含波动率"代理
    atr_ann = (data['atr_14'] / close) * np.sqrt(252)
    data['realized_vs_implied'] = vol_20_ann / (atr_ann + 1e-10)
    
    # === 19.6 上行/下行波动率分离 (非常重要!) ===
    up_ret = ret.where(ret > 0, 0)
    down_ret = ret.where(ret < 0, 0)
    up_vol = up_ret.rolling(20).std() * np.sqrt(252)
    down_vol = down_ret.rolling(20).std() * np.sqrt(252)
    data['vol_upside_20d'] = up_vol
    data['vol_downside_20d'] = down_vol
    data['vol_asymmetry'] = up_vol / (down_vol + 1e-10)  # >1 = 上行波动更大(利好)
    
    # === 19.7 波动率锥 (Volatility Cone) — 不同窗口的波动率关系 ===
    data['vol_term_structure'] = vol_5 / (vol_20_ann + 1e-10)  # 短/中期
    vol_60_ann = ret.rolling(60).std() * np.sqrt(252)
    data['vol_term_structure_long'] = vol_20_ann / (vol_60_ann + 1e-10)  # 中/长期
    
    # === 19.8 极端波动日统计 ===
    abs_ret = ret.abs()
    ret_p90 = abs_ret.rolling(60, min_periods=20).quantile(0.9)
    data['extreme_vol_days_10'] = (abs_ret > ret_p90).rolling(10).sum()  # 近10日极端波动天数
    data['extreme_vol_days_20'] = (abs_ret > ret_p90).rolling(20).sum()
    
    # === 19.9 Parkinson波动率 (用High-Low估计，比收盘价波动率更准) ===
    hl_ratio = np.log(high / (low + 1e-10))
    data['parkinson_vol_20'] = np.sqrt(
        hl_ratio.pow(2).rolling(20).mean() / (4 * np.log(2))
    ) * np.sqrt(252)
    
    # === 19.10 Garman-Klass波动率 (综合OHLC) ===
    hl2 = 0.5 * np.log(high / (low + 1e-10)).pow(2)
    co2 = (2 * np.log(2) - 1) * np.log(close / (open_ + 1e-10)).pow(2)
    data['gk_vol_20'] = np.sqrt((hl2 - co2).rolling(20).mean()) * np.sqrt(252)
    
    # ============================================================
    # 20. 精选交互特征 (V3新增 — 高重要性因子交叉)
    # ============================================================
    # 波动率 × 均线偏离 (高波动 + 超跌 = 强反弹信号)
    data['vol_x_ma60dev'] = vol_20_ann * data['ma60_diff'].abs()
    data['vol_x_ma30dev'] = vol_20_ann * data['ma30_diff'].abs()
    
    # 波动率 × 布林位置 (高波动 + 布林下轨 = 机会)
    data['vol_x_bbpos'] = vol_20_ann * (1 - data['bb_pos'].clip(0, 1))
    
    # 波动率 × RSI (高波动 + RSI超卖)
    data['vol_x_rsi_inv'] = vol_20_ann * (1 - data['rsi_14'] / 100)
    
    # 波动率收缩 × 量能萎缩 (双重收缩 → 即将爆发)
    data['vol_shrink_x_vol_dry'] = data['vol_contraction_5_20'] * data['vol_shrink']
    
    # 时间 × 波动率 (某些月份高波动更有效)
    if 'month' in data.columns:
        data['month_x_vol'] = data['month'] * vol_20_ann
    
    # 动量 × 波动率 (高波动环境下的动量)
    data['momentum_x_vol'] = data.get('ret_5d', ret.rolling(5).sum()) * vol_20_ann
    
    # 量价关系 × 波动率
    data['vol_price_corr_x_vol'] = data.get('vol_price_corr_20', 0) * vol_20_ann
    
    # ============================================================
    # 21. 外部特征注入 (基本面/板块/情绪 — 非K线来源)
    # ============================================================
    if extra_features:
        for key, value in extra_features.items():
            data[key] = value  # 标量会自动广播到所有行
    
    return data


# ============================================================
# 外部特征构建器 — 基本面 / 板块 / 情绪
# ============================================================

def build_extra_features(stock_code: str,
                         board_name: str = '',
                         bulk_valuation: pd.DataFrame = None,
                         industry_benchmarks: dict = None,
                         sector_heat: dict = None,
                         sentiment_data: dict = None) -> dict:
    """
    为单只股票构建外部特征字典 (传给 compute_v2_features 的 extra_features 参数)
    
    参数:
        stock_code: 股票代码
        board_name: 行业板块名称
        bulk_valuation: 全市场估值DataFrame (来自 fetch_bulk_valuation)
        industry_benchmarks: 行业PE/PB分位数 (来自 get_industry_benchmarks)
        sector_heat: 板块热度数据 (来自 get_sector_heat_map)
        sentiment_data: 市场情绪数据 (来自 get_market_sentiment)
    
    返回:
        dict: 可直接传给 compute_v2_features 的 extra_features
    """
    # 初始化所有16个外部特征为NaN (确保模型特征名称完整)
    feats = {
        # 基本面 (8个)
        'f_pe': np.nan, 'f_pb': np.nan, 'f_mv_log': np.nan,
        'f_turnover': np.nan, 'f_volume_ratio': np.nan, 'f_is_loss': np.nan,
        'f_pe_rank': np.nan, 'f_pb_rank': np.nan,
        # 板块 (3个)
        'f_sector_heat': np.nan, 'f_sector_pct': np.nan, 'f_sector_flow': np.nan,
        # 市场情绪 (5个)
        'f_sentiment': np.nan, 'f_sent_activity': np.nan,
        'f_sent_volume': np.nan, 'f_sent_fund': np.nan, 'f_sent_north': np.nan,
    }
    
    # ---- 基本面特征 ----
    if bulk_valuation is not None and not bulk_valuation.empty:
        row = bulk_valuation[bulk_valuation['stock_code'] == stock_code]
        if not row.empty:
            r = row.iloc[0]
            pe = r.get('pe')
            pb = r.get('pb')
            total_mv = r.get('total_mv')
            turnover = r.get('turnover')
            vol_ratio = r.get('volume_ratio')
            
            # PE (负值表示亏损, 标记为NaN让模型自行处理)
            feats['f_pe'] = float(pe) if pd.notna(pe) and pe > 0 else np.nan
            # PB
            feats['f_pb'] = float(pb) if pd.notna(pb) and pb > 0 else np.nan
            # log市值 (对数化处理大范围差异)
            feats['f_mv_log'] = float(np.log(total_mv + 1)) if pd.notna(total_mv) and total_mv > 0 else np.nan
            # 换手率
            feats['f_turnover'] = float(turnover) if pd.notna(turnover) else np.nan
            # 量比
            feats['f_volume_ratio'] = float(vol_ratio) if pd.notna(vol_ratio) else np.nan
            # 是否亏损 (二值特征)
            feats['f_is_loss'] = 1.0 if pd.notna(pe) and pe < 0 else 0.0
            
            # 行业相对估值分位数
            if industry_benchmarks and board_name:
                ind = industry_benchmarks.get(board_name, {})
                pe_q25 = ind.get('pe_q25')
                pe_q75 = ind.get('pe_q75')
                pb_q25 = ind.get('pb_q25')
                pb_q75 = ind.get('pb_q75')
                
                # PE分位数: 0=便宜, 0.5=中位, 1=贵
                if pe_q25 is not None and pe_q75 is not None and pd.notna(pe) and pe > 0:
                    if pe_q75 > pe_q25:
                        feats['f_pe_rank'] = float(np.clip((pe - pe_q25) / (pe_q75 - pe_q25), 0, 2))
                    else:
                        feats['f_pe_rank'] = 0.5
                
                # PB分位数
                if pb_q25 is not None and pb_q75 is not None and pd.notna(pb) and pb > 0:
                    if pb_q75 > pb_q25:
                        feats['f_pb_rank'] = float(np.clip((pb - pb_q25) / (pb_q75 - pb_q25), 0, 2))
                    else:
                        feats['f_pb_rank'] = 0.5
    
    # ---- 板块特征 ----
    if sector_heat is not None:
        sector_map = sector_heat.get('sector_map', {})
        if board_name and board_name in sector_map:
            s = sector_map[board_name]
            feats['f_sector_heat'] = float(s.get('heat_score', 50))
            feats['f_sector_pct'] = float(s.get('pct_change', 0))
            feats['f_sector_flow'] = float(s.get('main_net_pct', 0))
    
    # ---- 市场情绪特征 ----
    if sentiment_data is not None:
        feats['f_sentiment'] = float(sentiment_data.get('sentiment_score', 50))
        sub = sentiment_data.get('sub_scores', {})
        feats['f_sent_activity'] = float(sub.get('activity', 50))
        feats['f_sent_volume'] = float(sub.get('volume', 50))
        feats['f_sent_fund'] = float(sub.get('fund_flow', 50))
        feats['f_sent_north'] = float(sub.get('northbound', 50))
    
    return feats


# ============================================================
# 标签构建
# ============================================================

def create_labels(df: pd.DataFrame, future_days: int = 5, 
                  target_return: float = 0.03) -> pd.Series:
    """
    构建预测标签
    label = 1 if 未来 future_days 天内最高价涨幅 > target_return
    label = 0 otherwise
    
    使用"未来最高价"而非"收盘价"，更贴合实际交易
    """
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    
    # 未来N天的最高价
    future_max = high.shift(-1).rolling(future_days).max().shift(-(future_days - 1))
    
    # 涨幅
    future_return = (future_max - close) / close
    
    # 标签
    label = (future_return >= target_return).astype(int)
    
    return label


# ============================================================
# 数据集构建
# ============================================================

def build_dataset(cache, pool, 
                  future_days: int = 5, 
                  target_return: float = 0.03,
                  min_bars: int = 100,
                  progress_callback=None,
                  use_extra_features: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    构建全市场训练数据集
    
    参数:
        use_extra_features: 是否加载基本面/板块等外部特征 (默认True)
    
    返回:
        X: 特征矩阵
        y: 标签
        meta: 元信息 (stock_code, date)
    """
    tradeable = pool.get_tradeable_stocks()
    cached = cache.get_all_cached_stocks()
    tradeable_codes = set(tradeable['stock_code'].values)
    cached_codes = set(cached['stock_code'].values)
    valid_codes = sorted(tradeable_codes & cached_codes)
    
    # 构建 stock_code → board_name 映射
    code_board = {}
    for _, row in tradeable.iterrows():
        code_board[row['stock_code']] = row.get('board_name', '')
    
    # 加载外部数据源 (训练时: 基本面用当前快照, 板块/情绪为NaN)
    bulk_valuation = None
    industry_benchmarks = None
    if use_extra_features:
        try:
            from src.data.fundamental import fetch_bulk_valuation, get_industry_benchmarks
            bulk_valuation = fetch_bulk_valuation(cache_minutes=180)
            industry_benchmarks = get_industry_benchmarks(cache_minutes=180)
            if progress_callback:
                print(f"    [Extra] 已加载基本面数据: {len(bulk_valuation)} 只股票估值")
        except Exception as e:
            print(f"    [Extra] 加载基本面数据失败(跳过): {e}")
    
    all_X = []
    all_y = []
    all_meta = []
    
    total = len(valid_codes)
    for i, code in enumerate(valid_codes):
        if progress_callback and (i + 1) % 200 == 0:
            progress_callback(i + 1, total)
        
        try:
            df = cache.load_kline(code)
            if df is None or len(df) < min_bars:
                continue
            
            # 构建外部特征 (训练时: 基本面有值, 板块/情绪为NaN由XGBoost处理)
            extra = None
            if use_extra_features and bulk_valuation is not None:
                board = code_board.get(code, '')
                extra = build_extra_features(
                    stock_code=code,
                    board_name=board,
                    bulk_valuation=bulk_valuation,
                    industry_benchmarks=industry_benchmarks,
                    sector_heat=None,       # 训练时无历史板块数据
                    sentiment_data=None,    # 训练时无历史情绪数据
                )
            
            # 计算特征 (技术面 + 外部特征)
            data = compute_v2_features(df, extra_features=extra)
            if data.empty:
                continue
            
            # 构建标签
            labels = create_labels(data, future_days, target_return)
            
            # 取有效行 (跳过前60行无法计算的)
            start_idx = 65
            end_idx = len(data) - future_days - 1
            if end_idx <= start_idx:
                continue
            
            valid_data = data.iloc[start_idx:end_idx]
            valid_labels = labels.iloc[start_idx:end_idx]
            
            # 去掉label为NaN的行
            mask = valid_labels.notna()
            valid_data = valid_data[mask]
            valid_labels = valid_labels[mask]
            
            if len(valid_data) == 0:
                continue
            
            # 提取特征列 (排除非特征列)
            exclude_cols = ['stock_code', 'date', 'open', 'high', 'low', 'close', 
                           'volume', 'amount', 'amplitude', 'pctChg', 'change', 'turnover']
            feat_cols = [c for c in valid_data.columns if c not in exclude_cols]
            
            X = valid_data[feat_cols]
            y = valid_labels
            meta = valid_data[['date']].copy()
            meta['stock_code'] = code
            
            all_X.append(X)
            all_y.append(y)
            all_meta.append(meta)
            
        except Exception as e:
            continue
    
    if not all_X:
        return pd.DataFrame(), pd.Series(), pd.DataFrame()
    
    X_full = pd.concat(all_X, ignore_index=True)
    y_full = pd.concat(all_y, ignore_index=True)
    meta_full = pd.concat(all_meta, ignore_index=True)
    
    return X_full, y_full, meta_full


def get_feature_columns(X: pd.DataFrame) -> List[str]:
    """获取所有特征列名"""
    return list(X.columns)


# ============================================================
# 时间切分
# ============================================================

def split_by_time(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
                  train_end: str = '2025-06-30',
                  val_end: str = '2025-12-31') -> dict:
    """
    按时间切分训练集/验证集/测试集
    
    train: 2023-02 ~ 2025-06
    val:   2025-07 ~ 2025-12
    test:  2026-01 ~ 2026-02
    """
    dates = pd.to_datetime(meta['date'])
    
    train_mask = dates <= train_end
    val_mask = (dates > train_end) & (dates <= val_end)
    test_mask = dates > val_end
    
    return {
        'X_train': X[train_mask], 'y_train': y[train_mask],
        'X_val': X[val_mask], 'y_val': y[val_mask],
        'X_test': X[test_mask], 'y_test': y[test_mask],
        'meta_train': meta[train_mask],
        'meta_val': meta[val_mask],
        'meta_test': meta[test_mask],
    }


# ============================================================
# AI评分引擎 — 每日全市场打分
# ============================================================

class AIScorer:
    """
    AI评分引擎 V2
    加载训练好的XGBoost模型，给全市场股票实时打分
    """
    
    EXCLUDE_COLS = ['stock_code', 'date', 'open', 'high', 'low', 'close',
                    'volume', 'amount', 'amplitude', 'pctChg', 'change', 'turnover']
    
    def __init__(self, model_path: str = None):
        import xgboost as xgb
        
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'data', 'xgb_v2_model.json'
            )
        
        self.model_path = os.path.abspath(model_path)
        self.model = None
        self.feature_names = None
        self._load_model()
    
    def _load_model(self):
        """加载训练好的模型"""
        import xgboost as xgb
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        self.model = xgb.Booster()
        self.model.load_model(self.model_path)
        
        # 从模型获取特征名
        self.feature_names = self.model.feature_names
    
    def score_single(self, df: pd.DataFrame, extra_features: dict = None) -> Optional[Dict]:
        """
        对单只股票打分
        
        参数:
            df: 该股票的历史K线DataFrame (需至少100行)
            extra_features: 外部特征字典 (基本面/板块/情绪)
        
        返回:
            dict: {score, probability, features} 或 None
        """
        import xgboost as xgb
        
        if df is None or len(df) < 100:
            return None
        
        try:
            data = compute_v2_features(df, extra_features=extra_features)
            if data.empty or len(data) < 65:
                return None
            
            # 取最后一行
            last_row = data.iloc[-1:]
            
            # 提取特征
            feat_cols = [c for c in last_row.columns if c not in self.EXCLUDE_COLS]
            X = last_row[feat_cols].copy()
            
            # 处理缺失/无穷值
            X = X.replace([np.inf, -np.inf], np.nan)
            
            # 确保特征列与模型匹配
            if self.feature_names is not None:
                missing = set(self.feature_names) - set(X.columns)
                for col in missing:
                    X[col] = np.nan
                X = X[self.feature_names]
            
            dmat = xgb.DMatrix(X, feature_names=list(X.columns))
            prob = float(self.model.predict(dmat)[0])
            
            # 转换为0-100评分
            score = round(prob * 100, 1)
            
            # 收集关键指标
            close = float(data.iloc[-1]['close'])
            features = {}
            for col in ['volatility_20d', 'volatility_10d', 'ma60_diff', 'bb_pos',
                        'ret_5d', 'vol_ratio_5d', 'rsi_14', 'macd_hist']:
                v = data.iloc[-1].get(col)
                if v is not None and not pd.isna(v):
                    features[col] = round(float(v), 4)
            
            return {
                'score': score,
                'probability': round(prob, 4),
                'close': close,
                'features': features,
            }
        except Exception:
            return None
    
    def _compute_trade_advice(self, data: pd.DataFrame, close: float,
                               ai_probability: float = 0.5) -> Dict:
        """
        智能交易建议引擎 V2.0
        
        核心理念: 所有参数都由数据驱动，不使用硬编码阈值
        
        升级要点:
          1. 止损: 纯ATR驱动 + 支撑位感知，不设固定百分比上下限
          2. 止盈: 多目标体系(T1/T2/T3) + 阻力位自适应
          3. 仓位: 改进Kelly公式 × 波动率缩放 × AI置信度
          4. 持有期: ATR标准化目标距离 + 动量加速度估算
          5. 买入价: 成交密集区 + 多级支撑带 + ATR缓冲
        """
        last = data.iloc[-1]
        
        try:
            # ================================================================
            # 基础数据提取
            # ================================================================
            high_20 = float(data['high'].tail(20).max())
            low_20 = float(data['low'].tail(20).min())
            high_60 = float(data['high'].tail(60).max())
            low_60 = float(data['low'].tail(60).min())
            
            ma5 = float(data['close'].tail(5).mean())
            ma10 = float(data['close'].tail(10).mean())
            ma20 = float(data['close'].tail(20).mean())
            ma60_val = float(data['close'].tail(60).mean()) if len(data) >= 60 else ma20
            
            bb_mid = ma20
            bb_std = float(data['close'].tail(20).std())
            bb_lower = bb_mid - 2 * bb_std
            bb_upper = bb_mid + 2 * bb_std
            
            # ATR — 核心波动度量
            atr = float(last.get('atr_14', 0)) if not pd.isna(last.get('atr_14', np.nan)) else close * 0.03
            if atr <= 0:
                atr = close * 0.03
            atr_pct = atr / close  # ATR占股价比例
            
            # 各指标
            vol20d = float(last.get('volatility_20d', 0.3)) if not pd.isna(last.get('volatility_20d', np.nan)) else 0.3
            vol5d = float(last.get('volatility_5d', vol20d)) if not pd.isna(last.get('volatility_5d', np.nan)) else vol20d
            rsi = float(last.get('rsi_14', 50)) if not pd.isna(last.get('rsi_14', np.nan)) else 50
            bb_pos = float(last.get('bb_pos', 0.5)) if not pd.isna(last.get('bb_pos', np.nan)) else 0.5
            macd_hist = float(last.get('macd_hist', 0)) if not pd.isna(last.get('macd_hist', np.nan)) else 0
            ma60_diff = float(last.get('ma60_diff', 0)) if not pd.isna(last.get('ma60_diff', np.nan)) else 0
            
            # 趋势强度评分 (0~1, 越高趋势越强)
            ma_align = float(last.get('ma_alignment', 1)) if not pd.isna(last.get('ma_alignment', np.nan)) else 1
            trend_strength = ma_align / 3.0  # 标准化到 0~1
            
            # 动量加速度
            mom_accel = float(last.get('momentum_accel_5', 0)) if not pd.isna(last.get('momentum_accel_5', np.nan)) else 0
            
            # 波动率收缩度 (短/长期波动率比, <1 说明近期波动在收缩)
            vol_contraction = float(last.get('vol_contraction_5_20', 1.0)) if not pd.isna(last.get('vol_contraction_5_20', np.nan)) else 1.0
            
            # 上行/下行波动率不对称性
            vol_asym = float(last.get('vol_asymmetry', 1.0)) if not pd.isna(last.get('vol_asymmetry', np.nan)) else 1.0
            
            # 价格效率 (趋势型 vs 震荡型)
            efficiency = float(last.get('efficiency_10d', 0.5)) if not pd.isna(last.get('efficiency_10d', np.nan)) else 0.5
            
            # ================================================================
            # 1. 智能止损系统 — 纯ATR驱动 + 支撑位感知
            # ================================================================
            # 基础止损距离 = ATR × 动态倍数
            # 动态倍数由趋势强度和波动率状态决定
            
            # 趋势越强 → 止损可以给更多空间(不被洗出去)
            # 波动率越高 → 止损需要更宽(避免噪音触发)
            # 波动率收缩 → 止损可以收紧(突破前的宁静)
            trend_factor = 1.0 + trend_strength * 0.5   # 1.0 ~ 1.5
            vol_regime_factor = np.clip(vol20d / 0.3, 0.7, 2.0)  # 以30%年化波动率为基准
            contraction_factor = np.clip(vol_contraction, 0.6, 1.4)
            
            stop_atr_multi = 1.5 * trend_factor * vol_regime_factor * contraction_factor
            stop_atr_multi = np.clip(stop_atr_multi, 1.0, 3.5)  # 限制在1~3.5倍ATR
            
            raw_stop = close - atr * stop_atr_multi
            
            # 支撑位感知: 如果附近有强支撑, 止损设在支撑位下方
            support_candidates = []
            if bb_lower > 0 and bb_lower < close:
                support_candidates.append(bb_lower)
            if low_20 > 0 and low_20 < close:
                support_candidates.append(low_20)
            if ma20 > 0 and ma20 < close:
                support_candidates.append(ma20)
            if ma60_val > 0 and ma60_val < close and ma60_val > close * 0.85:
                support_candidates.append(ma60_val)
            
            # 找最近的、在ATR止损范围内的支撑位
            nearby_supports = [s for s in support_candidates 
                              if s > raw_stop and s < close]
            
            if nearby_supports:
                # 取最近的支撑位, 止损设在支撑位下方 0.5~1 ATR
                nearest_support = max(nearby_supports)
                support_stop = nearest_support - atr * 0.5
                # 取支撑止损和ATR止损的较优值(更紧但不低于原始ATR止损)
                sell_stop = max(support_stop, raw_stop)
            else:
                sell_stop = raw_stop
            
            # 计算实际止损百分比 (不设硬上下限, 但记录供参考)
            stop_pct = (close - sell_stop) / close
            
            # ================================================================
            # 2. 智能止盈系统 — 多目标 + 阻力位自适应
            # ================================================================
            # 上行波动率越大 → 目标可以更高
            # 动量加速 → 目标拉远
            # 趋势强 → 目标拉远
            
            upside_factor = np.clip(vol_asym, 0.7, 2.0)  # 上行波动率优势
            momentum_factor = 1.0 + np.clip(mom_accel * 20, -0.3, 0.5)  # 动量加速调整
            
            # 基础目标: ATR的倍数 (比止损倍数大, 确保盈亏比>1)
            target_atr_multi = stop_atr_multi * 1.8 * upside_factor * momentum_factor
            target_atr_multi = np.clip(target_atr_multi, 2.0, 8.0)
            
            # 三级目标体系
            t1_price = close + atr * target_atr_multi * 0.5    # 保守目标 (50%仓位)
            t2_price = close + atr * target_atr_multi           # 标准目标 (30%仓位)
            t3_price = close + atr * target_atr_multi * 1.5     # 激进目标 (20%仓位)
            
            # 阻力位参考 — 可能限制目标
            resistances = []
            if bb_upper > close:
                resistances.append(('BB上轨', bb_upper, 0.3))
            if high_20 > close:
                resistances.append(('20日高点', high_20, 0.35))
            if high_60 > close:
                resistances.append(('60日高点', high_60, 0.15))
            if ma60_val > close:
                resistances.append(('MA60', ma60_val, 0.2))
            
            # 智能调整: 如果阻力位在T1和T2之间, 把T2调整为阻力位
            if resistances:
                weighted_resist = sum(p * w for _, p, w in resistances) / sum(w for _, _, w in resistances)
                if t1_price < weighted_resist < t2_price:
                    t2_price = weighted_resist
                elif weighted_resist < t1_price and weighted_resist > close:
                    t1_price = weighted_resist
            
            sell_target = t2_price  # 默认展示标准目标
            
            # ================================================================
            # 3. 持有周期预测 — ATR标准化 + 动量推算
            # ================================================================
            # 核心思路: 达到目标需要移动多少个ATR, 历史上平均每天移动多少
            
            target_distance = sell_target - close
            daily_avg_move = atr * 0.6  # 日均有效移动约为ATR的60%
            
            # 效率修正: 趋势型市场移动更快, 震荡型更慢
            eff_adj = np.clip(efficiency, 0.2, 0.9)
            adjusted_daily_move = daily_avg_move * (eff_adj / 0.5)
            
            # 动量修正: 正动量加速到达, 负动量减速
            mom_adj = 1.0 + np.clip(mom_accel * 30, -0.4, 0.6)
            adjusted_daily_move *= mom_adj
            
            if adjusted_daily_move > 0:
                est_days = target_distance / adjusted_daily_move
                est_days = np.clip(est_days, 1, 30)
            else:
                est_days = 10
            
            # 生成持有建议区间 (±30%)
            day_low = max(1, int(est_days * 0.7))
            day_high = max(day_low + 1, int(est_days * 1.3))
            hold_suggestion = f"{day_low}~{day_high}天"
            
            # ================================================================
            # 4. 买入价格 — 成交密集区 + 多层支撑带 + ATR缓冲
            # ================================================================
            # VWAP参考 (成交量加权价格 — 机构常用的价值中枢)
            vwap20_diff = float(last.get('vwap20_diff', 0)) if not pd.isna(last.get('vwap20_diff', np.nan)) else 0
            vwap_center = close / (1 + vwap20_diff) if abs(vwap20_diff) < 0.1 else close
            
            # 多层支撑加权 (权重基于距离当前价的远近和可靠性)
            supports = []
            
            # VWAP价值中枢 (成交密集区, 最可靠的支撑)
            if vwap_center > 0 and vwap_center < close:
                dist = (close - vwap_center) / close
                weight = 0.35 * np.clip(1 - dist * 5, 0.3, 1.0)  # 越近权重越高
                supports.append(('VWAP中枢', vwap_center, weight))
            
            # 布林带中轨 (均值回归锚点)
            if bb_mid > 0 and bb_mid < close:
                supports.append(('BB中轨', bb_mid, 0.20))
            
            # 布林带下轨
            if bb_lower > 0 and bb_lower < close:
                supports.append(('BB下轨', bb_lower, 0.10))
            
            # MA支撑 (根据趋势方向给不同权重)
            if ma5 < close and ma5 > close * 0.95:
                supports.append(('MA5', ma5, 0.15 if trend_strength >= 0.5 else 0.08))
            if ma20 > 0 and ma20 < close:
                supports.append(('MA20', ma20, 0.20 if trend_strength >= 0.3 else 0.12))
            if ma60_val > 0 and ma60_val < close and ma60_val > close * 0.88:
                supports.append(('MA60', ma60_val, 0.15))
            
            # 近期低点 (经过验证的支撑)
            low_5 = float(data['low'].tail(5).min())
            if low_5 < close:
                supports.append(('5日低点', low_5, 0.15))
            
            # 20日低点 (较强支撑)
            if low_20 < close:
                supports.append(('20日低点', low_20, 0.10))
            
            if supports:
                total_weight = sum(w for _, _, w in supports)
                buy_price = sum(p * w for _, p, w in supports) / total_weight
            else:
                buy_price = close - atr * 0.8
            
            buy_price = min(buy_price, close)
            buy_price = max(buy_price, close - atr * 3)  # 不低于3ATR
            
            # 买入上限: 基于ATR和波动率收缩状态
            # 波动率收缩时, 入场容忍度更高(即将突破)
            if vol_contraction < 0.7:
                # 波动率强收缩, 可以追高一点
                upper_atr_multi = 0.6
            elif vol_contraction < 0.9:
                upper_atr_multi = 0.4
            else:
                upper_atr_multi = 0.25
            
            buy_upper = close + atr * upper_atr_multi
            
            # ================================================================
            # 5. Kelly公式仓位管理 — AI概率驱动
            # ================================================================
            # f* = (p * b - q) / b
            # p = AI预测上涨概率
            # q = 1 - p
            # b = 期望收益/期望损失 (odds)
            
            profit_amt = sell_target - close
            loss_amt = close - sell_stop
            risk_reward = round(profit_amt / loss_amt, 2) if loss_amt > 0 else 99
            
            p = np.clip(ai_probability, 0.05, 0.95)
            q = 1 - p
            b = risk_reward if risk_reward > 0 and risk_reward < 99 else 2.0
            
            kelly_fraction = (p * b - q) / b if b > 0 else 0
            kelly_fraction = np.clip(kelly_fraction, 0, 1)
            
            # 使用半Kelly (更保守, 实际中更稳健)
            half_kelly = kelly_fraction * 0.5
            
            # 波动率缩放: 目标组合波动率约15%
            target_portfolio_vol = 0.15
            vol_scalar = target_portfolio_vol / (vol20d + 0.01)
            vol_scalar = np.clip(vol_scalar, 0.3, 2.0)
            
            # AI置信度加权: 评分越高, 越接近Kelly建议
            confidence_weight = np.clip((ai_probability - 0.4) / 0.4, 0.2, 1.0)
            
            # 最终仓位 = 半Kelly × 波动率缩放 × 置信度
            raw_position = half_kelly * vol_scalar * confidence_weight
            
            # 仓位分档 (5%~35%, 步长5%)
            position_value = np.clip(raw_position, 0.05, 0.35)
            position_value = round(position_value * 20) / 20  # 四舍五入到5%
            position_value = max(0.05, position_value)
            position_pct = f"{int(position_value * 100)}%"
            
            # 仓位建议描述
            if position_value >= 0.30:
                position_advice = f"Kelly最优: 高概率({p:.0%}) × 高盈亏比({b:.1f})"
            elif position_value >= 0.20:
                position_advice = f"Kelly建议: 概率{p:.0%}, 盈亏比{b:.1f}, 波动适中"
            elif position_value >= 0.15:
                position_advice = f"适中仓位: 信号可靠但波动率偏高({vol20d:.0%})"
            elif position_value >= 0.10:
                position_advice = f"轻仓试探: 概率{p:.0%}, 盈亏比{b:.1f}"
            else:
                position_advice = f"最小仓位: 信号较弱或风险较高"
            
            # ================================================================
            # 6. 买入条件与时机
            # ================================================================
            buy_conditions = []
            
            if bb_pos <= 0.2:
                buy_conditions.append("布林带下轨附近(超卖)")
            elif bb_pos <= 0.4:
                buy_conditions.append("布林带中下区间")
            
            if rsi <= 30:
                buy_conditions.append(f"RSI超卖({rsi:.0f})")
            elif rsi <= 40:
                buy_conditions.append(f"RSI偏低({rsi:.0f})")
            
            if ma60_diff < -0.05:
                buy_conditions.append(f"远离MA60({ma60_diff*100:.1f}%)超跌")
            elif ma60_diff < -0.02:
                buy_conditions.append(f"低于MA60({ma60_diff*100:.1f}%)")
            
            if macd_hist > 0:
                buy_conditions.append("MACD金叉")
            
            if vol_contraction < 0.7:
                buy_conditions.append("波动率收缩(蓄势)")
            
            vol_r = float(last.get('vol_ratio_5d', 1)) if not pd.isna(last.get('vol_ratio_5d', np.nan)) else 1
            if vol_r >= 1.5:
                buy_conditions.append(f"放量({vol_r:.1f}倍)")
            
            if not buy_conditions:
                buy_conditions.append(f"AI概率{p:.0%}推荐")
            
            buy_condition_str = " + ".join(buy_conditions[:3])
            
            # 买入时机 (结合多因素)
            urgency_score = 0
            if bb_pos <= 0.15: urgency_score += 2
            elif bb_pos <= 0.3: urgency_score += 1
            if rsi <= 30: urgency_score += 2
            elif rsi <= 40: urgency_score += 1
            if vol_contraction < 0.7: urgency_score += 1
            if mom_accel > 0.005: urgency_score += 1
            
            if urgency_score >= 4:
                buy_timing = "可立即建仓(多信号共振)"
            elif urgency_score >= 2:
                buy_timing = "可分批建仓(信号偏积极)"
            elif close <= buy_price * 1.01:
                buy_timing = "接近买入价,等确认信号"
            else:
                buy_timing = "等回调至支撑位再入场"
            
            # ================================================================
            # 7. 卖出条件描述
            # ================================================================
            sell_conditions = []
            sell_conditions.append(f"T1:{t1_price:.2f}(+{(t1_price/close-1)*100:.1f}%减半)")
            sell_conditions.append(f"T2:{sell_target:.2f}(+{(sell_target/close-1)*100:.1f}%)")
            sell_conditions.append(f"止损:{sell_stop:.2f}(-{stop_pct*100:.1f}%)")
            
            sell_condition_str = " | ".join(sell_conditions[:3])
            
            # ================================================================
            # 8. 退出优先级 + 预测有效期
            # ================================================================
            day_mid = int(round(est_days))
            validity_end = max(day_mid + 1, int(est_days * 1.5))
            
            # 退出优先级说明 (核心: 价格为王, 时间兜底)
            exit_rules = (
                f"❶止损{sell_stop:.2f}(-{stop_pct*100:.1f}%) "
                f"❷止盈T2:{sell_target:.2f}(+{(sell_target/close-1)*100:.1f}%) "
                f"❸追踪止损(盈利后回撤1ATR) "
                f"❹超{validity_end}天未触发以上→收紧止损清仓"
            )
            
            return {
                'buy_price': round(buy_price, 2),
                'buy_upper': round(buy_upper, 2),
                'buy_condition': buy_condition_str,
                'buy_timing': buy_timing,
                'sell_target': round(sell_target, 2),
                'sell_stop': round(sell_stop, 2),
                'sell_condition': sell_condition_str,
                'hold_days': hold_suggestion,      # 保留字段兼容, 含义改为"预测有效期"
                'risk_reward': risk_reward,
                'position_pct': position_pct,
                'position_advice': position_advice,
                'bb_lower': round(bb_lower, 2),
                'bb_upper': round(bb_upper, 2),
                'ma5': round(ma5, 2),
                'ma20': round(ma20, 2),
                'ma60': round(ma60_val, 2),
                'atr': round(atr, 2),
                'high_20d': round(high_20, 2),
                'low_20d': round(low_20, 2),
                # 多目标和详细参数
                'sell_t1': round(t1_price, 2),
                'sell_t3': round(t3_price, 2),
                'stop_pct': round(stop_pct * 100, 1),
                'target_pct': round((sell_target / close - 1) * 100, 1),
                'kelly_fraction': round(kelly_fraction, 3),
                'position_value': round(position_value, 2),
                'est_hold_days': round(float(est_days), 1),
                # 退出规则
                'exit_rules': exit_rules,
                'validity_days': validity_end,
            }
        except Exception:
            return {
                'buy_price': round(close * 0.97, 2),
                'buy_upper': round(close * 1.01, 2),
                'buy_condition': 'AI模型推荐',
                'buy_timing': '等回调确认后入场',
                'sell_target': round(close * 1.05, 2),
                'sell_stop': round(close * 0.95, 2),
                'sell_condition': f"目标+5% | 止损-5%",
                'hold_days': '~10天',
                'risk_reward': 1.0,
                'position_pct': '10%',
                'position_advice': '轻仓试探',
                'bb_lower': None, 'bb_upper': None,
                'ma5': None, 'ma20': None, 'ma60': None,
                'atr': None, 'high_20d': None, 'low_20d': None,
                'sell_t1': None, 'sell_t3': None,
                'stop_pct': 5.0, 'target_pct': 5.0,
                'kelly_fraction': 0, 'position_value': 0.10,
                'est_hold_days': 7,
                'exit_rules': '', 'validity_days': 15,
            }
    
    def scan_market(self, cache, pool, 
                    top_n: int = 50,
                    progress_callback=None) -> pd.DataFrame:
        """
        全市场AI评分扫描 (融合基本面/板块/情绪特征)
        
        参数:
            cache: DataCache 实例
            pool: StockPool 实例
            top_n: 返回评分最高的前N只
            progress_callback: 进度回调 fn(current, total)
        
        返回:
            DataFrame: 排序后的全市场评分结果
        """
        import xgboost as xgb
        
        tradeable = pool.get_tradeable_stocks()
        cached = cache.get_all_cached_stocks()
        tradeable_codes = set(tradeable['stock_code'].values)
        cached_codes = set(cached['stock_code'].values)
        valid_codes = sorted(tradeable_codes & cached_codes)
        
        # 名称/行业映射
        code_info = {}
        for _, row in tradeable.iterrows():
            code_info[row['stock_code']] = {
                'name': row.get('stock_name', ''),
                'board': row.get('board_name', ''),
            }
        
        # ====== 预加载外部数据 (一次性获取, 避免重复请求) ======
        bulk_valuation = None
        industry_benchmarks = None
        sector_heat = None
        sentiment_data = None
        
        try:
            from src.data.fundamental import fetch_bulk_valuation, get_industry_benchmarks
            bulk_valuation = fetch_bulk_valuation(cache_minutes=30)
            industry_benchmarks = get_industry_benchmarks(cache_minutes=60)
            if progress_callback:
                print(f"    [Scan] 已加载基本面估值: {len(bulk_valuation)} 只")
        except Exception as e:
            print(f"    [Scan] 基本面数据加载失败(跳过): {e}")
        
        try:
            from src.data.sector_flow import get_sector_heat_map
            sector_heat = get_sector_heat_map()
            if progress_callback:
                print(f"    [Scan] 已加载板块热度: {sector_heat.get('total_sectors', 0)} 个板块")
        except Exception as e:
            print(f"    [Scan] 板块数据加载失败(跳过): {e}")
        
        try:
            # 尝试读取缓存的情绪数据 (避免每次扫描都重新采集)
            import json
            sentiment_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'data', 'market_sentiment.json')
            if os.path.exists(sentiment_path):
                with open(sentiment_path, 'r', encoding='utf-8') as f:
                    sentiment_data = json.load(f)
                if progress_callback:
                    print(f"    [Scan] 已加载情绪数据: {sentiment_data.get('sentiment_score', '?')}分")
        except Exception:
            pass
        
        results = []
        total = len(valid_codes)
        
        for i, code in enumerate(valid_codes):
            if progress_callback and (i + 1) % 200 == 0:
                progress_callback(i + 1, total)
            
            try:
                df = cache.load_kline(code)
                if df is None or len(df) < 100:
                    continue
                
                info = code_info.get(code, {})
                board = info.get('board', '')
                
                # 构建外部特征 (预测时: 全量数据可用)
                extra = build_extra_features(
                    stock_code=code,
                    board_name=board,
                    bulk_valuation=bulk_valuation,
                    industry_benchmarks=industry_benchmarks,
                    sector_heat=sector_heat,
                    sentiment_data=sentiment_data,
                )
                
                data = compute_v2_features(df, extra_features=extra)
                if data.empty or len(data) < 65:
                    continue
                
                last_row = data.iloc[-1:]
                feat_cols = [c for c in last_row.columns if c not in self.EXCLUDE_COLS]
                X = last_row[feat_cols].copy()
                X = X.replace([np.inf, -np.inf], np.nan)
                
                if self.feature_names is not None:
                    missing = set(self.feature_names) - set(X.columns)
                    for col in missing:
                        X[col] = np.nan
                    X = X[self.feature_names]
                
                dmat = xgb.DMatrix(X, feature_names=list(X.columns))
                prob = float(self.model.predict(dmat)[0])
                score = round(prob * 100, 1)
                
                close = float(data.iloc[-1]['close'])
                
                # 收集关键指标
                vol20 = data.iloc[-1].get('volatility_20d')
                bb = data.iloc[-1].get('bb_pos')
                rsi = data.iloc[-1].get('rsi_14')
                ret5 = data.iloc[-1].get('ret_5d')
                vol_r = data.iloc[-1].get('vol_ratio_5d')
                ma60d = data.iloc[-1].get('ma60_diff')
                
                # ====== 计算买卖建议 (传入AI概率, 驱动Kelly仓位) ======
                advice = self._compute_trade_advice(data, close, ai_probability=prob)
                
                results.append({
                    'stock_code': code,
                    'stock_name': info.get('name', ''),
                    'board_name': board,
                    'ai_score': score,
                    'probability': round(prob, 4),
                    'close': close,
                    'volatility_20d': round(float(vol20), 4) if vol20 is not None and not pd.isna(vol20) else None,
                    'bb_pos': round(float(bb), 4) if bb is not None and not pd.isna(bb) else None,
                    'rsi_14': round(float(rsi), 1) if rsi is not None and not pd.isna(rsi) else None,
                    'ret_5d': round(float(ret5) * 100, 2) if ret5 is not None and not pd.isna(ret5) else None,
                    'vol_ratio': round(float(vol_r), 2) if vol_r is not None and not pd.isna(vol_r) else None,
                    'ma60_diff': round(float(ma60d) * 100, 2) if ma60d is not None and not pd.isna(ma60d) else None,
                    **advice,
                })
                
            except Exception:
                continue
        
        if not results:
            return pd.DataFrame()
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('ai_score', ascending=False).reset_index(drop=True)
        df_results['rank'] = range(1, len(df_results) + 1)
        
        return df_results
