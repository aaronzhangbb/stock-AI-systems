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

def compute_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算V2高阶特征集 (~100+ 个特征)
    
    输入: 含有 date/open/high/low/close/volume/pctChg 的 DataFrame
    输出: 增加了100+个特征列的 DataFrame
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
    
    return data


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
                  progress_callback=None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    构建全市场训练数据集
    
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
            
            # 计算特征
            data = compute_v2_features(df)
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
    
    def score_single(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        对单只股票打分
        
        参数:
            df: 该股票的历史K线DataFrame (需至少100行)
        
        返回:
            dict: {score, probability, features} 或 None
        """
        import xgboost as xgb
        
        if df is None or len(df) < 100:
            return None
        
        try:
            data = compute_v2_features(df)
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
    
    def scan_market(self, cache, pool, 
                    top_n: int = 50,
                    progress_callback=None) -> pd.DataFrame:
        """
        全市场AI评分扫描
        
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
        
        results = []
        total = len(valid_codes)
        
        for i, code in enumerate(valid_codes):
            if progress_callback and (i + 1) % 200 == 0:
                progress_callback(i + 1, total)
            
            try:
                df = cache.load_kline(code)
                if df is None or len(df) < 100:
                    continue
                
                data = compute_v2_features(df)
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
                info = code_info.get(code, {})
                
                # 收集关键指标
                vol20 = data.iloc[-1].get('volatility_20d')
                bb = data.iloc[-1].get('bb_pos')
                rsi = data.iloc[-1].get('rsi_14')
                ret5 = data.iloc[-1].get('ret_5d')
                vol_r = data.iloc[-1].get('vol_ratio_5d')
                ma60d = data.iloc[-1].get('ma60_diff')
                
                results.append({
                    'stock_code': code,
                    'stock_name': info.get('name', ''),
                    'board_name': info.get('board', ''),
                    'ai_score': score,
                    'probability': round(prob, 4),
                    'close': close,
                    'volatility_20d': round(float(vol20), 4) if vol20 is not None and not pd.isna(vol20) else None,
                    'bb_pos': round(float(bb), 4) if bb is not None and not pd.isna(bb) else None,
                    'rsi_14': round(float(rsi), 1) if rsi is not None and not pd.isna(rsi) else None,
                    'ret_5d': round(float(ret5) * 100, 2) if ret5 is not None and not pd.isna(ret5) else None,
                    'vol_ratio': round(float(vol_r), 2) if vol_r is not None and not pd.isna(vol_r) else None,
                    'ma60_diff': round(float(ma60d) * 100, 2) if ma60d is not None and not pd.isna(ma60d) else None,
                })
                
            except Exception:
                continue
        
        if not results:
            return pd.DataFrame()
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('ai_score', ascending=False).reset_index(drop=True)
        df_results['rank'] = range(1, len(df_results) + 1)
        
        return df_results
