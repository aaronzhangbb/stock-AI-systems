# -*- coding: utf-8 -*-
"""
形态聚类引擎 (第二层)
====================
完全脱离指标体系，纯粹从"价格走势形状"中发现规律

核心思路:
1. 将每只股票过去N天的走势归一化为向量(形状指纹)
2. 用 K-Means 自动聚类为几百种"形态"
3. 标注每种形态的历史胜率
4. 每天扫描: 寻找与"高胜率形态"最相似的股票

特征维度:
- 价格走势形状 (归一化收盘价曲线)
- 成交量走势形状 (归一化成交量曲线)
- K线实体/影线序列 (微观形态)
"""
import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ============================================================
# 配置
# ============================================================
WINDOW = 20           # 观察窗口: 过去20个交易日
FUTURE_DAYS = 5       # 预测窗口: 未来5天
TARGET_RETURN = 0.03  # 目标收益: 3%
N_CLUSTERS = 200      # 聚类数量
MIN_SAMPLES = 50      # 每个聚类最少样本数(否则合并)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


# ============================================================
# 走势特征提取
# ============================================================

def extract_pattern_vector(df: pd.DataFrame, idx: int, window: int = WINDOW) -> Optional[np.ndarray]:
    """
    从K线DataFrame的指定位置提取形态向量
    
    输入: df[idx-window+1 : idx+1] 共 window 行
    输出: 归一化的形态向量 (固定长度)
    
    向量构成:
      [0:window]      归一化收盘价曲线 (价格形状)
      [window:2*window] 归一化成交量曲线 (量能形状)
      [2*window:3*window] K线实体比率序列 (微观形态)
      [3*window:4*window] 涨跌序列 (动量方向)
      [4*window:4*window+5] 统计特征 (波动率/偏度/趋势等)
    
    总维度: 4*window + 5 = 85
    """
    if idx < window - 1 or idx >= len(df):
        return None
    
    chunk = df.iloc[idx - window + 1: idx + 1]
    if len(chunk) < window:
        return None
    
    close = chunk['close'].values.astype(float)
    volume = chunk['volume'].values.astype(float)
    high = chunk['high'].values.astype(float)
    low = chunk['low'].values.astype(float)
    open_ = chunk['open'].values.astype(float)
    
    # 检查有效性
    if close[0] <= 0 or np.any(np.isnan(close)) or np.any(close <= 0):
        return None
    
    # ---- 1. 归一化收盘价曲线 (价格形状) ----
    # 用第一天收盘价做基准, 转为相对变化
    price_norm = close / close[0] - 1.0  # 范围约 [-0.3, +0.3]
    
    # ---- 2. 归一化成交量曲线 (量能形状) ----
    vol_mean = volume.mean()
    if vol_mean <= 0:
        vol_norm = np.zeros(window)
    else:
        vol_norm = volume / vol_mean - 1.0  # 0 = 平均, >0 = 放量, <0 = 缩量
        vol_norm = np.clip(vol_norm, -2, 5)  # 防止极端值
    
    # ---- 3. K线实体比率序列 (微观形态) ----
    range_ = high - low + 1e-10
    body_ratio = (close - open_) / range_  # [-1, +1], 正=阳线, 负=阴线
    
    # ---- 4. 涨跌序列 ----
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0], returns])  # 补齐第一天
    returns = np.clip(returns, -0.15, 0.15)  # 限幅
    
    # ---- 5. 统计摘要特征 ----
    total_return = close[-1] / close[0] - 1           # 总收益
    volatility = np.std(returns)                       # 波动率
    skewness = _safe_skew(returns)                     # 偏度
    max_drawdown = _max_drawdown(close)                # 最大回撤
    trend_strength = _trend_strength(close)            # 趋势强度
    
    stats = np.array([total_return, volatility, skewness, max_drawdown, trend_strength])
    
    # ---- 拼接 ----
    vector = np.concatenate([price_norm, vol_norm, body_ratio, returns, stats])
    
    # 替换NaN/Inf
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    
    return vector.astype(np.float32)


def _safe_skew(arr):
    """安全计算偏度"""
    n = len(arr)
    if n < 3:
        return 0.0
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-10:
        return 0.0
    return np.mean(((arr - mean) / std) ** 3)


def _max_drawdown(prices):
    """计算最大回撤 (负值)"""
    peak = np.maximum.accumulate(prices)
    dd = (prices - peak) / (peak + 1e-10)
    return float(np.min(dd))


def _trend_strength(prices):
    """趋势强度: 净位移 / 总路径长度, [-1, 1]"""
    net = abs(prices[-1] - prices[0])
    gross = np.sum(np.abs(np.diff(prices)))
    if gross < 1e-10:
        return 0.0
    return float(net / gross) * np.sign(prices[-1] - prices[0])


def get_vector_dim(window: int = WINDOW) -> int:
    """获取向量维度"""
    return 4 * window + 5


# ============================================================
# 形态聚类引擎
# ============================================================

class PatternEngine:
    """
    形态聚类引擎
    
    训练流程:
        1. 从全市场历史数据提取形态向量
        2. K-Means 聚类
        3. 标注每个聚类的胜率
        4. 保存模型
    
    预测流程:
        1. 提取当前股票的形态向量
        2. 找到最近的聚类
        3. 返回该聚类的历史胜率
    """
    
    def __init__(self, n_clusters: int = N_CLUSTERS, window: int = WINDOW):
        self.n_clusters = n_clusters
        self.window = window
        self.vector_dim = get_vector_dim(window)
        
        self.kmeans = None
        self.scaler = None
        self.cluster_stats = {}   # {cluster_id: {win_rate, avg_return, count, ...}}
        self.is_trained = False
    
    def build_training_data(self, cache, pool, 
                            progress_callback=None) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        构建训练数据: 从全市场历史K线提取形态向量 + 标签
        
        返回:
            vectors: (N, vector_dim) 形态向量矩阵
            labels:  (N,) 0/1标签 (未来5天涨>3%)
            meta:    [(stock_code, date), ...] 元信息
        """
        tradeable = pool.get_tradeable_stocks()
        cached = cache.get_all_cached_stocks()
        tradeable_codes = set(tradeable['stock_code'].values)
        cached_codes = set(cached['stock_code'].values)
        valid_codes = sorted(tradeable_codes & cached_codes)
        
        all_vectors = []
        all_labels = []
        all_meta = []
        
        total = len(valid_codes)
        for i, code in enumerate(valid_codes):
            if progress_callback and (i + 1) % 500 == 0:
                progress_callback(i + 1, total)
            
            try:
                df = cache.load_kline(code)
                if df is None or len(df) < self.window + FUTURE_DAYS + 10:
                    continue
                
                df = df.sort_values('date').reset_index(drop=True)
                close = df['close'].values.astype(float)
                high = df['high'].values.astype(float)
                
                # 从 window 开始, 到 len-future_days 结束
                for idx in range(self.window - 1, len(df) - FUTURE_DAYS):
                    vec = extract_pattern_vector(df, idx, self.window)
                    if vec is None:
                        continue
                    
                    # 标签: 未来5天最高价涨幅 > 3%
                    current_close = close[idx]
                    future_high = high[idx + 1: idx + 1 + FUTURE_DAYS].max()
                    future_return = (future_high - current_close) / current_close
                    label = 1 if future_return >= TARGET_RETURN else 0
                    
                    all_vectors.append(vec)
                    all_labels.append(label)
                    all_meta.append((code, str(df.iloc[idx]['date'])))
                    
            except Exception:
                continue
        
        if not all_vectors:
            return np.array([]), np.array([]), []
        
        vectors = np.array(all_vectors, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.int32)
        
        return vectors, labels, all_meta
    
    def train(self, vectors: np.ndarray, labels: np.ndarray, 
              meta: list = None) -> Dict:
        """
        训练聚类模型
        
        步骤:
            1. StandardScaler 标准化
            2. K-Means 聚类
            3. 标注每个聚类的胜率
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import MiniBatchKMeans
        
        n_samples = len(vectors)
        print(f"  训练数据: {n_samples:,} 个形态, 维度 {vectors.shape[1]}")
        print(f"  正样本率: {labels.mean()*100:.1f}%")
        
        # 1. 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(vectors)
        
        # 2. K-Means 聚类
        print(f"  聚类中 (K={self.n_clusters})...")
        t0 = time.time()
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=10000,
            n_init=3,
            max_iter=300,
            random_state=42,
        )
        cluster_ids = self.kmeans.fit_predict(X_scaled)
        t_cluster = time.time() - t0
        print(f"  聚类完成, 耗时 {t_cluster:.1f}秒")
        
        # 3. 标注每个聚类
        self.cluster_stats = {}
        
        # 按时间分割: 用前80%训练胜率, 后20%验证
        if meta:
            dates = [m[1] for m in meta]
            date_series = pd.to_datetime(dates)
            # 训练集: <= 2025-06-30
            train_mask = date_series <= '2025-06-30'
            val_mask = (date_series > '2025-06-30') & (date_series <= '2025-12-31')
            test_mask = date_series > '2025-12-31'
        else:
            n_train = int(n_samples * 0.7)
            n_val = int(n_samples * 0.15)
            train_mask = np.zeros(n_samples, dtype=bool)
            train_mask[:n_train] = True
            val_mask = np.zeros(n_samples, dtype=bool)
            val_mask[n_train:n_train+n_val] = True
            test_mask = np.zeros(n_samples, dtype=bool)
            test_mask[n_train+n_val:] = True
        
        for cid in range(self.n_clusters):
            mask = cluster_ids == cid
            count = mask.sum()
            
            if count < 10:
                self.cluster_stats[cid] = {
                    'count': int(count),
                    'win_rate': 0.5,
                    'avg_return': 0.0,
                    'is_valid': False,
                    'train_win_rate': 0.5,
                    'val_win_rate': None,
                    'test_win_rate': None,
                }
                continue
            
            # 训练集胜率
            train_in_cluster = mask & train_mask
            train_wr = labels[train_in_cluster].mean() if train_in_cluster.sum() > 5 else 0.5
            
            # 验证集胜率
            val_in_cluster = mask & val_mask
            val_wr = labels[val_in_cluster].mean() if val_in_cluster.sum() > 5 else None
            
            # 测试集胜率
            test_in_cluster = mask & test_mask
            test_wr = labels[test_in_cluster].mean() if test_in_cluster.sum() > 5 else None
            
            # 总体胜率
            overall_wr = labels[mask].mean()
            
            self.cluster_stats[cid] = {
                'count': int(count),
                'win_rate': round(float(overall_wr), 4),
                'train_win_rate': round(float(train_wr), 4),
                'val_win_rate': round(float(val_wr), 4) if val_wr is not None else None,
                'test_win_rate': round(float(test_wr), 4) if test_wr is not None else None,
                'is_valid': count >= MIN_SAMPLES,
            }
        
        self.is_trained = True
        
        # 统计
        valid_clusters = [c for c in self.cluster_stats.values() if c['is_valid']]
        high_wr = [c for c in valid_clusters if c['win_rate'] >= 0.6]
        very_high_wr = [c for c in valid_clusters if c['win_rate'] >= 0.7]
        low_wr = [c for c in valid_clusters if c['win_rate'] <= 0.4]
        
        report = {
            'total_clusters': self.n_clusters,
            'valid_clusters': len(valid_clusters),
            'high_winrate_clusters': len(high_wr),
            'very_high_winrate_clusters': len(very_high_wr),
            'low_winrate_clusters': len(low_wr),
            'cluster_time': round(t_cluster, 1),
            'total_samples': n_samples,
            'train_samples': int(train_mask.sum()) if isinstance(train_mask, np.ndarray) else int(train_mask.sum()),
            'val_samples': int(val_mask.sum()) if isinstance(val_mask, np.ndarray) else int(val_mask.sum()),
            'test_samples': int(test_mask.sum()) if isinstance(test_mask, np.ndarray) else int(test_mask.sum()),
        }
        
        # 胜率分布
        wr_list = [c['win_rate'] for c in valid_clusters]
        if wr_list:
            report['avg_win_rate'] = round(float(np.mean(wr_list)), 4)
            report['max_win_rate'] = round(float(np.max(wr_list)), 4)
            report['min_win_rate'] = round(float(np.min(wr_list)), 4)
            report['std_win_rate'] = round(float(np.std(wr_list)), 4)
        
        # 高胜率聚类的验证集表现
        if high_wr:
            val_wrs = [c['val_win_rate'] for c in high_wr if c['val_win_rate'] is not None]
            test_wrs = [c['test_win_rate'] for c in high_wr if c['test_win_rate'] is not None]
            if val_wrs:
                report['high_wr_val_avg'] = round(float(np.mean(val_wrs)), 4)
            if test_wrs:
                report['high_wr_test_avg'] = round(float(np.mean(test_wrs)), 4)
        
        return report
    
    def predict_single(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        对单只股票的当前走势进行形态匹配
        
        返回:
            {cluster_id, win_rate, confidence, pattern_desc}
        """
        if not self.is_trained or self.kmeans is None:
            return None
        
        if df is None or len(df) < self.window:
            return None
        
        df = df.sort_values('date').reset_index(drop=True)
        idx = len(df) - 1
        
        vec = extract_pattern_vector(df, idx, self.window)
        if vec is None:
            return None
        
        # 标准化
        vec_scaled = self.scaler.transform(vec.reshape(1, -1))
        
        # 找最近聚类
        cluster_id = int(self.kmeans.predict(vec_scaled)[0])
        
        # 计算与聚类中心的距离
        center = self.kmeans.cluster_centers_[cluster_id]
        distance = float(np.linalg.norm(vec_scaled[0] - center))
        
        # 获取聚类统计
        stats = self.cluster_stats.get(cluster_id, {})
        win_rate = stats.get('win_rate', 0.5)
        count = stats.get('count', 0)
        is_valid = stats.get('is_valid', False)
        
        # 置信度: 基于 (1) 聚类有效性 (2) 样本距离 (3) 胜率偏离中性
        if not is_valid:
            confidence = 0
        else:
            # 距离越近越好 (用指数衰减)
            dist_score = np.exp(-distance / 5.0)
            # 胜率偏离0.5越多越好
            wr_score = abs(win_rate - 0.5) * 2
            # 样本量越多越好
            count_score = min(count / 500, 1.0)
            confidence = round((dist_score * 0.4 + wr_score * 0.4 + count_score * 0.2) * 100, 1)
        
        # 形态描述
        pattern_desc = self._describe_pattern(vec)
        
        return {
            'cluster_id': cluster_id,
            'win_rate': round(win_rate * 100, 1),
            'confidence': confidence,
            'distance': round(distance, 2),
            'sample_count': count,
            'is_valid': is_valid,
            'pattern_desc': pattern_desc,
            'val_win_rate': stats.get('val_win_rate'),
            'test_win_rate': stats.get('test_win_rate'),
        }
    
    def scan_market(self, cache, pool, top_n: int = 50,
                    progress_callback=None) -> pd.DataFrame:
        """
        全市场形态扫描
        
        对每只股票的当前走势进行形态匹配, 返回:
        - 匹配高胜率形态的股票
        - 排序: 胜率 × 置信度
        """
        if not self.is_trained:
            return pd.DataFrame()
        
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
            if progress_callback and (i + 1) % 500 == 0:
                progress_callback(i + 1, total)
            
            try:
                df = cache.load_kline(code)
                if df is None or len(df) < self.window + 5:
                    continue
                
                result = self.predict_single(df)
                if result is None or not result['is_valid']:
                    continue
                
                info = code_info.get(code, {})
                close = float(df.sort_values('date').iloc[-1]['close'])
                
                # 综合评分 = 胜率权重 + 置信度权重
                pattern_score = result['win_rate'] * 0.7 + result['confidence'] * 0.3
                
                results.append({
                    'stock_code': code,
                    'stock_name': info.get('name', ''),
                    'board_name': info.get('board', ''),
                    'close': close,
                    'cluster_id': result['cluster_id'],
                    'pattern_win_rate': result['win_rate'],
                    'pattern_confidence': result['confidence'],
                    'pattern_score': round(pattern_score, 1),
                    'pattern_distance': result['distance'],
                    'pattern_samples': result['sample_count'],
                    'pattern_desc': result['pattern_desc'],
                })
                
            except Exception:
                continue
        
        if not results:
            return pd.DataFrame()
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('pattern_score', ascending=False).reset_index(drop=True)
        
        return df_results.head(top_n)
    
    def _describe_pattern(self, vec: np.ndarray) -> str:
        """根据向量生成人类可读的形态描述"""
        w = self.window
        price_curve = vec[:w]
        vol_curve = vec[w:2*w]
        body_ratios = vec[2*w:3*w]
        
        total_return = vec[4*w]      # 统计特征
        volatility = vec[4*w + 1]
        trend = vec[4*w + 4]
        
        parts = []
        
        # 价格趋势
        if total_return > 0.05:
            parts.append("强势上涨")
        elif total_return > 0.02:
            parts.append("温和上涨")
        elif total_return > -0.02:
            parts.append("横盘震荡")
        elif total_return > -0.05:
            parts.append("温和下跌")
        else:
            parts.append("大幅下跌")
        
        # 波动率
        if volatility > 0.04:
            parts.append("高波动")
        elif volatility > 0.02:
            parts.append("中波动")
        else:
            parts.append("低波动")
        
        # 量能
        last_5_vol = vol_curve[-5:].mean()
        if last_5_vol > 0.5:
            parts.append("近期放量")
        elif last_5_vol < -0.3:
            parts.append("近期缩量")
        
        # 近期K线
        last_3_body = body_ratios[-3:].mean()
        if last_3_body > 0.3:
            parts.append("连续阳线")
        elif last_3_body < -0.3:
            parts.append("连续阴线")
        
        # 趋势强度
        if abs(trend) > 0.5:
            parts.append("趋势明确")
        elif abs(trend) < 0.2:
            parts.append("方向不明")
        
        return " · ".join(parts) if parts else "普通形态"
    
    def save(self, path: str = None):
        """保存模型"""
        if path is None:
            path = os.path.join(DATA_DIR, 'pattern_engine.pkl')
        
        data = {
            'n_clusters': self.n_clusters,
            'window': self.window,
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'cluster_stats': self.cluster_stats,
            'is_trained': self.is_trained,
            'save_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        return path
    
    @classmethod
    def load(cls, path: str = None) -> 'PatternEngine':
        """加载模型"""
        if path is None:
            path = os.path.join(DATA_DIR, 'pattern_engine.pkl')
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"形态引擎模型不存在: {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        engine = cls(n_clusters=data['n_clusters'], window=data['window'])
        engine.kmeans = data['kmeans']
        engine.scaler = data['scaler']
        engine.cluster_stats = data['cluster_stats']
        engine.is_trained = data['is_trained']
        
        return engine
    
    def get_top_clusters(self, top_n: int = 20) -> List[Dict]:
        """获取胜率最高的聚类"""
        valid = [(cid, s) for cid, s in self.cluster_stats.items() if s['is_valid']]
        valid.sort(key=lambda x: x[1]['win_rate'], reverse=True)
        
        results = []
        for cid, s in valid[:top_n]:
            results.append({
                'cluster_id': cid,
                **s,
            })
        return results
    
    def get_bottom_clusters(self, top_n: int = 10) -> List[Dict]:
        """获取胜率最低的聚类 (避开的形态)"""
        valid = [(cid, s) for cid, s in self.cluster_stats.items() if s['is_valid']]
        valid.sort(key=lambda x: x[1]['win_rate'])
        
        results = []
        for cid, s in valid[:top_n]:
            results.append({
                'cluster_id': cid,
                **s,
            })
        return results
