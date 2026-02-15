# -*- coding: utf-8 -*-
"""
Transformer 时序引擎 (第三层)
=============================
将每日K线 [O,H,L,C,V] 视为Token序列，用轻量Transformer Encoder
学习长周期价格模式和上下文关联。

核心思路:
1. 输入: 过去60天的 [O,H,L,C,V, 衍生特征] → 形成 (60, D) 的序列
2. 位置编码: 可学习的时间位置嵌入
3. Transformer Encoder: 4层 × 4头注意力
4. 输出: 未来5天涨>3%的概率
5. 利用RTX 5080 GPU加速训练与推理

模型约 ~80万参数, 远小于16G显存上限
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# 序列长度与特征
SEQ_LEN = 60       # 过去60个交易日
FUTURE_DAYS = 5    # 预测窗口
TARGET_RETURN = 0.03


# ============================================================
# 特征提取 (每日Token)
# ============================================================

def build_daily_features(df: pd.DataFrame) -> np.ndarray:
    """
    从K线DataFrame构建每日特征矩阵 (纯向量化, 高性能)
    
    每天一个Token, 维度D = 12:
      [0] 归一化收盘价 (相对60日前)
      [1] 归一化最高价
      [2] 归一化最低价
      [3] 归一化开盘价
      [4] 归一化成交量 (相对20日均量)
      [5] 日收益率
      [6] 振幅 (high-low)/close
      [7] K线实体比率 (close-open)/(high-low)
      [8] 上影线比率
      [9] 下影线比率
      [10] 5日均线偏离
      [11] 20日均线偏离
    
    返回: (T, 12) 的numpy数组
    """
    data = df.sort_values('date').reset_index(drop=True)
    close = data['close'].values.astype(np.float64)
    high = data['high'].values.astype(np.float64)
    low = data['low'].values.astype(np.float64)
    open_ = data['open'].values.astype(np.float64)
    volume = data['volume'].values.astype(np.float64)
    
    T = len(data)
    features = np.zeros((T, 12), dtype=np.float32)
    
    # 向量化计算基准价 (60日前收盘价)
    base = np.empty(T, dtype=np.float64)
    base[:59] = close[0]
    base[59:] = close[:T - 59]
    base[base <= 0] = 1.0
    
    # [0-3] 归一化OHLC
    features[:, 0] = close / base - 1.0
    features[:, 1] = high / base - 1.0
    features[:, 2] = low / base - 1.0
    features[:, 3] = open_ / base - 1.0
    
    # [4] 归一化成交量
    vol_ma20 = pd.Series(volume).rolling(20, min_periods=1).mean().values
    vol_ma20[vol_ma20 <= 0] = 1.0
    features[:, 4] = np.clip(volume / vol_ma20 - 1.0, -2, 10)
    
    # [5] 日收益率
    features[1:, 5] = np.clip(close[1:] / np.maximum(close[:-1], 1e-10) - 1.0, -0.2, 0.2)
    
    # [6] 振幅
    features[:, 6] = (high - low) / np.maximum(close, 1e-10)
    
    # [7-9] K线形态
    hl_range = high - low + 1e-10
    features[:, 7] = (close - open_) / hl_range
    features[:, 8] = (high - np.maximum(close, open_)) / hl_range
    features[:, 9] = (np.minimum(close, open_) - low) / hl_range
    
    # [10-11] 均线偏离
    ma5 = pd.Series(close).rolling(5, min_periods=1).mean().values
    ma20 = pd.Series(close).rolling(20, min_periods=1).mean().values
    features[:, 10] = np.clip(close / np.maximum(ma5, 1e-10) - 1.0, -0.3, 0.3)
    features[:, 11] = np.clip(close / np.maximum(ma20, 1e-10) - 1.0, -0.5, 0.5)
    
    # NaN → 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features


# ============================================================
# PyTorch 模型定义
# ============================================================

def _get_torch():
    """延迟导入torch"""
    import torch
    import torch.nn as nn
    return torch, nn


class StockTransformer:
    """
    轻量级Transformer Encoder用于股票时序预测
    
    架构:
        Input (seq_len, 12) 
        → Linear(12, d_model=64) 投影层
        → + Positional Encoding (可学习)
        → Transformer Encoder (4层, 4头, d_ff=256)
        → [CLS] token 或 全局池化
        → Linear(64, 1) → Sigmoid
    
    参数量: ~80万
    """
    
    def __init__(self, d_input: int = 12, d_model: int = 64, 
                 nhead: int = 4, num_layers: int = 4, d_ff: int = 256,
                 seq_len: int = SEQ_LEN, dropout: float = 0.1):
        torch, nn = _get_torch()
        
        self.d_input = d_input
        self.d_model = d_model
        self.seq_len = seq_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = _TransformerModel(
            d_input=d_input, d_model=d_model, nhead=nhead,
            num_layers=num_layers, d_ff=d_ff, seq_len=seq_len, dropout=dropout
        ).to(self.device)
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  模型参数: {total_params:,} (可训练: {trainable_params:,})")
        print(f"  设备: {self.device}")
    
    def build_dataset(self, cache, pool, progress_callback=None) -> Dict:
        """
        构建训练数据集
        
        返回: {
            'X_train': (N, seq_len, 12),
            'y_train': (N,),
            'X_val': ..., 'y_val': ...,
            'X_test': ..., 'y_test': ...
        }
        """
        torch, _ = _get_torch()
        
        tradeable = pool.get_tradeable_stocks()
        cached = cache.get_all_cached_stocks()
        valid_codes = sorted(set(tradeable['stock_code']) & set(cached['stock_code']))
        
        all_X = []
        all_y = []
        all_dates = []
        
        total = len(valid_codes)
        for i, code in enumerate(valid_codes):
            if progress_callback and (i + 1) % 500 == 0:
                progress_callback(i + 1, total)
            
            try:
                df = cache.load_kline(code)
                if df is None or len(df) < self.seq_len + FUTURE_DAYS + 10:
                    continue
                
                df = df.sort_values('date').reset_index(drop=True)
                features = build_daily_features(df)
                close = df['close'].values.astype(float)
                high = df['high'].values.astype(float)
                dates = df['date'].astype(str).values
                T = len(df)
                
                # 向量化构建滑动窗口序列
                # 有效范围: idx from seq_len-1 to T-FUTURE_DAYS-1
                start = self.seq_len - 1
                end = T - FUTURE_DAYS
                if start >= end:
                    continue
                
                # 批量构建标签
                indices = np.arange(start, end)
                cur_closes = close[indices]
                
                # 未来5天最高价 (向量化)
                fut_highs = np.array([high[i + 1: i + 1 + FUTURE_DAYS].max() for i in indices])
                labels = ((fut_highs - cur_closes) / cur_closes >= TARGET_RETURN).astype(np.float32)
                
                # 批量构建序列 (滑动窗口)
                seq_indices = np.arange(self.seq_len)[None, :] + indices[:, None] - (self.seq_len - 1)
                seqs = features[seq_indices]  # (N, seq_len, 12)
                
                # 过滤无效序列
                valid = ~(np.isnan(seqs).any(axis=(1, 2)) | (np.abs(seqs).max(axis=(1, 2)) > 100))
                
                all_X.append(seqs[valid])
                all_y.append(labels[valid])
                all_dates.extend(dates[indices[valid]].tolist())
                    
            except Exception:
                continue
        
        if not all_X:
            return {}
        
        X = np.concatenate(all_X, axis=0).astype(np.float32)
        y = np.concatenate(all_y, axis=0).astype(np.float32)
        dates_arr = np.array(all_dates)
        
        # 时间分割
        train_mask = dates_arr <= '2025-06-30'
        val_mask = (dates_arr > '2025-06-30') & (dates_arr <= '2025-12-31')
        test_mask = dates_arr > '2025-12-31'
        
        dataset = {
            'X_train': X[train_mask], 'y_train': y[train_mask],
            'X_val': X[val_mask], 'y_val': y[val_mask],
            'X_test': X[test_mask], 'y_test': y[test_mask],
            'total': len(X),
            'pos_rate': float(y.mean()),
        }
        
        return dataset
    
    def train_model(self, dataset: Dict, epochs: int = 30, batch_size: int = 2048,
                    lr: float = 1e-3, weight_decay: float = 1e-4) -> Dict:
        """
        训练Transformer模型 (数据留在CPU, 批量传输到GPU)
        """
        torch, nn = _get_torch()
        from torch.utils.data import TensorDataset, DataLoader
        
        # 数据留在CPU, 用pin_memory加速传输
        X_train = torch.FloatTensor(dataset['X_train'])
        y_train = torch.FloatTensor(dataset['y_train'])
        X_val_np = dataset['X_val']
        y_val_np = dataset['y_val']
        
        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                  drop_last=True, pin_memory=True, num_workers=0)
        
        # 正负样本比例 → 加权Loss
        pos_rate = dataset['pos_rate']
        neg_rate = 1.0 - pos_rate
        pos_weight = torch.tensor([neg_rate / (pos_rate + 1e-6)]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
        best_val_auc = 0
        best_state = None
        patience = 8
        no_improve = 0
        history = []
        
        def _batch_predict(X_np):
            """分批GPU推理 (数据在CPU/numpy)"""
            all_logits = []
            bs = 8192
            for s in range(0, len(X_np), bs):
                e = min(s + bs, len(X_np))
                bx = torch.FloatTensor(X_np[s:e]).to(self.device)
                logits = self.model(bx).squeeze(-1)
                all_logits.append(logits.cpu())
            return torch.cat(all_logits, dim=0)
        
        print(f"\n  开始训练: {epochs} epochs, batch={batch_size}, lr={lr}")
        print(f"  训练集: {len(X_train):,}, 验证集: {len(X_val_np):,}")
        
        for epoch in range(epochs):
            # -- Train --
            self.model.train()
            train_loss = 0
            n_batches = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_X).squeeze(-1)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= max(n_batches, 1)
            scheduler.step()
            
            # -- Validate (分批GPU推理, 结果回CPU) --
            self.model.eval()
            with torch.no_grad():
                val_logits = _batch_predict(X_val_np)  # CPU tensor
                val_y_tensor = torch.FloatTensor(y_val_np)  # CPU tensor
                # criterion在GPU上, 需要创建CPU版本计算val_loss
                val_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.cpu())
                val_loss = val_criterion(val_logits, val_y_tensor).item()
                val_probs = torch.sigmoid(val_logits).numpy()
                val_labels = y_val_np
            
            # AUC
            from sklearn.metrics import roc_auc_score
            try:
                val_auc = roc_auc_score(val_labels, val_probs)
            except Exception:
                val_auc = 0.5
            
            # Precision@Top50
            top_k = 50
            if len(val_probs) > top_k:
                top_idx = np.argsort(val_probs)[-top_k:]
                p_at_k = val_labels[top_idx].mean()
            else:
                p_at_k = val_labels.mean()
            
            history.append({
                'epoch': epoch + 1,
                'train_loss': round(train_loss, 4),
                'val_loss': round(val_loss, 4),
                'val_auc': round(val_auc, 4),
                'val_p@50': round(float(p_at_k), 4),
            })
            
            if (epoch + 1) % 3 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:>2d}/{epochs}  "
                      f"loss={train_loss:.4f}/{val_loss:.4f}  "
                      f"AUC={val_auc:.4f}  P@50={p_at_k:.3f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch+1} (patience={patience})")
                break
        
        # 恢复最佳模型
        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        
        # -- Test (分批GPU推理) --
        X_test_np = dataset['X_test']
        y_test_np = dataset['y_test']
        
        self.model.eval()
        with torch.no_grad():
            test_logits = _batch_predict(X_test_np)
            test_probs = torch.sigmoid(test_logits).numpy()
            test_labels = y_test_np
        
        try:
            test_auc = roc_auc_score(test_labels, test_probs)
        except Exception:
            test_auc = 0.5
        
        # Precision@TopK
        test_metrics = {}
        for k in [20, 50, 100]:
            if len(test_probs) > k:
                idx = np.argsort(test_probs)[-k:]
                test_metrics[f'test_p@{k}'] = round(float(test_labels[idx].mean()), 4)
        
        # 区分度
        if len(test_probs) > 100:
            top10_idx = np.argsort(test_probs)[-int(len(test_probs) * 0.1):]
            bot10_idx = np.argsort(test_probs)[:int(len(test_probs) * 0.1)]
            discrimination = float(test_labels[top10_idx].mean() - test_labels[bot10_idx].mean())
        else:
            discrimination = 0
        
        report = {
            'best_val_auc': round(best_val_auc, 4),
            'test_auc': round(test_auc, 4),
            'test_discrimination': round(discrimination, 4),
            **test_metrics,
            'epochs_trained': len(history),
            'history': history,
            'model_params': sum(p.numel() for p in self.model.parameters()),
        }
        
        print(f"\n  最佳验证AUC: {best_val_auc:.4f}")
        print(f"  测试AUC: {test_auc:.4f}")
        print(f"  区分度: {discrimination:.4f}")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        return report
    
    def predict_single(self, df: pd.DataFrame) -> Optional[float]:
        """
        对单只股票预测: 返回 0~100 的分数
        """
        torch, _ = _get_torch()
        
        if df is None or len(df) < self.seq_len:
            return None
        
        df = df.sort_values('date').reset_index(drop=True)
        features = build_daily_features(df)
        
        # 取最后seq_len天
        seq = features[-self.seq_len:]
        if seq.shape[0] != self.seq_len or np.isnan(seq).any():
            return None
        
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
            logit = self.model(X).squeeze()
            prob = torch.sigmoid(logit).item()
        
        return round(prob * 100, 1)
    
    def scan_market(self, cache, pool, top_n: int = 50,
                    progress_callback=None) -> pd.DataFrame:
        """全市场Transformer评分扫描"""
        torch, _ = _get_torch()
        
        tradeable = pool.get_tradeable_stocks()
        cached = cache.get_all_cached_stocks()
        valid_codes = sorted(set(tradeable['stock_code']) & set(cached['stock_code']))
        
        code_info = {}
        for _, row in tradeable.iterrows():
            code_info[row['stock_code']] = {
                'name': row.get('stock_name', ''),
                'board': row.get('board_name', ''),
            }
        
        # 批量预测 (GPU并行)
        results = []
        total = len(valid_codes)
        
        # 收集有效序列
        batch_seqs = []
        batch_codes = []
        batch_closes = []
        
        for i, code in enumerate(valid_codes):
            if progress_callback and (i + 1) % 1000 == 0:
                progress_callback(i + 1, total)
            try:
                df = cache.load_kline(code)
                if df is None or len(df) < self.seq_len + 5:
                    continue
                df = df.sort_values('date').reset_index(drop=True)
                features = build_daily_features(df)
                seq = features[-self.seq_len:]
                if seq.shape[0] != self.seq_len:
                    continue
                if np.isnan(seq).any() or np.abs(seq).max() > 100:
                    continue
                batch_seqs.append(seq)
                batch_codes.append(code)
                batch_closes.append(float(df.iloc[-1]['close']))
            except Exception:
                continue
        
        if not batch_seqs:
            return pd.DataFrame()
        
        # GPU批量推理
        self.model.eval()
        all_probs = []
        batch_size = 4096
        X_all = np.array(batch_seqs, dtype=np.float32)
        
        with torch.no_grad():
            for start in range(0, len(X_all), batch_size):
                end = min(start + batch_size, len(X_all))
                batch = torch.FloatTensor(X_all[start:end]).to(self.device)
                logits = self.model(batch).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.tolist())
        
        # 构建结果
        for idx, (code, prob) in enumerate(zip(batch_codes, all_probs)):
            info = code_info.get(code, {})
            results.append({
                'stock_code': code,
                'stock_name': info.get('name', ''),
                'board_name': info.get('board', ''),
                'close': batch_closes[idx],
                'transformer_score': round(prob * 100, 1),
            })
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('transformer_score', ascending=False).reset_index(drop=True)
        
        return df_results.head(top_n)
    
    def save(self, path: str = None):
        """保存模型"""
        torch, _ = _get_torch()
        if path is None:
            path = os.path.join(DATA_DIR, 'transformer_model.pt')
        torch.save({
            'model_state': self.model.state_dict(),
            'd_input': self.d_input,
            'd_model': self.d_model,
            'seq_len': self.seq_len,
            'save_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        }, path)
        return path
    
    @classmethod
    def load(cls, path: str = None) -> 'StockTransformer':
        """加载模型"""
        torch, _ = _get_torch()
        if path is None:
            path = os.path.join(DATA_DIR, 'transformer_model.pt')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Transformer模型不存在: {path}")
        
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        engine = cls(
            d_input=checkpoint.get('d_input', 12),
            d_model=checkpoint.get('d_model', 64),
            seq_len=checkpoint.get('seq_len', SEQ_LEN),
        )
        engine.model.load_state_dict(checkpoint['model_state'])
        engine.model.to(engine.device)
        return engine


# ============================================================
# Transformer 模型 (PyTorch nn.Module)
# ============================================================

def _build_transformer_model():
    """延迟定义模型类"""
    torch, nn = _get_torch()
    
    class TransformerModel(nn.Module):
        def __init__(self, d_input=12, d_model=64, nhead=4, 
                     num_layers=4, d_ff=256, seq_len=60, dropout=0.1):
            super().__init__()
            self.d_model = d_model
            self.seq_len = seq_len
            
            # 输入投影
            self.input_proj = nn.Linear(d_input, d_model)
            
            # 可学习的位置编码
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
            
            # [CLS] token
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            
            # Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                activation='gelu',
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # LayerNorm
            self.norm = nn.LayerNorm(d_model)
            
            # 分类头
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
        
        def forward(self, x):
            """
            x: (B, seq_len, d_input)
            returns: (B, 1) logits
            """
            B = x.size(0)
            
            # 投影
            x = self.input_proj(x)  # (B, seq_len, d_model)
            
            # 加位置编码
            x = x + self.pos_embed
            
            # 拼接CLS token
            cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
            x = torch.cat([cls, x], dim=1)  # (B, seq_len+1, d_model)
            
            # Transformer Encoder
            x = self.encoder(x)  # (B, seq_len+1, d_model)
            
            # 取CLS token输出
            cls_out = x[:, 0, :]  # (B, d_model)
            cls_out = self.norm(cls_out)
            
            # 分类
            logits = self.head(cls_out)  # (B, 1)
            
            return logits
    
    return TransformerModel


# 全局模型类 (延迟初始化)
_TransformerModel = None

def _ensure_model_class():
    global _TransformerModel
    if _TransformerModel is None:
        _TransformerModel = _build_transformer_model()

# 在模块级别延迟初始化
class _LazyTransformerModel:
    """代理类, 首次使用时才导入torch"""
    _real_class = None
    
    def __new__(cls, *args, **kwargs):
        if cls._real_class is None:
            cls._real_class = _build_transformer_model()
        return cls._real_class(*args, **kwargs)

_TransformerModel = _LazyTransformerModel
