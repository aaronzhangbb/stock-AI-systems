"""
数据本地缓存模块
将历史行情数据缓存到本地 SQLite 数据库，避免重复请求 API

缓存数据库位置: data/stock_cache.db
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config

# 缓存数据库路径（项目根目录/data/stock_cache.db）
CACHE_DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'stock_cache.db')


class DataCache:
    """股票数据本地缓存管理"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = CACHE_DB_PATH
        self.db_path = os.path.abspath(db_path)
        self._init_db()

    def _init_db(self):
        """初始化缓存数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 日线数据缓存表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_kline (
                stock_code TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                amount REAL,
                amplitude REAL,
                pctChg REAL,
                change_val REAL,
                turnover REAL,
                PRIMARY KEY (stock_code, date)
            )
        ''')

        # 缓存元信息表（记录每只股票最后更新时间）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_meta (
                stock_code TEXT PRIMARY KEY,
                first_date TEXT NOT NULL,
                last_date TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')

        # 股票名称缓存
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_names (
                stock_code TEXT PRIMARY KEY,
                stock_name TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')

        conn.commit()
        conn.close()

    def save_kline(self, stock_code: str, df: pd.DataFrame):
        """
        保存K线数据到本地缓存

        参数:
            stock_code: 股票代码
            df: 包含日线数据的 DataFrame
        """
        if df.empty:
            return

        conn = sqlite3.connect(self.db_path)
        try:
            for _, row in df.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], (datetime, pd.Timestamp)) else str(row['date'])
                conn.execute('''
                    INSERT OR REPLACE INTO daily_kline 
                    (stock_code, date, open, high, low, close, volume, amount, amplitude, pctChg, change_val, turnover)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    stock_code, date_str,
                    float(row.get('open', 0)) if pd.notna(row.get('open')) else None,
                    float(row.get('high', 0)) if pd.notna(row.get('high')) else None,
                    float(row.get('low', 0)) if pd.notna(row.get('low')) else None,
                    float(row.get('close', 0)) if pd.notna(row.get('close')) else None,
                    float(row.get('volume', 0)) if pd.notna(row.get('volume')) else None,
                    float(row.get('amount', 0)) if pd.notna(row.get('amount')) else None,
                    float(row.get('amplitude', 0)) if pd.notna(row.get('amplitude')) else None,
                    float(row.get('pctChg', 0)) if pd.notna(row.get('pctChg')) else None,
                    float(row.get('change', 0)) if pd.notna(row.get('change')) else None,
                    float(row.get('turnover', 0)) if pd.notna(row.get('turnover')) else None,
                ))

            # 更新缓存元信息
            first_date = df['date'].min()
            last_date = df['date'].max()
            if isinstance(first_date, (datetime, pd.Timestamp)):
                first_date = first_date.strftime('%Y-%m-%d')
            if isinstance(last_date, (datetime, pd.Timestamp)):
                last_date = last_date.strftime('%Y-%m-%d')

            # 合并已有的日期范围
            existing = self.get_cache_info(stock_code)
            if existing:
                if existing['first_date'] < str(first_date):
                    first_date = existing['first_date']
                if existing['last_date'] > str(last_date):
                    last_date = existing['last_date']

            conn.execute('''
                INSERT OR REPLACE INTO cache_meta (stock_code, first_date, last_date, updated_at)
                VALUES (?, ?, ?, ?)
            ''', (stock_code, str(first_date), str(last_date), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

            conn.commit()
            print(f"[缓存] 已保存 {stock_code} 的 {len(df)} 条数据到本地")
        except Exception as e:
            print(f"[缓存] 保存失败: {e}")
            conn.rollback()
        finally:
            conn.close()

    def load_kline(self, stock_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        从本地缓存加载K线数据

        参数:
            stock_code: 股票代码
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'

        返回:
            DataFrame: 缓存的K线数据，如果无缓存返回空 DataFrame
        """
        conn = sqlite3.connect(self.db_path)
        try:
            query = 'SELECT * FROM daily_kline WHERE stock_code = ?'
            params = [stock_code]

            if start_date:
                query += ' AND date >= ?'
                params.append(start_date)
            if end_date:
                query += ' AND date <= ?'
                params.append(end_date)

            query += ' ORDER BY date ASC'

            df = pd.read_sql_query(query, conn, params=params)

            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                # 重命名 change_val 回 change
                if 'change_val' in df.columns:
                    df = df.rename(columns={'change_val': 'change'})
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except Exception as e:
            print(f"[缓存] 加载失败: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def get_cache_info(self, stock_code: str) -> dict:
        """获取某只股票的缓存信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT first_date, last_date, updated_at FROM cache_meta WHERE stock_code = ?', (stock_code,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {'first_date': row[0], 'last_date': row[1], 'updated_at': row[2]}
        return None

    def is_cache_fresh(self, stock_code: str) -> bool:
        """
        判断缓存是否是最新的（今天已更新过）
        """
        info = self.get_cache_info(stock_code)
        if info is None:
            return False
        today = datetime.now().strftime('%Y-%m-%d')
        return info['updated_at'].startswith(today)

    def save_stock_name(self, stock_code: str, stock_name: str):
        """缓存股票名称"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT OR REPLACE INTO stock_names (stock_code, stock_name, updated_at)
            VALUES (?, ?, ?)
        ''', (stock_code, stock_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()

    def load_stock_name(self, stock_code: str) -> str:
        """加载缓存的股票名称"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT stock_name FROM stock_names WHERE stock_code = ?', (stock_code,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def get_all_cached_stocks(self) -> pd.DataFrame:
        """获取所有已缓存的股票列表"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT m.stock_code, n.stock_name, m.first_date, m.last_date, m.updated_at,
                   (SELECT COUNT(*) FROM daily_kline k WHERE k.stock_code = m.stock_code) as total_records
            FROM cache_meta m
            LEFT JOIN stock_names n ON m.stock_code = n.stock_code
            ORDER BY m.updated_at DESC
        ''', conn)
        conn.close()
        return df

    def clear_cache(self, stock_code: str = None):
        """清除缓存"""
        conn = sqlite3.connect(self.db_path)
        if stock_code:
            conn.execute('DELETE FROM daily_kline WHERE stock_code = ?', (stock_code,))
            conn.execute('DELETE FROM cache_meta WHERE stock_code = ?', (stock_code,))
        else:
            conn.execute('DELETE FROM daily_kline')
            conn.execute('DELETE FROM cache_meta')
        conn.commit()
        conn.close()
