# -*- coding: utf-8 -*-
"""
股票池模块
使用申万行业分类（一级）+ AkShare 获取A股全量股票并按行业板块分类
数据缓存到本地 SQLite
"""

import akshare as ak
import pandas as pd
import sqlite3
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 数据库路径
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'stock_pool.db')


class StockPool:
    """A股股票池管理（按申万行业板块分类）"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = DB_PATH
        self.db_path = os.path.abspath(db_path)
        self._init_db()

    def _init_db(self):
        """初始化数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 行业板块列表（申万一级行业）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS industry_boards (
                board_code TEXT PRIMARY KEY,
                board_name TEXT NOT NULL,
                stock_count INTEGER DEFAULT 0,
                updated_at TEXT
            )
        ''')

        # 股票与板块的关联
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS board_stocks (
                board_code TEXT NOT NULL,
                board_name TEXT NOT NULL,
                stock_code TEXT NOT NULL,
                stock_name TEXT NOT NULL,
                PRIMARY KEY (board_code, stock_code)
            )
        ''')

        # 全量股票名称表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS all_stocks (
                stock_code TEXT PRIMARY KEY,
                stock_name TEXT NOT NULL,
                board_name TEXT DEFAULT '',
                tradeable INTEGER DEFAULT 1,
                exclude_reason TEXT DEFAULT ''
            )
        ''')

        # 兼容旧表：如果tradeable列不存在则添加
        try:
            cursor.execute('SELECT tradeable FROM all_stocks LIMIT 1')
        except sqlite3.OperationalError:
            cursor.execute('ALTER TABLE all_stocks ADD COLUMN tradeable INTEGER DEFAULT 1')
            cursor.execute("ALTER TABLE all_stocks ADD COLUMN exclude_reason TEXT DEFAULT ''")

        # 元信息
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pool_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        ''')

        conn.commit()
        conn.close()

    @staticmethod
    def _clear_proxy():
        """清除可能干扰请求的代理环境变量"""
        for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                     'http_proxy', 'https_proxy', 'all_proxy']:
            os.environ.pop(key, None)

    def _fetch_with_retry(self, func, retries=3, **kwargs):
        """带重试的 API 调用"""
        for attempt in range(retries):
            try:
                return func(**kwargs)
            except Exception as e:
                if attempt < retries - 1:
                    wait = 2 * (attempt + 1)
                    print(f"    重试 {attempt+1}/{retries}（{wait}s后）: {e}")
                    time.sleep(wait)
                else:
                    raise

    def update_industry_boards(self, progress_callback=None):
        """
        从 AkShare 更新全部申万行业板块及其成分股
        每个板块同步后立即提交数据库

        参数:
            progress_callback: 进度回调函数 callback(current, total, board_name)
        """
        self._clear_proxy()

        # ===== 第1步：获取申万一级行业列表 =====
        try:
            print("[股票池] 正在获取申万一级行业列表...")
            sw_df = self._fetch_with_retry(ak.sw_index_first_info)
            total = len(sw_df)
            print(f"[股票池] 获取到 {total} 个申万一级行业，开始同步成分股...")
        except Exception as e:
            print(f"[股票池] 获取行业列表失败: {e}")
            return

        # ===== 第2步：逐个行业获取成分股 =====
        all_stock_count = 0
        success_count = 0
        fail_count = 0

        for idx, row in sw_df.iterrows():
            board_code = row['行业代码']
            board_name = row['行业名称']
            expected_count = int(row['成份个数'])

            if progress_callback:
                progress_callback(idx + 1, total, board_name)

            try:
                # 获取该行业的成分股（用申万代码去掉 .SI 后缀）
                code_clean = board_code.replace('.SI', '')
                stocks_df = self._fetch_with_retry(ak.index_component_sw, symbol=code_clean)
                stock_count = len(stocks_df)
                all_stock_count += stock_count

                # 写入数据库（每个板块独立事务）
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT OR REPLACE INTO industry_boards (board_code, board_name, stock_count, updated_at)
                    VALUES (?, ?, ?, ?)
                ''', (board_code, board_name, stock_count, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

                cursor.execute('DELETE FROM board_stocks WHERE board_code = ?', (board_code,))

                for _, stock_row in stocks_df.iterrows():
                    s_code = str(stock_row['证券代码']).zfill(6)
                    s_name = stock_row['证券名称']

                    cursor.execute('''
                        INSERT OR REPLACE INTO board_stocks (board_code, board_name, stock_code, stock_name)
                        VALUES (?, ?, ?, ?)
                    ''', (board_code, board_name, s_code, s_name))

                    cursor.execute('''
                        INSERT OR REPLACE INTO all_stocks (stock_code, stock_name, board_name)
                        VALUES (?, ?, ?)
                    ''', (s_code, s_name, board_name))

                cursor.execute('''
                    INSERT OR REPLACE INTO pool_meta (key, value) VALUES (?, ?)
                ''', ('last_update', datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

                conn.commit()
                conn.close()

                success_count += 1
                print(f"  [{idx+1}/{total}] {board_name}: {stock_count}/{expected_count} 只股票 [已保存]")

            except Exception as e:
                fail_count += 1
                print(f"  [{idx+1}/{total}] {board_name}: 获取失败 - {e}")
                time.sleep(1)
                continue

        print(f"\n[股票池] 同步完成！成功 {success_count}/{total} 个行业，失败 {fail_count} 个，共 {all_stock_count} 只股票")

        # ===== 第3步：自动标记不可交易股票 =====
        marked = self.mark_tradeable_status()
        print(f"[股票池] 可交易标记完成！排除 {marked['excluded']} 只，可交易 {marked['tradeable']} 只")

    def get_last_update_time(self) -> str:
        """获取最后更新时间"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM pool_meta WHERE key='last_update'")
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else "从未更新"

    def get_industry_boards(self) -> pd.DataFrame:
        """获取所有行业板块列表"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            'SELECT board_code, board_name, stock_count FROM industry_boards ORDER BY board_name',
            conn
        )
        conn.close()
        return df

    def get_stocks_by_board(self, board_name: str) -> pd.DataFrame:
        """获取某个板块下的所有股票"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            'SELECT stock_code, stock_name FROM board_stocks WHERE board_name = ? ORDER BY stock_code',
            conn, params=[board_name]
        )
        conn.close()
        return df

    def get_all_stocks(self) -> pd.DataFrame:
        """获取全部股票（去重）"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            'SELECT stock_code, stock_name, board_name FROM all_stocks ORDER BY stock_code',
            conn
        )
        conn.close()
        return df

    def search_stock(self, keyword: str) -> pd.DataFrame:
        """按代码或名称搜索股票"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            'SELECT stock_code, stock_name, board_name FROM all_stocks WHERE stock_code LIKE ? OR stock_name LIKE ? LIMIT 50',
            conn, params=[f'%{keyword}%', f'%{keyword}%']
        )
        conn.close()
        return df

    def get_stock_name(self, stock_code: str) -> str:
        """根据代码获取股票名称"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT stock_name FROM all_stocks WHERE stock_code = ?', (stock_code,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else stock_code

    def is_empty(self) -> bool:
        """数据库是否为空"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM industry_boards')
        count = cursor.fetchone()[0]
        conn.close()
        return count == 0

    def get_stats(self) -> dict:
        """获取统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM industry_boards')
        board_count = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM all_stocks')
        stock_count = cursor.fetchone()[0]
        try:
            cursor.execute('SELECT COUNT(*) FROM all_stocks WHERE tradeable = 1')
            tradeable_count = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            tradeable_count = stock_count
        conn.close()
        return {
            'board_count': board_count,
            'stock_count': stock_count,
            'tradeable_count': tradeable_count,
            'last_update': self.get_last_update_time(),
        }

    # ============================================================
    # 可交易股票过滤
    # ============================================================
    def mark_tradeable_status(self) -> dict:
        """
        标记所有股票的可交易状态
        排除规则：
          1. ST/*ST股 — 风险警示，波动规则不同（涨跌停±5%）
          2. 退市整理股 — 即将退市
          3. B股 — 200xxx（深B）/ 900xxx（沪B），外币计价
          4. 北交所股票 — 920xxx，门槛50万，流动性差
          5. 名称含"退"的股票

        返回: {'total': int, 'tradeable': int, 'excluded': int, 'details': dict}
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 先把所有股票标记为可交易
        cursor.execute("UPDATE all_stocks SET tradeable = 1, exclude_reason = ''")

        # 规则1: ST/*ST股
        cursor.execute("""
            UPDATE all_stocks SET tradeable = 0, exclude_reason = 'ST风险警示'
            WHERE stock_name LIKE '%ST%'
        """)
        st_count = cursor.rowcount

        # 规则2: 退市整理
        cursor.execute("""
            UPDATE all_stocks SET tradeable = 0, exclude_reason = '退市整理'
            WHERE stock_name LIKE '%退%' AND tradeable = 1
        """)
        delist_count = cursor.rowcount

        # 规则3: B股 (200xxx深B, 900xxx沪B)
        cursor.execute("""
            UPDATE all_stocks SET tradeable = 0, exclude_reason = 'B股(外币)'
            WHERE (stock_code LIKE '2%' OR (stock_code LIKE '9%' AND stock_code < '920000'))
            AND tradeable = 1
        """)
        b_count = cursor.rowcount

        # 规则4: 北交所 (920xxx) - 门槛高，流动性差
        cursor.execute("""
            UPDATE all_stocks SET tradeable = 0, exclude_reason = '北交所(门槛高)'
            WHERE stock_code LIKE '92%' AND tradeable = 1
        """)
        bj_count = cursor.rowcount

        conn.commit()

        # 统计
        cursor.execute('SELECT COUNT(*) FROM all_stocks')
        total = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM all_stocks WHERE tradeable = 1')
        tradeable = cursor.fetchone()[0]
        conn.close()

        excluded = total - tradeable
        return {
            'total': total,
            'tradeable': tradeable,
            'excluded': excluded,
            'details': {
                'ST风险警示': st_count,
                '退市整理': delist_count,
                'B股': b_count,
                '北交所': bj_count,
            }
        }

    def get_tradeable_stocks(self) -> pd.DataFrame:
        """获取所有可交易的股票"""
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(
                "SELECT stock_code, stock_name, board_name FROM all_stocks WHERE tradeable = 1 ORDER BY stock_code",
                conn
            )
        except sqlite3.OperationalError:
            # 如果tradeable列不存在，返回所有
            df = pd.read_sql_query(
                "SELECT stock_code, stock_name, board_name FROM all_stocks ORDER BY stock_code",
                conn
            )
        conn.close()
        return df

    def get_excluded_stocks(self) -> pd.DataFrame:
        """获取被排除的股票及排除原因"""
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(
                "SELECT stock_code, stock_name, board_name, exclude_reason FROM all_stocks WHERE tradeable = 0 ORDER BY exclude_reason, stock_code",
                conn
            )
        except sqlite3.OperationalError:
            df = pd.DataFrame()
        conn.close()
        return df

    def get_tradeable_stocks_by_board(self, board_name: str) -> pd.DataFrame:
        """获取某个板块下的可交易股票"""
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(
                """SELECT bs.stock_code, bs.stock_name FROM board_stocks bs
                   JOIN all_stocks a ON bs.stock_code = a.stock_code
                   WHERE bs.board_name = ? AND a.tradeable = 1
                   ORDER BY bs.stock_code""",
                conn, params=[board_name]
            )
        except sqlite3.OperationalError:
            df = pd.read_sql_query(
                "SELECT stock_code, stock_name FROM board_stocks WHERE board_name = ? ORDER BY stock_code",
                conn, params=[board_name]
            )
        conn.close()
        return df

    def get_tradeable_stats(self) -> dict:
        """获取可交易统计详情"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT exclude_reason, COUNT(*) FROM all_stocks WHERE tradeable = 0 GROUP BY exclude_reason")
            excluded_detail = {row[0]: row[1] for row in cursor.fetchall()}
            cursor.execute("SELECT COUNT(*) FROM all_stocks WHERE tradeable = 1")
            tradeable = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM all_stocks")
            total = cursor.fetchone()[0]

            # 按市场板块统计可交易
            cursor.execute("""
                SELECT
                    CASE
                        WHEN stock_code LIKE '60%' OR stock_code LIKE '68%' THEN '沪市(主板+科创)'
                        WHEN stock_code LIKE '00%' OR stock_code LIKE '30%' THEN '深市(主板+创业)'
                        ELSE '其他'
                    END as market,
                    COUNT(*) as cnt
                FROM all_stocks WHERE tradeable = 1
                GROUP BY market
            """)
            market_dist = {row[0]: row[1] for row in cursor.fetchall()}
        except sqlite3.OperationalError:
            excluded_detail = {}
            tradeable = 0
            total = 0
            market_dist = {}
        conn.close()

        return {
            'total': total,
            'tradeable': tradeable,
            'excluded': total - tradeable,
            'excluded_detail': excluded_detail,
            'market_distribution': market_dist,
        }
