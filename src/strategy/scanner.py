# -*- coding: utf-8 -*-
"""
全市场扫描引擎
遍历股票池全部股票，批量运行策略组合，输出信号结果
"""

import pandas as pd
import sqlite3
import os
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from src.data.data_fetcher import get_history_data
from src.data.stock_pool import StockPool
from src.strategy.strategies import run_all_strategies, STRATEGY_REGISTRY
from src.strategy.ai_scoring import score_stock, compute_price_targets
from src.strategy.strategy_validator import validate_all_strategies, compute_composite_score
from src.strategy.strategy_discovery import apply_learned_rules, load_learned_rules

# 扫描结果数据库
SIGNAL_DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'signals.db')


class MarketScanner:
    """全市场扫描引擎"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = SIGNAL_DB_PATH
        self.db_path = os.path.abspath(db_path)
        self.pool = StockPool()
        self._init_db()

    def _init_db(self):
        """初始化信号记录数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 扫描信号记录
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scan_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date TEXT NOT NULL,
                scan_time TEXT NOT NULL,
                stock_code TEXT NOT NULL,
                stock_name TEXT NOT NULL,
                board_name TEXT DEFAULT '',
                signal_type TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                strength INTEGER DEFAULT 0,
                reason TEXT DEFAULT '',
                close_price REAL DEFAULT 0,
                indicators TEXT DEFAULT '',
                ml_score REAL DEFAULT 0,
                risk_score REAL DEFAULT 0,
                risk_level TEXT DEFAULT '',
                ml_confidence REAL DEFAULT 0,
                composite_score REAL DEFAULT 0,
                confidence_grade TEXT DEFAULT '',
                risk_grade TEXT DEFAULT '',
                buy_price REAL DEFAULT 0,
                target_price REAL DEFAULT 0,
                stop_price REAL DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        ''')

        # 扫描任务记录
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scan_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date TEXT NOT NULL,
                scan_time TEXT NOT NULL,
                total_stocks INTEGER DEFAULT 0,
                scanned_stocks INTEGER DEFAULT 0,
                buy_signals INTEGER DEFAULT 0,
                sell_signals INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0,
                duration_seconds REAL DEFAULT 0,
                status TEXT DEFAULT 'running',
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        ''')

        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_date ON scan_signals(scan_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_code ON scan_signals(stock_code)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_type ON scan_signals(signal_type)')

        conn.commit()
        conn.close()

        # 兼容旧表：补充新增字段
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(scan_signals)")
        existing = [r[1] for r in cursor.fetchall()]
        add_cols = {
            'ml_score': "ALTER TABLE scan_signals ADD COLUMN ml_score REAL DEFAULT 0",
            'risk_score': "ALTER TABLE scan_signals ADD COLUMN risk_score REAL DEFAULT 0",
            'risk_level': "ALTER TABLE scan_signals ADD COLUMN risk_level TEXT DEFAULT ''",
            'ml_confidence': "ALTER TABLE scan_signals ADD COLUMN ml_confidence REAL DEFAULT 0",
            'composite_score': "ALTER TABLE scan_signals ADD COLUMN composite_score REAL DEFAULT 0",
            'confidence_grade': "ALTER TABLE scan_signals ADD COLUMN confidence_grade TEXT DEFAULT ''",
            'risk_grade': "ALTER TABLE scan_signals ADD COLUMN risk_grade TEXT DEFAULT ''",
            'buy_price': "ALTER TABLE scan_signals ADD COLUMN buy_price REAL DEFAULT 0",
            'target_price': "ALTER TABLE scan_signals ADD COLUMN target_price REAL DEFAULT 0",
            'stop_price': "ALTER TABLE scan_signals ADD COLUMN stop_price REAL DEFAULT 0",
        }
        for col, sql in add_cols.items():
            if col not in existing:
                cursor.execute(sql)
        conn.commit()
        conn.close()

    # 北交所股票（92开头）数据源覆盖不全，跳过
    SKIP_PREFIXES = ('92',)

    def _scan_single_stock(self, stock_code: str, stock_name: str, board_name: str,
                           strategy_ids: list = None, days: int = 730) -> list:
        """扫描单只股票（优先使用本地缓存）"""
        signals = []
        if stock_code.startswith(self.SKIP_PREFIXES):
            return signals
        try:
            # use_cache=True：优先从本地读取，仅增量更新缺失数据
            df = get_history_data(stock_code, days=days, use_cache=True)
            if df.empty or len(df) < 30:
                return signals

            results = run_all_strategies(df, strategy_ids=strategy_ids)
            
            # 同时应用从数据中学习到的策略
            try:
                learned = apply_learned_rules(df)
                for lr in learned:
                    lr['strategy_id'] = lr.get('strategy_id', 'ml_learned')
                    lr['strategy'] = lr.get('strategy', 'ML学习策略')
                    results.append(lr)
            except Exception:
                pass
            
            close_price = float(df['close'].iloc[-1]) if not df.empty else 0
            ml = score_stock(df)

            # 策略验证 + 综合评分
            try:
                validations = validate_all_strategies(df, hold_days=config.STRATEGY_HOLD_DAYS)
                composite = compute_composite_score(results, validations)
            except Exception:
                validations = {}
                composite = ml.get('score', 0)

            # 价格目标（仅对买入信号计算）
            prices = compute_price_targets(df, close_price)

            for r in results:
                sid = r.get('strategy_id', '')
                v = validations.get(sid, {})
                conf_grade = v.get('confidence_grade', '')
                risk_grd = v.get('risk_grade', '')

                signals.append({
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'board_name': board_name,
                    'signal_type': r['signal'],
                    'strategy_id': r['strategy_id'],
                    'strategy_name': r['strategy'],
                    'strength': r.get('strength', 0),
                    'reason': r.get('reason', ''),
                    'close_price': close_price,
                    'indicators': str(r.get('indicators', {})),
                    'ml_score': ml.get('score', 0),
                    'risk_score': ml.get('risk_score', 0),
                    'risk_level': ml.get('risk_level', ''),
                    'ml_confidence': ml.get('confidence', 0),
                    'composite_score': composite,
                    'confidence_grade': conf_grade,
                    'risk_grade': risk_grd,
                    'buy_price': prices.get('buy_price', 0) if r['signal'] == 'buy' else 0,
                    'target_price': prices.get('target_price', 0) if r['signal'] == 'buy' else 0,
                    'stop_price': prices.get('stop_price', 0) if r['signal'] == 'buy' else 0,
                })

        except Exception:
            pass  # 静默失败，不中断扫描

        return signals

    def scan_market(self, board_names: list = None, strategy_ids: list = None,
                    signal_filter: str = None, days: int = 730,
                    progress_callback=None, max_workers: int = 2) -> dict:
        """
        全市场扫描

        参数:
            board_names: 指定扫描的板块（None=全部板块）
            strategy_ids: 指定策略（None=全部启用策略）
            signal_filter: 'buy' / 'sell' / None（全部）
            days: 获取多少天历史数据
            progress_callback: 进度回调 callback(scanned, total, current_stock)
            max_workers: 并发线程数

        返回:
            dict: {
                'task_id': 任务ID,
                'buy_signals': [...],
                'sell_signals': [...],
                'stats': {...}
            }
        """
        scan_date = datetime.now().strftime('%Y-%m-%d')
        scan_time = datetime.now().strftime('%H:%M:%S')
        start_time = time.time()

        # 获取股票列表（只使用可交易股票）
        if board_names:
            all_stocks = pd.DataFrame()
            for bn in board_names:
                try:
                    stocks = self.pool.get_tradeable_stocks_by_board(bn)
                except AttributeError:
                    stocks = self.pool.get_stocks_by_board(bn)
                if not stocks.empty:
                    stocks['board_name'] = bn
                    all_stocks = pd.concat([all_stocks, stocks], ignore_index=True)
        else:
            try:
                all_stocks = self.pool.get_tradeable_stocks()
            except AttributeError:
                all_stocks = self.pool.get_all_stocks()
            if 'board_name' not in all_stocks.columns:
                all_stocks['board_name'] = ''

        if all_stocks.empty:
            return {'task_id': None, 'buy_signals': [], 'sell_signals': [], 'stats': {}}

        # 去重
        all_stocks = all_stocks.drop_duplicates(subset=['stock_code'])
        total = len(all_stocks)

        # 创建扫描任务记录
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO scan_tasks (scan_date, scan_time, total_stocks, status)
            VALUES (?, ?, ?, 'running')
        ''', (scan_date, scan_time, total))
        task_id = cursor.lastrowid
        conn.commit()
        conn.close()

        print(f"[扫描] 开始全市场扫描：{total} 只股票")

        buy_signals = []
        sell_signals = []
        error_count = 0
        scanned = 0

        # 使用线程池并发扫描
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for _, row in all_stocks.iterrows():
                future = executor.submit(
                    self._scan_single_stock,
                    row['stock_code'], row['stock_name'],
                    row.get('board_name', ''), strategy_ids, days
                )
                futures[future] = row

            for future in as_completed(futures):
                scanned += 1
                row = futures[future]
                try:
                    signals = future.result()
                    for sig in signals:
                        if signal_filter and sig['signal_type'] != signal_filter:
                            continue
                        if sig['signal_type'] == 'buy':
                            buy_signals.append(sig)
                        else:
                            sell_signals.append(sig)
                except Exception:
                    error_count += 1

                if progress_callback and scanned % 10 == 0:
                    progress_callback(scanned, total, row['stock_name'])

        duration = time.time() - start_time

        # 按综合评分排序（优先），其次信号强度
        buy_signals.sort(key=lambda x: (x.get('composite_score', 0), x['strength']), reverse=True)
        sell_signals.sort(key=lambda x: (x.get('composite_score', 0), x['strength']), reverse=True)

        # 保存信号到数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        all_sigs = buy_signals + sell_signals
        for sig in all_sigs:
            cursor.execute('''
                INSERT INTO scan_signals
                (scan_date, scan_time, stock_code, stock_name, board_name,
                 signal_type, strategy_id, strategy_name, strength, reason, close_price, indicators,
                 ml_score, risk_score, risk_level, ml_confidence,
                 composite_score, confidence_grade, risk_grade, buy_price, target_price, stop_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                scan_date, scan_time, sig['stock_code'], sig['stock_name'], sig['board_name'],
                sig['signal_type'], sig['strategy_id'], sig['strategy_name'],
                sig['strength'], sig['reason'], sig['close_price'], sig['indicators'],
                sig.get('ml_score', 0), sig.get('risk_score', 0),
                sig.get('risk_level', ''), sig.get('ml_confidence', 0),
                sig.get('composite_score', 0), sig.get('confidence_grade', ''),
                sig.get('risk_grade', ''), sig.get('buy_price', 0),
                sig.get('target_price', 0), sig.get('stop_price', 0)
            ))

        # 更新任务状态
        cursor.execute('''
            UPDATE scan_tasks SET
                scanned_stocks=?, buy_signals=?, sell_signals=?, errors=?,
                duration_seconds=?, status='done'
            WHERE id=?
        ''', (scanned, len(buy_signals), len(sell_signals), error_count, round(duration, 1), task_id))

        conn.commit()
        conn.close()

        # 聚合推荐（多策略共振）
        buy_recs = self.aggregate_recommendations(buy_signals, min_strategies=2)
        sell_recs = self.aggregate_recommendations(sell_signals, min_strategies=2)

        print(f"[扫描] 完成！耗时 {duration:.1f}s，"
              f"原始买入信号 {len(buy_signals)} 个，卖出信号 {len(sell_signals)} 个，"
              f"多策略共振推荐 买入 {len(buy_recs)} 只 卖出 {len(sell_recs)} 只")

        return {
            'task_id': task_id,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'buy_recommendations': buy_recs,
            'sell_recommendations': sell_recs,
            'stats': {
                'total': total,
                'scanned': scanned,
                'buy_count': len(buy_signals),
                'sell_count': len(sell_signals),
                'buy_rec_count': len(buy_recs),
                'sell_rec_count': len(sell_recs),
                'errors': error_count,
                'duration': round(duration, 1),
                'scan_date': scan_date,
                'scan_time': scan_time,
            }
        }

    def get_today_signals(self, signal_type: str = None, min_strength: int = 0) -> pd.DataFrame:
        """获取今日扫描信号"""
        today = datetime.now().strftime('%Y-%m-%d')
        return self.get_signals_by_date(today, signal_type, min_strength)

    def get_signals_by_date(self, date: str, signal_type: str = None,
                            min_strength: int = 0) -> pd.DataFrame:
        """获取指定日期的扫描信号"""
        conn = sqlite3.connect(self.db_path)

        query = 'SELECT * FROM scan_signals WHERE scan_date = ?'
        params = [date]

        if signal_type:
            query += ' AND signal_type = ?'
            params.append(signal_type)

        if min_strength > 0:
            query += ' AND strength >= ?'
            params.append(min_strength)

        query += ' ORDER BY strength DESC, scan_time DESC'

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def get_latest_scan_task(self) -> dict:
        """获取最近一次扫描任务信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM scan_tasks ORDER BY id DESC LIMIT 1')
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        columns = ['id', 'scan_date', 'scan_time', 'total_stocks', 'scanned_stocks',
                    'buy_signals', 'sell_signals', 'errors', 'duration_seconds', 'status', 'created_at']
        return dict(zip(columns, row))

    def get_scan_history(self, limit: int = 10) -> pd.DataFrame:
        """获取扫描历史"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            'SELECT * FROM scan_tasks ORDER BY id DESC LIMIT ?',
            conn, params=[limit]
        )
        conn.close()
        return df

    def warmup_cache(self, board_names: list = None, days: int = 730,
                     progress_callback=None) -> dict:
        """
        预热缓存：批量下载所有股票的历史数据到本地
        下载后扫描就不再需要网络请求，速度极快

        参数:
            board_names: 板块筛选（None=全部）
            days: 下载多少天
            progress_callback: callback(current, total, stock_name, status)

        返回:
            dict: {total, success, cached, failed}
        """
        # 只预热可交易股票
        if board_names:
            all_stocks = pd.DataFrame()
            for bn in board_names:
                try:
                    stocks = self.pool.get_tradeable_stocks_by_board(bn)
                except AttributeError:
                    stocks = self.pool.get_stocks_by_board(bn)
                if not stocks.empty:
                    stocks['board_name'] = bn
                    all_stocks = pd.concat([all_stocks, stocks], ignore_index=True)
        else:
            try:
                all_stocks = self.pool.get_tradeable_stocks()
            except AttributeError:
                all_stocks = self.pool.get_all_stocks()

        all_stocks = all_stocks.drop_duplicates(subset=['stock_code'])
        total = len(all_stocks)
        success = 0
        cached = 0
        failed = 0

        print(f"[预热] 开始预热缓存：{total} 只股票")

        for idx, row in all_stocks.iterrows():
            code = row['stock_code']
            name = row['stock_name']

            # 跳过北交所（数据源覆盖不全）
            if code.startswith(self.SKIP_PREFIXES):
                success += 1  # 计入成功，不影响统计
                count = success + cached + failed
                if progress_callback and count % 20 == 0:
                    progress_callback(count, total, name, 'skip')
                continue

            try:
                # get_history_data 会自动处理缓存逻辑
                df = get_history_data(code, days=days, use_cache=True)
                if not df.empty:
                    success += 1
                    status = 'ok'
                else:
                    failed += 1
                    status = 'empty'
            except Exception:
                failed += 1
                status = 'error'

            count = success + cached + failed
            if progress_callback and count % 20 == 0:
                progress_callback(count, total, name, status)

            if count % 100 == 0:
                print(f"  [{count}/{total}] 成功={success} 失败={failed}")

        print(f"[预热] 完成！成功={success} 失败={failed} / 总共={total}")
        return {'total': total, 'success': success, 'cached': cached, 'failed': failed}

    @staticmethod
    def aggregate_recommendations(raw_signals: list, min_strategies: int = 2,
                                  min_validated_grade: str = 'C') -> list:
        """
        将原始信号聚合为「每只股票一条推荐」

        核心逻辑：
        1. 按 (stock_code, signal_type) 分组
        2. 只保留 2+ 策略共振（同方向）的股票
        3. 排除可信度全为 C 的股票（至少一个 A 或 B）
        4. 按综合评分排序

        参数:
            raw_signals: _scan_single_stock 返回的原始信号列表
            min_strategies: 最少策略共振数（默认 2）
            min_validated_grade: 至少有一个策略达到此等级（A > B > C）

        返回:
            list[dict]: 聚合后的推荐列表，每只股票一条
        """
        from collections import defaultdict

        grade_rank = {'A': 3, 'B': 2, 'C': 1, '': 0}
        min_grade_val = grade_rank.get(min_validated_grade, 1)

        # 按 (stock_code, signal_type) 分组
        groups = defaultdict(list)
        for sig in raw_signals:
            key = (sig['stock_code'], sig['signal_type'])
            groups[key].append(sig)

        recommendations = []
        for (code, sig_type), sigs in groups.items():
            # 条件1: 至少 min_strategies 个策略共振
            if len(sigs) < min_strategies:
                continue

            # 条件2: 至少一个策略的可信度达标
            best_grade_val = max(grade_rank.get(s.get('confidence_grade', ''), 0) for s in sigs)
            if best_grade_val < min_grade_val:
                continue

            # 聚合信息
            first = sigs[0]
            strategy_names = []
            strategy_grades = []
            total_strength = 0
            validated_count = 0  # A或B等级策略数

            for s in sigs:
                g = s.get('confidence_grade', 'C')
                strategy_names.append(s['strategy_name'])
                strategy_grades.append(f"{s['strategy_name']}({g})")
                total_strength += s.get('strength', 0)
                if g in ('A', 'B'):
                    validated_count += 1

            avg_strength = total_strength / len(sigs)
            composite = first.get('composite_score', 0)

            # 多策略共振加成：每多一个策略 +8 分，每多一个 A/B 策略额外 +5
            resonance_bonus = (len(sigs) - 1) * 8 + validated_count * 5
            final_score = min(100, composite + resonance_bonus)

            # 汇总原因
            reasons = [s.get('reason', '') for s in sigs if s.get('reason')]
            combined_reason = ' | '.join(reasons[:3])
            if len(reasons) > 3:
                combined_reason += f" 等{len(reasons)}条"

            rec = {
                'stock_code': code,
                'stock_name': first.get('stock_name', ''),
                'board_name': first.get('board_name', ''),
                'signal_type': sig_type,
                'strategy_count': len(sigs),
                'validated_count': validated_count,
                'strategies': ' + '.join(strategy_names),
                'strategy_detail': ', '.join(strategy_grades),
                'composite_score': round(final_score, 1),
                'avg_strength': round(avg_strength),
                'close_price': first.get('close_price', 0),
                'buy_price': first.get('buy_price', 0),
                'target_price': first.get('target_price', 0),
                'stop_price': first.get('stop_price', 0),
                'ml_score': first.get('ml_score', 0),
                'risk_score': first.get('risk_score', 0),
                'risk_level': first.get('risk_level', ''),
                'reason': combined_reason,
                # 保留原始信号供详情页使用
                '_raw_signals': sigs,
            }
            recommendations.append(rec)

        # 按综合评分降序
        recommendations.sort(key=lambda x: x['composite_score'], reverse=True)
        return recommendations

    def get_stock_signal_detail(self, stock_code: str, date: str = None) -> pd.DataFrame:
        """获取单只股票的信号详情"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            'SELECT * FROM scan_signals WHERE stock_code = ? AND scan_date = ? ORDER BY strength DESC',
            conn, params=[stock_code, date]
        )
        conn.close()
        return df
