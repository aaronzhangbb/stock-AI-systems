# -*- coding: utf-8 -*-
"""
AI自动交易引擎

核心逻辑:
    1. 读取AI扫描结果 (ai_daily_scores.json)
    2. 先卖: 检查现有持仓, 触发止损/止盈/追踪/超期 → 自动卖出
    3. 后买: 筛选符合条件的新标的 → 自动买入
    4. 记录所有决策到 auto_trade_log 表

原则:
    - 价格为王: 所有卖出由价格触发, 不因时间直接强卖
    - 严格遵守AI策略: 买入不超过buy_upper, 仓位用Kelly
    - 先卖后买: 释放资金后再考虑新标的
"""

import os
import sys
import json
import sqlite3
import logging
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from src.trading.paper_trading import PaperTradingAccount
from src.trading.position_monitor import check_single_position
from src.data.data_fetcher import get_realtime_price, batch_get_realtime_prices

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


class AutoTrader:
    """AI自动交易引擎"""

    def __init__(self, account: PaperTradingAccount = None):
        self.account = account or PaperTradingAccount()
        self.db_path = self.account.db_path
        self._init_log_table()

    def _init_log_table(self):
        """创建自动交易日志表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS auto_trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date TEXT NOT NULL,
                stock_code TEXT NOT NULL,
                stock_name TEXT DEFAULT '',
                action TEXT NOT NULL,
                price REAL NOT NULL,
                shares INTEGER NOT NULL,
                amount REAL DEFAULT 0,
                reason TEXT DEFAULT '',
                ai_score REAL DEFAULT 0,
                stop_price REAL DEFAULT 0,
                target_price REAL DEFAULT 0,
                pnl REAL DEFAULT 0,
                pnl_pct REAL DEFAULT 0,
                advice_json TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def _log_trade(self, action: str, stock_code: str, stock_name: str,
                   price: float, shares: int, reason: str,
                   ai_score: float = 0, stop_price: float = 0,
                   target_price: float = 0, pnl: float = 0,
                   pnl_pct: float = 0, advice_json: str = ''):
        """记录自动交易日志"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        today = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO auto_trade_log '
            '(trade_date, stock_code, stock_name, action, price, shares, amount, '
            'reason, ai_score, stop_price, target_price, pnl, pnl_pct, advice_json, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (today, stock_code, stock_name, action, price, shares,
             round(price * shares, 2), reason, ai_score, stop_price,
             target_price, pnl, pnl_pct, advice_json, now)
        )
        conn.commit()
        conn.close()

    def _load_ai_recommendations(self) -> list:
        """加载AI扫描推荐结果"""
        score_path = os.path.join(DATA_DIR, 'ai_daily_scores.json')
        if not os.path.exists(score_path):
            logger.warning("AI扫描结果文件不存在: %s", score_path)
            return []
        try:
            with open(score_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data.get('top50', [])
            elif isinstance(data, list):
                return data[:50]
            return []
        except Exception as e:
            logger.error("读取AI扫描结果失败: %s", e)
            return []

    def _get_current_prices(self, codes: list) -> dict:
        """批量获取当前价格"""
        prices = {}
        try:
            rt_map = batch_get_realtime_prices(codes)
            for code, info in rt_map.items():
                if info.get('close', 0) > 0:
                    prices[code] = info['close']
                elif info.get('price', 0) > 0:
                    prices[code] = info['price']
        except Exception as e:
            logger.warning("批量获取价格失败: %s, 将逐只获取", e)
            for code in codes:
                try:
                    rt = get_realtime_price(code)
                    if rt and rt.get('close', 0) > 0:
                        prices[code] = rt['close']
                except Exception:
                    pass
        return prices

    def execute(self, rescan: bool = False, progress_callback=None) -> dict:
        """
        执行一轮完整的自动交易决策

        参数:
            rescan: 是否在卖出后重新运行AI扫描再买入 (默认False用已有推荐)
            progress_callback: 进度回调 fn(stage: str, message: str)
                stage: 'sell' / 'scan' / 'buy' / 'snapshot' / 'done'

        返回:
            {
                'sell_actions': [卖出操作列表],
                'buy_actions': [买入操作列表],
                'skipped': [跳过的候选],
                'scan_result': {扫描结果摘要} or None,
                'summary': '执行摘要文字',
                'timestamp': '执行时间',
            }
        """
        if not config.AUTO_ENABLED:
            return {
                'sell_actions': [], 'buy_actions': [], 'skipped': [],
                'scan_result': None,
                'summary': '自动交易已关闭 (AUTO_ENABLED=False)',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }

        def _progress(stage, msg):
            if progress_callback:
                progress_callback(stage, msg)

        logger.info("=" * 50)
        logger.info("AI自动交易引擎启动 (rescan=%s)", rescan)
        logger.info("=" * 50)

        sell_actions = []
        buy_actions = []
        skipped = []
        scan_result = None

        # ========== 第一步: 检查持仓, 决定卖出 ==========
        _progress('sell', '正在检查持仓，执行止盈止损卖出...')
        sell_actions = self._execute_sells()
        sold_codes = {a['stock_code'] for a in sell_actions}
        _progress('sell', f'卖出完成: {len(sell_actions)}只')

        # ========== 第二步: 重新AI扫描 (可选) ==========
        if rescan:
            _progress('scan', '正在运行AI策略扫描，生成最新推荐 (耗时3~10分钟)...')
            scan_result = self._refresh_ai_scores()
            n_scored = scan_result.get('total_scored', 0) if scan_result else 0
            _progress('scan', f'扫描完成: 评估{n_scored}只股票')

        # ========== 第三步: 筛选新标的, 决定买入 ==========
        _progress('buy', '正在筛选标的，执行买入...')
        buy_actions, skipped = self._execute_buys(exclude_codes=sold_codes)
        _progress('buy', f'买入完成: {len(buy_actions)}只')

        # ========== 第四步: 保存每日资产快照 ==========
        _progress('snapshot', '保存资产快照...')
        self._save_daily_snapshot()

        # ========== 汇总 ==========
        n_sell = len(sell_actions)
        n_buy = len(buy_actions)
        total_sell_pnl = sum(a.get('pnl', 0) for a in sell_actions)

        summary_parts = []
        if n_sell > 0:
            summary_parts.append(f"卖出{n_sell}只(盈亏¥{total_sell_pnl:+,.0f})")
        if rescan:
            summary_parts.append("已重新扫描")
        if n_buy > 0:
            summary_parts.append(f"买入{n_buy}只")
        if not summary_parts:
            summary_parts.append("无操作")

        summary = f"AI自动交易完成: {', '.join(summary_parts)}"
        logger.info(summary)
        _progress('done', summary)

        return {
            'sell_actions': sell_actions,
            'buy_actions': buy_actions,
            'skipped': skipped,
            'scan_result': scan_result,
            'summary': summary,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    def _refresh_ai_scores(self) -> dict:
        """重新运行AI扫描，生成最新推荐 (更新 ai_daily_scores.json)"""
        try:
            from daily_job import run_ai_super_scan
            result = run_ai_super_scan()
            logger.info("[扫描] AI扫描完成, %d只操作推荐", len(result.get('action_list', [])))
            return result
        except Exception as e:
            logger.error("[扫描] AI扫描失败: %s", e, exc_info=True)
            return {'action_list': [], 'total_scored': 0, 'error': str(e)}

    def _execute_sells(self) -> list:
        """检查持仓并执行卖出"""
        sell_actions = []
        positions = self.account.get_positions()

        if positions.empty:
            logger.info("[卖出] 无持仓, 跳过")
            return sell_actions

        # 批量获取实时价格
        codes = positions['stock_code'].tolist()
        rt_prices = self._get_current_prices(codes)

        for _, pos in positions.iterrows():
            code = pos['stock_code']
            name = pos.get('stock_name', '')
            avg_cost = pos['avg_cost']
            shares = pos['shares']
            current_price = rt_prices.get(code, 0)

            if current_price <= 0:
                logger.warning("[卖出] %s(%s) 无法获取价格, 跳过", name, code)
                continue

            # 使用 position_monitor 检查卖出条件
            buy_date = pos.get('created_at', datetime.now().strftime('%Y-%m-%d'))
            if isinstance(buy_date, str) and len(buy_date) > 10:
                buy_date = buy_date[:10]

            try:
                result = check_single_position(
                    stock_code=code,
                    stock_name=name,
                    buy_price=avg_cost,
                    buy_date=buy_date,
                    shares=shares,
                    use_realtime=False,
                    realtime_price=current_price,
                )
            except Exception as e:
                logger.error("[卖出] %s(%s) 检查失败: %s", name, code, e)
                continue

            advice = result.get('advice', '继续持有')
            alerts = result.get('alerts', [])

            # 根据配置决定卖出紧急度
            should_sell = False
            if advice == '立即卖出':
                should_sell = True
            elif advice == '建议卖出' and config.AUTO_SELL_URGENCY <= 1:
                should_sell = True

            if should_sell:
                reason = '; '.join(alerts) if alerts else advice
                sell_result = self.account.sell(code, name, current_price, shares)

                if sell_result.get('success'):
                    pnl = sell_result.get('profit', 0)
                    pnl_pct = sell_result.get('profit_pct', 0)

                    action_info = {
                        'stock_code': code,
                        'stock_name': name,
                        'price': current_price,
                        'shares': shares,
                        'reason': reason,
                        'pnl': round(pnl, 2),
                        'pnl_pct': round(pnl_pct, 2),
                        'avg_cost': avg_cost,
                        'time_phase': result.get('time_phase_name', ''),
                    }
                    sell_actions.append(action_info)

                    self._log_trade(
                        action='卖出', stock_code=code, stock_name=name,
                        price=current_price, shares=shares, reason=reason,
                        stop_price=result.get('stop_price', 0),
                        target_price=result.get('target_price', 0),
                        pnl=round(pnl, 2), pnl_pct=round(pnl_pct, 2),
                    )
                    logger.info("[卖出] %s(%s) @%.2f %d股, 盈亏%.2f(%.1f%%), 原因: %s",
                                name, code, current_price, shares, pnl, pnl_pct, reason)
                else:
                    logger.error("[卖出] %s(%s) 失败: %s", name, code, sell_result.get('message'))
            else:
                if alerts:
                    logger.info("[持有] %s(%s) 价格%.2f, 提示: %s",
                                name, code, current_price, '; '.join(alerts))

        return sell_actions

    def _execute_buys(self, exclude_codes: set = None) -> tuple:
        """
        筛选标的并执行买入

        参数:
            exclude_codes: 需要排除的股票代码集合 (如刚卖出的股票，防止立即回买)
        """
        buy_actions = []
        skipped = []
        exclude_codes = exclude_codes or set()

        # 当前持仓数
        positions = self.account.get_positions()
        current_count = len(positions)
        available_slots = config.AUTO_MAX_POSITIONS - current_count

        if available_slots <= 0:
            logger.info("[买入] 持仓已满(%d/%d), 跳过",
                        current_count, config.AUTO_MAX_POSITIONS)
            return buy_actions, skipped

        # 已持仓代码 + 刚卖出的代码 (避免重复买入/立即回买)
        held_codes = set(positions['stock_code'].tolist()) if not positions.empty else set()
        blocked_codes = held_codes | exclude_codes
        if exclude_codes:
            logger.info("[买入] 排除刚卖出的 %d 只股票, 防止立即回买", len(exclude_codes))

        # 加载AI推荐
        recommendations = self._load_ai_recommendations()
        if not recommendations:
            logger.info("[买入] 无AI推荐数据, 跳过")
            return buy_actions, skipped

        # 筛选候选
        candidates = []
        for rec in recommendations:
            code = rec.get('stock_code', '')
            score = rec.get('ai_score', 0) or rec.get('final_score', 0)
            buy_upper = rec.get('buy_upper', 0)
            buy_price = rec.get('buy_price', 0)

            # 过滤条件
            if code in blocked_codes:
                continue
            if score < config.AUTO_SCORE_THRESHOLD:
                continue
            if buy_upper <= 0 or buy_price <= 0:
                continue

            candidates.append({
                'code': code,
                'name': rec.get('stock_name', ''),
                'score': score,
                'buy_price': buy_price,
                'buy_upper': buy_upper,
                'sell_target': rec.get('sell_target', 0),
                'sell_stop': rec.get('sell_stop', 0),
                'position_value': rec.get('position_value', config.POSITION_RATIO),
                'position_pct': rec.get('position_pct', ''),
                'risk_reward': rec.get('risk_reward', 0),
                'hold_days': rec.get('hold_days', ''),
                'exit_rules': rec.get('exit_rules', ''),
                'rec': rec,
            })

        if not candidates:
            logger.info("[买入] 无符合条件的候选 (阈值: score>=%d)", config.AUTO_SCORE_THRESHOLD)
            return buy_actions, skipped

        # 按评分排序
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # 批量获取当前价格 (先获取再过滤价格不合适的)
        cand_codes = [c['code'] for c in candidates]
        rt_prices = self._get_current_prices(cand_codes)

        # 过滤掉价格不可用或超过上限的
        valid_candidates = []
        for cand in candidates:
            code = cand['code']
            current_price = rt_prices.get(code, 0)
            if current_price <= 0:
                skipped.append({'code': code, 'name': cand['name'], 'reason': '无法获取价格'})
                continue
            if current_price > cand['buy_upper']:
                skipped.append({
                    'code': code, 'name': cand['name'],
                    'reason': f"现价{current_price:.2f}超过上限{cand['buy_upper']:.2f}"
                })
                continue
            cand['current_price'] = current_price
            valid_candidates.append(cand)

        if not valid_candidates:
            return buy_actions, skipped

        # 限制在可用空位数内
        valid_candidates = valid_candidates[:available_slots]
        n_to_buy = len(valid_candidates)

        account_info = self.account.get_account_info()
        available_cash = account_info['cash']

        # 仓位分配策略:
        # 候选 <= 10 只: 用 Kelly 仓位 (精选重仓)
        # 候选 > 10 只: 等权分配 (分散验证)
        use_equal_weight = (n_to_buy > 10)
        if use_equal_weight:
            per_stock_amount = available_cash * 0.98 / n_to_buy  # 留2%缓冲
            logger.info("[买入] 等权分配模式: %d只, 每只约¥%.0f", n_to_buy, per_stock_amount)

        for cand in valid_candidates:
            code = cand['code']
            name = cand['name']
            current_price = cand['current_price']

            # 计算买入股数
            if use_equal_weight:
                target_amount = min(per_stock_amount, available_cash * 0.95)
            elif config.AUTO_USE_KELLY_SIZE:
                position_ratio = min(cand['position_value'], config.MAX_SINGLE_POSITION)
                position_ratio = max(position_ratio, config.MIN_SINGLE_POSITION)
                target_amount = account_info['initial_capital'] * position_ratio
                target_amount = min(target_amount, available_cash * 0.95)
            else:
                target_amount = account_info['initial_capital'] * config.POSITION_RATIO
                target_amount = min(target_amount, available_cash * 0.95)

            shares = int(target_amount / current_price / 100) * 100
            if shares < 100:
                skipped.append({
                    'code': code, 'name': name,
                    'reason': f"资金不足(需{current_price * 100:.0f}, 可用{available_cash:.0f})"
                })
                continue

            # 执行买入
            buy_result = self.account.buy(code, name, current_price, shares)

            if buy_result.get('success'):
                cost = buy_result.get('cost', current_price * shares)
                available_cash -= cost

                action_info = {
                    'stock_code': code,
                    'stock_name': name,
                    'price': current_price,
                    'shares': shares,
                    'cost': round(cost, 2),
                    'ai_score': cand['score'],
                    'sell_target': cand['sell_target'],
                    'sell_stop': cand['sell_stop'],
                    'position_pct': cand['position_pct'],
                    'risk_reward': cand['risk_reward'],
                }
                buy_actions.append(action_info)

                advice_json = json.dumps({
                    'buy_price': cand['buy_price'],
                    'buy_upper': cand['buy_upper'],
                    'sell_target': cand['sell_target'],
                    'sell_stop': cand['sell_stop'],
                    'hold_days': cand['hold_days'],
                    'exit_rules': cand['exit_rules'],
                }, ensure_ascii=False)

                self._log_trade(
                    action='买入', stock_code=code, stock_name=name,
                    price=current_price, shares=shares,
                    reason=f"AI评分{cand['score']}, 盈亏比{cand['risk_reward']:.1f}",
                    ai_score=cand['score'],
                    stop_price=cand['sell_stop'],
                    target_price=cand['sell_target'],
                    advice_json=advice_json,
                )
                logger.info("[买入] %s(%s) @%.2f %d股, AI=%s, 止损=%.2f, 目标=%.2f",
                            name, code, current_price, shares,
                            cand['score'], cand['sell_stop'], cand['sell_target'])
            else:
                skipped.append({
                    'code': code, 'name': name,
                    'reason': buy_result.get('message', '买入失败')
                })

        return buy_actions, skipped

    def _save_daily_snapshot(self):
        """保存每日资产快照"""
        try:
            positions = self.account.get_positions()
            codes = positions['stock_code'].tolist() if not positions.empty else []
            rt_prices = self._get_current_prices(codes) if codes else {}
            equity = self.account.get_total_equity(rt_prices)

            today = datetime.now().strftime('%Y-%m-%d')
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                'INSERT OR REPLACE INTO daily_snapshot (date, cash, stock_value, total_equity) '
                'VALUES (?, ?, ?, ?)',
                (today, equity['cash'], equity['stock_value'], equity['total_equity'])
            )
            conn.commit()
            conn.close()
            logger.info("[快照] 资产=¥%.0f, 现金=¥%.0f, 持仓=¥%.0f",
                        equity['total_equity'], equity['cash'], equity['stock_value'])
        except Exception as e:
            logger.error("[快照] 保存失败: %s", e)

    def get_trade_log(self, limit: int = 50) -> list:
        """获取自动交易日志"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM auto_trade_log ORDER BY created_at DESC LIMIT ?',
            (limit,)
        )
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        return [dict(zip(columns, row)) for row in rows]

    def get_trade_log_df(self, limit: int = 200):
        """获取自动交易日志 (DataFrame)"""
        import pandas as pd
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            f'SELECT * FROM auto_trade_log ORDER BY created_at DESC LIMIT {limit}',
            conn
        )
        conn.close()
        return df

    def get_daily_snapshots(self, limit: int = 365):
        """获取每日资产快照"""
        import pandas as pd
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            f'SELECT * FROM daily_snapshot ORDER BY date DESC LIMIT {limit}',
            conn
        )
        conn.close()
        return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    rescan = '--rescan' in sys.argv
    trader = AutoTrader()
    result = trader.execute(rescan=rescan)
    print(f"\n{result['summary']}")
    print(f"卖出: {len(result['sell_actions'])}笔")
    print(f"买入: {len(result['buy_actions'])}笔")
    print(f"跳过: {len(result['skipped'])}个")
