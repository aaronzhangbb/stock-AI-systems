"""
模拟交易账户模块
管理虚拟资金、持仓、交易流水
"""

import sqlite3
import pandas as pd
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


class PaperTradingAccount:
    """模拟交易账户"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), '..', '..', config.DB_PATH)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 账户信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS account (
                id INTEGER PRIMARY KEY,
                initial_capital REAL NOT NULL,
                cash REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')

        # 持仓表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT NOT NULL,
                stock_name TEXT DEFAULT '',
                shares INTEGER NOT NULL,
                avg_cost REAL NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(stock_code)
            )
        ''')

        # 交易流水表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT NOT NULL,
                stock_name TEXT DEFAULT '',
                action TEXT NOT NULL,
                price REAL NOT NULL,
                shares INTEGER NOT NULL,
                amount REAL NOT NULL,
                commission REAL NOT NULL,
                stamp_tax REAL DEFAULT 0,
                profit REAL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        ''')

        # 每日资产快照表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_snapshot (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                cash REAL NOT NULL,
                stock_value REAL NOT NULL,
                total_equity REAL NOT NULL
            )
        ''')

        # 手动买入跟踪表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS manual_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT NOT NULL,
                stock_name TEXT DEFAULT '',
                buy_price REAL NOT NULL,
                buy_date TEXT NOT NULL,
                shares INTEGER DEFAULT 0,
                note TEXT DEFAULT '',
                status TEXT DEFAULT 'holding',
                sell_price REAL DEFAULT 0,
                sell_date TEXT DEFAULT '',
                actual_pnl REAL DEFAULT 0,
                actual_pnl_pct REAL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(stock_code, buy_date)
            )
        ''')

        # 自动迁移：为旧表添加卖出字段（如果缺失）
        try:
            cursor.execute('SELECT sell_price FROM manual_positions LIMIT 1')
        except sqlite3.OperationalError:
            cursor.execute('ALTER TABLE manual_positions ADD COLUMN sell_price REAL DEFAULT 0')
            cursor.execute('ALTER TABLE manual_positions ADD COLUMN sell_date TEXT DEFAULT ""')
            cursor.execute('ALTER TABLE manual_positions ADD COLUMN actual_pnl REAL DEFAULT 0')
            cursor.execute('ALTER TABLE manual_positions ADD COLUMN actual_pnl_pct REAL DEFAULT 0')

        conn.commit()

        # 检查是否已有账户，没有则创建
        cursor.execute('SELECT COUNT(*) FROM account')
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                'INSERT INTO account (id, initial_capital, cash, created_at) VALUES (1, ?, ?, ?)',
                (config.INITIAL_CAPITAL, config.INITIAL_CAPITAL, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            )
            conn.commit()

        conn.close()

    def get_account_info(self) -> dict:
        """获取账户信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT initial_capital, cash FROM account WHERE id=1')
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'initial_capital': row[0],
                'cash': row[1],
            }
        return {'initial_capital': 0, 'cash': 0}

    def get_positions(self) -> pd.DataFrame:
        """获取当前持仓"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('SELECT * FROM positions WHERE shares > 0', conn)
        conn.close()
        return df

    def get_trades(self, limit: int = 50) -> pd.DataFrame:
        """获取交易记录"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            f'SELECT * FROM trades ORDER BY created_at DESC LIMIT {limit}', conn
        )
        conn.close()
        return df

    def add_manual_position(self, stock_code: str, stock_name: str, buy_price: float,
                            buy_date: str, shares: int = 0, note: str = '') -> dict:
        """添加手动买入跟踪记录"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                'INSERT OR REPLACE INTO manual_positions '
                '(stock_code, stock_name, buy_price, buy_date, shares, note, status, created_at, updated_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (stock_code, stock_name, buy_price, buy_date, shares, note, 'holding', now, now)
            )
            conn.commit()
            return {'success': True, 'message': '已记录手动买入'}
        except Exception as e:
            return {'success': False, 'message': f'记录失败: {e}'}
        finally:
            conn.close()

    def list_manual_positions(self) -> pd.DataFrame:
        """获取手动买入跟踪列表"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('SELECT * FROM manual_positions WHERE status = "holding"', conn)
        conn.close()
        return df

    def remove_manual_position(self, stock_code: str, buy_date: str) -> None:
        """移除手动买入跟踪记录（不记录卖出信息，仅状态关闭）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE manual_positions SET status="closed", updated_at=? WHERE stock_code=? AND buy_date=?',
            (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), stock_code, buy_date)
        )
        conn.commit()
        conn.close()

    def sell_manual_position(self, stock_code: str, buy_date: str,
                              sell_price: float, sell_date: str) -> dict:
        """
        录入卖出信息并关闭持仓

        参数:
            stock_code: 股票代码
            buy_date: 买入日期（用于定位持仓记录）
            sell_price: 卖出价格
            sell_date: 卖出日期
        返回:
            dict: {success, message, pnl, pnl_pct}
        """
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            # 查找对应持仓
            cursor.execute(
                'SELECT buy_price, shares, stock_name FROM manual_positions '
                'WHERE stock_code=? AND buy_date=? AND status="holding"',
                (stock_code, buy_date)
            )
            row = cursor.fetchone()
            if not row:
                return {'success': False, 'message': '未找到对应的持仓记录'}

            buy_price, shares, stock_name = row[0], row[1], row[2]

            # 计算盈亏
            pnl_per_share = sell_price - buy_price
            pnl_pct = (pnl_per_share / buy_price) * 100 if buy_price > 0 else 0
            total_pnl = pnl_per_share * shares if shares > 0 else pnl_per_share

            # 更新记录
            cursor.execute(
                'UPDATE manual_positions SET '
                'status="sold", sell_price=?, sell_date=?, '
                'actual_pnl=?, actual_pnl_pct=?, updated_at=? '
                'WHERE stock_code=? AND buy_date=? AND status="holding"',
                (sell_price, sell_date, round(total_pnl, 2), round(pnl_pct, 2),
                 now, stock_code, buy_date)
            )
            conn.commit()

            pnl_sign = "+" if pnl_pct >= 0 else ""
            return {
                'success': True,
                'message': f'{stock_name}({stock_code}) 卖出 @{sell_price:.2f}，盈亏 {pnl_sign}{pnl_pct:.1f}%',
                'pnl': round(total_pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
            }
        except Exception as e:
            return {'success': False, 'message': f'卖出录入失败: {e}'}
        finally:
            conn.close()

    def list_closed_positions(self, limit: int = 50) -> pd.DataFrame:
        """获取已卖出/已关闭的持仓记录"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            f'SELECT * FROM manual_positions WHERE status IN ("sold", "closed") '
            f'ORDER BY updated_at DESC LIMIT {limit}', conn
        )
        conn.close()
        return df

    def buy(self, stock_code: str, stock_name: str, price: float, shares: int = None) -> dict:
        """
        模拟买入

        参数:
            stock_code: 股票代码
            stock_name: 股票名称
            price: 买入价格
            shares: 买入股数，为 None 时按仓位比例自动计算

        返回:
            dict: 交易结果
        """
        account = self.get_account_info()
        cash = account['cash']

        if shares is None:
            # 按仓位比例计算
            available = cash * config.POSITION_RATIO
            shares = int(available / price / 100) * 100  # A股最少100股

        if shares < 100:
            return {'success': False, 'message': '资金不足，无法买入最少100股'}

        amount = shares * price
        commission = max(amount * config.COMMISSION_RATE, config.MIN_COMMISSION)
        total_cost = amount + commission

        if total_cost > cash:
            return {'success': False, 'message': f'资金不足！需要 ¥{total_cost:.2f}，可用 ¥{cash:.2f}'}

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # 更新现金
            new_cash = cash - total_cost
            cursor.execute('UPDATE account SET cash=? WHERE id=1', (new_cash,))

            # 更新持仓
            cursor.execute('SELECT shares, avg_cost FROM positions WHERE stock_code=?', (stock_code,))
            existing = cursor.fetchone()

            if existing:
                old_shares, old_cost = existing
                new_shares = old_shares + shares
                new_avg_cost = (old_shares * old_cost + shares * price) / new_shares
                cursor.execute(
                    'UPDATE positions SET shares=?, avg_cost=?, updated_at=? WHERE stock_code=?',
                    (new_shares, new_avg_cost, now, stock_code)
                )
            else:
                cursor.execute(
                    'INSERT INTO positions (stock_code, stock_name, shares, avg_cost, created_at, updated_at) '
                    'VALUES (?, ?, ?, ?, ?, ?)',
                    (stock_code, stock_name, shares, price, now, now)
                )

            # 记录交易
            cursor.execute(
                'INSERT INTO trades (stock_code, stock_name, action, price, shares, amount, commission, created_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (stock_code, stock_name, '买入', price, shares, amount, commission, now)
            )

            conn.commit()

            return {
                'success': True,
                'message': f'买入成功！{stock_name}({stock_code}) {shares}股 @ ¥{price:.2f}，'
                           f'花费 ¥{total_cost:.2f}（含佣金 ¥{commission:.2f}）',
                'shares': shares,
                'price': price,
                'cost': total_cost,
                'remaining_cash': new_cash,
            }

        except Exception as e:
            conn.rollback()
            return {'success': False, 'message': f'买入失败: {e}'}
        finally:
            conn.close()

    def sell(self, stock_code: str, stock_name: str, price: float, shares: int = None) -> dict:
        """
        模拟卖出

        参数:
            stock_code: 股票代码
            stock_name: 股票名称
            price: 卖出价格
            shares: 卖出股数，为 None 时全部卖出

        返回:
            dict: 交易结果
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT shares, avg_cost FROM positions WHERE stock_code=?', (stock_code,))
        position = cursor.fetchone()

        if not position or position[0] <= 0:
            conn.close()
            return {'success': False, 'message': f'没有 {stock_code} 的持仓'}

        held_shares, avg_cost = position

        if shares is None:
            shares = held_shares

        if shares > held_shares:
            conn.close()
            return {'success': False, 'message': f'持仓不足！持有 {held_shares} 股，要卖 {shares} 股'}

        amount = shares * price
        commission = max(amount * config.COMMISSION_RATE, config.MIN_COMMISSION)
        stamp_tax = amount * config.STAMP_TAX_RATE
        net_revenue = amount - commission - stamp_tax
        profit = (price - avg_cost) * shares - commission - stamp_tax

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        try:
            # 更新现金
            account = self.get_account_info()
            new_cash = account['cash'] + net_revenue
            cursor.execute('UPDATE account SET cash=? WHERE id=1', (new_cash,))

            # 更新持仓
            remaining = held_shares - shares
            if remaining > 0:
                cursor.execute(
                    'UPDATE positions SET shares=?, updated_at=? WHERE stock_code=?',
                    (remaining, now, stock_code)
                )
            else:
                cursor.execute('DELETE FROM positions WHERE stock_code=?', (stock_code,))

            # 记录交易
            cursor.execute(
                'INSERT INTO trades (stock_code, stock_name, action, price, shares, amount, commission, stamp_tax, profit, created_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (stock_code, stock_name, '卖出', price, shares, amount, commission, stamp_tax, profit, now)
            )

            conn.commit()

            profit_pct = (price - avg_cost) / avg_cost * 100
            return {
                'success': True,
                'message': f'卖出成功！{stock_name}({stock_code}) {shares}股 @ ¥{price:.2f}，'
                           f'盈亏 ¥{profit:.2f}（{profit_pct:+.2f}%）',
                'shares': shares,
                'price': price,
                'revenue': net_revenue,
                'profit': profit,
                'profit_pct': profit_pct,
                'remaining_cash': new_cash,
            }

        except Exception as e:
            conn.rollback()
            return {'success': False, 'message': f'卖出失败: {e}'}
        finally:
            conn.close()

    def get_total_equity(self, current_prices: dict = None) -> dict:
        """
        计算总资产

        参数:
            current_prices: {stock_code: current_price} 当前价格字典

        返回:
            dict: 资产信息
        """
        account = self.get_account_info()
        positions = self.get_positions()

        stock_value = 0.0
        position_details = []

        for _, pos in positions.iterrows():
            code = pos['stock_code']
            shares = pos['shares']
            avg_cost = pos['avg_cost']

            if current_prices and code in current_prices:
                current_price = current_prices[code]
            else:
                current_price = avg_cost  # 没有实时价格时用成本价

            value = shares * current_price
            profit = (current_price - avg_cost) * shares
            profit_pct = (current_price - avg_cost) / avg_cost * 100 if avg_cost > 0 else 0

            stock_value += value
            position_details.append({
                'code': code,
                'name': pos['stock_name'],
                'shares': shares,
                'avg_cost': avg_cost,
                'current_price': current_price,
                'value': value,
                'profit': profit,
                'profit_pct': profit_pct,
            })

        total_equity = account['cash'] + stock_value
        total_profit = total_equity - account['initial_capital']
        total_profit_pct = total_profit / account['initial_capital'] * 100 if account['initial_capital'] > 0 else 0

        return {
            'initial_capital': account['initial_capital'],
            'cash': account['cash'],
            'stock_value': stock_value,
            'total_equity': total_equity,
            'total_profit': total_profit,
            'total_profit_pct': total_profit_pct,
            'positions': position_details,
        }

    def reset_account(self):
        """重置模拟账户"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM positions')
        cursor.execute('DELETE FROM trades')
        cursor.execute('DELETE FROM daily_snapshot')
        cursor.execute(
            'UPDATE account SET cash=initial_capital WHERE id=1'
        )
        conn.commit()
        conn.close()


if __name__ == '__main__':
    print("=" * 60)
    print("模拟交易账户测试")
    print("=" * 60)

    account = PaperTradingAccount()
    info = account.get_account_info()
    print(f"\n初始资金: ¥{info['initial_capital']:,.2f}")
    print(f"可用现金: ¥{info['cash']:,.2f}")

    # 模拟买入
    result = account.buy('600519', '贵州茅台', 1500.0)
    print(f"\n{result['message']}")

    # 查看持仓
    equity = account.get_total_equity({'600519': 1550.0})
    print(f"\n总资产: ¥{equity['total_equity']:,.2f}")
    print(f"总盈亏: ¥{equity['total_profit']:,.2f} ({equity['total_profit_pct']:.2f}%)")

    # 重置
    account.reset_account()
    print("\n账户已重置")

