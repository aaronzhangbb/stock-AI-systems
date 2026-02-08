"""
回测系统模块
使用自研轻量级回测引擎，模拟历史交易
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from src.strategy.strategy import apply_strategy


class BacktestResult:
    """回测结果"""

    def __init__(self):
        self.initial_capital = 0.0
        self.final_capital = 0.0
        self.total_return = 0.0
        self.annual_return = 0.0
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.win_trades = 0
        self.lose_trades = 0
        self.win_rate = 0.0
        self.trade_log = []
        self.equity_curve = []

    def summary(self) -> str:
        """生成回测摘要"""
        return (
            f"{'=' * 50}\n"
            f"          回测结果报告\n"
            f"{'=' * 50}\n"
            f"初始资金:     ¥{self.initial_capital:>12,.2f}\n"
            f"最终资金:     ¥{self.final_capital:>12,.2f}\n"
            f"总收益率:     {self.total_return:>11.2f}%\n"
            f"年化收益率:   {self.annual_return:>11.2f}%\n"
            f"最大回撤:     {self.max_drawdown:>11.2f}%\n"
            f"{'─' * 50}\n"
            f"总交易次数:   {self.total_trades:>8d} 次\n"
            f"盈利次数:     {self.win_trades:>8d} 次\n"
            f"亏损次数:     {self.lose_trades:>8d} 次\n"
            f"胜率:         {self.win_rate:>11.2f}%\n"
            f"{'=' * 50}"
        )


def run_backtest(df: pd.DataFrame, initial_capital: float = None, position_ratio: float = None) -> BacktestResult:
    """
    运行回测

    参数:
        df: 原始 OHLCV 数据（未应用策略的）
        initial_capital: 初始资金
        position_ratio: 单次买入仓位比例

    返回:
        BacktestResult: 回测结果
    """
    if initial_capital is None:
        initial_capital = config.INITIAL_CAPITAL
    if position_ratio is None:
        position_ratio = config.POSITION_RATIO

    # 应用策略
    df = apply_strategy(df)
    if df.empty or 'signal' not in df.columns:
        result = BacktestResult()
        result.initial_capital = initial_capital
        result.final_capital = initial_capital
        return result

    result = BacktestResult()
    result.initial_capital = initial_capital

    cash = initial_capital
    shares = 0          # 持仓股数
    buy_price = 0.0     # 买入价格
    equity_curve = []   # 资金曲线
    trade_log = []      # 交易记录

    for i, row in df.iterrows():
        current_price = row['close']
        signal = row['signal']
        date = row['date']

        if signal == 1 and shares == 0:
            # 买入信号，且当前空仓
            available = cash * position_ratio
            # A股最少买100股（1手）
            buy_shares = int(available / current_price / 100) * 100
            if buy_shares >= 100:
                cost = buy_shares * current_price
                # 计算佣金
                commission = max(cost * config.COMMISSION_RATE, config.MIN_COMMISSION)
                total_cost = cost + commission

                if total_cost <= cash:
                    cash -= total_cost
                    shares = buy_shares
                    buy_price = current_price
                    trade_log.append({
                        'date': date,
                        'action': '买入',
                        'price': current_price,
                        'shares': buy_shares,
                        'cost': total_cost,
                        'commission': commission,
                    })

        elif signal == -1 and shares > 0:
            # 卖出信号，且当前有持仓
            revenue = shares * current_price
            # 计算佣金和印花税
            commission = max(revenue * config.COMMISSION_RATE, config.MIN_COMMISSION)
            stamp_tax = revenue * config.STAMP_TAX_RATE
            net_revenue = revenue - commission - stamp_tax

            profit = net_revenue - (shares * buy_price)
            profit_pct = (current_price - buy_price) / buy_price * 100

            cash += net_revenue

            trade_log.append({
                'date': date,
                'action': '卖出',
                'price': current_price,
                'shares': shares,
                'revenue': net_revenue,
                'commission': commission,
                'stamp_tax': stamp_tax,
                'profit': profit,
                'profit_pct': profit_pct,
            })

            if profit > 0:
                result.win_trades += 1
            else:
                result.lose_trades += 1

            shares = 0
            buy_price = 0.0

        # 记录每日资产
        total_equity = cash + shares * current_price
        equity_curve.append({
            'date': date,
            'cash': cash,
            'stock_value': shares * current_price,
            'total_equity': total_equity,
        })

    # 计算回测指标
    result.final_capital = cash + shares * df.iloc[-1]['close']
    result.total_return = (result.final_capital - initial_capital) / initial_capital * 100
    result.total_trades = len([t for t in trade_log if t['action'] == '卖出'])

    if result.total_trades > 0:
        result.win_rate = result.win_trades / result.total_trades * 100

    # 计算年化收益率
    if len(df) > 1:
        days = (df.iloc[-1]['date'] - df.iloc[0]['date']).days
        if days > 0:
            result.annual_return = (pow(result.final_capital / initial_capital, 365 / days) - 1) * 100

    # 计算最大回撤
    if equity_curve:
        equity_df = pd.DataFrame(equity_curve)
        equity_df['peak'] = equity_df['total_equity'].cummax()
        equity_df['drawdown'] = (equity_df['total_equity'] - equity_df['peak']) / equity_df['peak'] * 100
        result.max_drawdown = equity_df['drawdown'].min()

    result.trade_log = trade_log
    result.equity_curve = equity_curve

    return result


if __name__ == '__main__':
    from src.data.data_fetcher import get_history_data

    print("=" * 60)
    print("回测测试 - 贵州茅台(600519) 近一年")
    print("=" * 60)

    df = get_history_data('600519', days=365)
    if not df.empty:
        result = run_backtest(df)
        print(result.summary())

        if result.trade_log:
            print("\n交易记录:")
            for trade in result.trade_log:
                if trade['action'] == '买入':
                    print(f"  {trade['date'].strftime('%Y-%m-%d')} 买入 {trade['shares']}股 @ ¥{trade['price']:.2f}")
                else:
                    print(f"  {trade['date'].strftime('%Y-%m-%d')} 卖出 {trade['shares']}股 @ ¥{trade['price']:.2f}  "
                          f"盈亏: ¥{trade['profit']:.2f} ({trade['profit_pct']:.2f}%)")
    else:
        print("获取数据失败!")

