# -*- coding: utf-8 -*-
"""
持仓监控引擎
- 每日检查所有持仓股
- 检测止损 / 止盈 / 追踪止损 / 策略卖出信号
- 生成卖出建议
- 支持邮件推送
- 支持盘中实时价格监控
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from src.data.data_fetcher import get_history_data, get_realtime_price, batch_get_realtime_prices
from src.strategy.strategies import run_all_strategies
from src.trading.paper_trading import PaperTradingAccount


def is_trading_time() -> bool:
    """判断当前是否在A股交易时间内（9:30-11:30, 13:00-15:00）"""
    now = datetime.now()
    weekday = now.weekday()
    if weekday >= 5:  # 周六日
        return False
    hour, minute = now.hour, now.minute
    t = hour * 60 + minute
    # 上午 9:30-11:30 = 570-690, 下午 13:00-15:00 = 780-900
    return (570 <= t <= 690) or (780 <= t <= 900)


def check_single_position(stock_code: str, stock_name: str, buy_price: float,
                           buy_date: str, shares: int = 0,
                           use_realtime: bool = False,
                           realtime_price: float = None) -> dict:
    """
    检查单只持仓股的卖出条件

    参数:
        use_realtime: 是否使用实时价格（盘中监控时设为True）
        realtime_price: 外部传入的实时价格（批量获取后传入，避免逐只请求）

    返回:
        dict: {
            stock_code, stock_name, buy_price, current_price,
            pnl_pct, stop_price, target_price, trailing_stop_price,
            alerts: list[str],  # 触发的卖出原因
            advice: str,  # '持有' / '建议卖出' / '立即卖出'
            sell_signals: list  # 策略卖出信号
        }
    """
    result = {
        'stock_code': stock_code,
        'stock_name': stock_name,
        'buy_price': buy_price,
        'buy_date': buy_date,
        'shares': shares,
        'current_price': 0,
        'pnl_pct': 0,
        'stop_price': round(buy_price * (1 - config.STOP_LOSS_PCT), 2),
        'target_price': round(buy_price * (1 + config.TAKE_PROFIT_PCT), 2),
        'trailing_stop_price': 0,
        'high_since_buy': 0,
        'alerts': [],
        'advice': '持有',
        'sell_signals': [],
        'price_source': 'close',  # 'close' 或 'realtime'
    }

    try:
        df = get_history_data(stock_code, days=120, use_cache=True)
        if df.empty:
            result['alerts'].append('无法获取行情数据')
            return result

        # 优先使用外部传入的实时价格，其次尝试实时接口，最后用收盘价
        current_price = float(df['close'].iloc[-1])
        if realtime_price is not None and realtime_price > 0:
            current_price = float(realtime_price)
            result['price_source'] = 'realtime'
        elif use_realtime and is_trading_time():
            try:
                rt = get_realtime_price(stock_code)
                if rt and rt.get('close', 0) > 0:
                    current_price = float(rt['close'])
                    result['price_source'] = 'realtime'
            except Exception:
                pass  # 实时价格获取失败时降级使用收盘价

        result['current_price'] = current_price

        # 计算盈亏
        pnl_pct = (current_price - buy_price) / buy_price * 100
        result['pnl_pct'] = round(pnl_pct, 2)

        # 买入以来最高价（用于追踪止损）
        buy_date_str = buy_date[:10]  # 取日期部分
        mask = df['date'].astype(str) >= buy_date_str
        df_since = df[mask]
        if df_since.empty:
            df_since = df.tail(30)  # 回退到最近30天

        high_since = float(df_since['high'].max())
        result['high_since_buy'] = high_since

        # 追踪止损价 = 最高价 × (1 - 追踪止损比例)
        trailing_stop = round(high_since * (1 - config.TRAILING_STOP_PCT), 2)
        result['trailing_stop_price'] = trailing_stop

        # ---- 检查卖出条件 ----
        urgency = 0  # 0=持有, 1=建议卖出, 2=立即卖出

        # 1) 止损
        if current_price <= result['stop_price']:
            result['alerts'].append(f"触发止损（止损价 {result['stop_price']:.2f}）")
            urgency = max(urgency, 2)

        # 2) 止盈
        if current_price >= result['target_price']:
            result['alerts'].append(f"触发止盈（目标价 {result['target_price']:.2f}）")
            urgency = max(urgency, 1)

        # 3) 追踪止损（只在盈利状态下生效）
        if pnl_pct > 5 and current_price <= trailing_stop:
            result['alerts'].append(
                f"触发追踪止损（最高 {high_since:.2f} → 回落至 {current_price:.2f}）"
            )
            urgency = max(urgency, 2)

        # 4) 策略卖出信号
        try:
            sigs = run_all_strategies(df)
            sell_sigs = [s for s in sigs if s['signal'] == 'sell']
            if sell_sigs:
                result['sell_signals'] = sell_sigs
                names = ', '.join([s['strategy'] for s in sell_sigs])
                result['alerts'].append(f"策略卖出信号（{names}）")
                urgency = max(urgency, 1)
        except Exception:
            pass

        # 汇总建议
        if urgency >= 2:
            result['advice'] = '立即卖出'
        elif urgency >= 1:
            result['advice'] = '建议卖出'
        else:
            result['advice'] = '继续持有'

    except Exception as e:
        result['alerts'].append(f'监控异常: {e}')

    return result


def check_all_manual_positions(account: PaperTradingAccount = None,
                                use_realtime: bool = False) -> list:
    """
    检查所有手动买入跟踪的持仓（批量获取实时价格，大幅提速）

    参数:
        account: 交易账户实例
        use_realtime: 是否使用实时价格（盘中监控时设为True）

    返回:
        list[dict]: 每只持仓的检查结果
    """
    if account is None:
        account = PaperTradingAccount()

    manual_df = account.list_manual_positions()
    if manual_df.empty:
        return []

    # 批量获取所有持仓的实时价格（一次网络请求）
    all_codes = manual_df['stock_code'].tolist()
    rt_prices = {}
    try:
        rt_map = batch_get_realtime_prices(all_codes)
        for code, info in rt_map.items():
            if info.get('close', 0) > 0:
                rt_prices[code] = info['close']
    except Exception as e:
        print(f"批量获取实时价格失败，将逐只获取: {e}")

    results = []
    for _, row in manual_df.iterrows():
        code = row['stock_code']
        r = check_single_position(
            stock_code=code,
            stock_name=row.get('stock_name', ''),
            buy_price=float(row['buy_price']),
            buy_date=row['buy_date'],
            shares=int(row.get('shares', 0)),
            use_realtime=use_realtime,
            realtime_price=rt_prices.get(code),
        )
        results.append(r)

    return results


def get_sell_alerts(results: list) -> list:
    """
    从检查结果中筛选出需要操作的持仓（有卖出提醒的）

    返回:
        list[dict]: 需要操作的持仓列表
    """
    return [r for r in results if r['alerts'] and r['advice'] != '继续持有']


def format_sell_alerts_text(alerts: list) -> str:
    """
    将卖出提醒格式化为可读文本（用于邮件/通知）
    """
    if not alerts:
        return ""

    lines = ["【持仓卖出提醒】", ""]
    for i, a in enumerate(alerts, 1):
        status_icon = "🔴" if a['advice'] == '立即卖出' else "🟡"
        pnl_sign = "+" if a['pnl_pct'] >= 0 else ""
        lines.append(
            f"{i}. {a['stock_name']}({a['stock_code']}) "
            f"买入价:{a['buy_price']:.2f} 现价:{a['current_price']:.2f} "
            f"({pnl_sign}{a['pnl_pct']:.1f}%)"
        )
        lines.append(f"   {status_icon} 状态: {a['advice']}")
        for alert_msg in a['alerts']:
            lines.append(f"   · {alert_msg}")
        lines.append("")

    return "\n".join(lines)
