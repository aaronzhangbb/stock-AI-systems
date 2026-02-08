"""
策略引擎模块
实现双均线策略 + RSI 过滤
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


def calculate_ma(df: pd.DataFrame, period: int, col: str = 'close') -> pd.Series:
    """计算移动平均线"""
    return df[col].rolling(window=period).mean()


def calculate_rsi(df: pd.DataFrame, period: int = 14, col: str = 'close') -> pd.Series:
    """
    计算 RSI（相对强弱指标）
    RSI = 100 - 100 / (1 + RS)
    RS = 平均上涨幅度 / 平均下跌幅度
    """
    delta = df[col].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    应用双均线 + RSI 策略

    策略规则:
    - 买入信号 (金叉): MA_SHORT 从下往上穿过 MA_LONG，且 RSI < RSI_OVERBOUGHT
    - 卖出信号 (死叉): MA_SHORT 从上往下穿过 MA_LONG，且 RSI > RSI_OVERSOLD

    参数:
        df: 包含 OHLCV 数据的 DataFrame

    返回:
        DataFrame: 增加了 MA、RSI、信号列的 DataFrame
    """
    if df.empty or len(df) < config.MA_LONG:
        return df

    df = df.copy()

    # 计算均线
    df['ma_short'] = calculate_ma(df, config.MA_SHORT)
    df['ma_long'] = calculate_ma(df, config.MA_LONG)

    # 计算 RSI
    df['rsi'] = calculate_rsi(df, config.RSI_PERIOD)

    # 判断均线位置关系: 1=短线在上, 0=短线在下
    df['ma_position'] = np.where(df['ma_short'] > df['ma_long'], 1, 0)

    # 检测交叉: diff=1 表示金叉（从下穿上），diff=-1 表示死叉（从上穿下）
    df['ma_cross'] = df['ma_position'].diff()

    # 生成信号
    # signal: 1=买入, -1=卖出, 0=观望
    df['signal'] = 0

    # 金叉买入（且 RSI 不在超买区）
    buy_condition = (df['ma_cross'] == 1) & (df['rsi'] < config.RSI_OVERBOUGHT)
    df.loc[buy_condition, 'signal'] = 1

    # 死叉卖出（且 RSI 不在超卖区）
    sell_condition = (df['ma_cross'] == -1) & (df['rsi'] > config.RSI_OVERSOLD)
    df.loc[sell_condition, 'signal'] = -1

    return df


def get_latest_signal(df: pd.DataFrame) -> dict:
    """
    获取最新的交易信号

    参数:
        df: 已经应用策略后的 DataFrame

    返回:
        dict: 包含信号信息
            - signal: 1(买入) / -1(卖出) / 0(观望)
            - signal_name: 信号名称
            - date: 信号日期
            - close: 当前价格
            - ma_short: 短期均线值
            - ma_long: 长期均线值
            - rsi: RSI值
    """
    if df.empty or 'signal' not in df.columns:
        return {'signal': 0, 'signal_name': '无数据'}

    latest = df.iloc[-1]

    signal = int(latest['signal'])
    if signal == 1:
        signal_name = f"🟢 买入信号（金叉）"
    elif signal == -1:
        signal_name = f"🔴 卖出信号（死叉）"
    else:
        # 判断当前趋势
        if pd.notna(latest.get('ma_short')) and pd.notna(latest.get('ma_long')):
            if latest['ma_short'] > latest['ma_long']:
                signal_name = "⚪ 多头趋势，持仓观望"
            else:
                signal_name = "⚪ 空头趋势，空仓观望"
        else:
            signal_name = "⚪ 数据不足，无法判断"

    return {
        'signal': signal,
        'signal_name': signal_name,
        'date': str(latest['date'].date()) if pd.notna(latest.get('date')) else '',
        'close': float(latest['close']) if pd.notna(latest.get('close')) else 0.0,
        'ma_short': float(latest['ma_short']) if pd.notna(latest.get('ma_short')) else 0.0,
        'ma_long': float(latest['ma_long']) if pd.notna(latest.get('ma_long')) else 0.0,
        'rsi': float(latest['rsi']) if pd.notna(latest.get('rsi')) else 0.0,
    }


def get_signal_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    获取所有买卖信号的历史记录

    参数:
        df: 已经应用策略后的 DataFrame

    返回:
        DataFrame: 只包含有信号的行
    """
    if df.empty or 'signal' not in df.columns:
        return pd.DataFrame()

    signals = df[df['signal'] != 0].copy()
    signals['signal_name'] = signals['signal'].map({1: '买入', -1: '卖出'})
    return signals


if __name__ == '__main__':
    # 测试策略
    from src.data.data_fetcher import get_history_data

    print("=" * 60)
    print("测试双均线策略 - 贵州茅台(600519)")
    print("=" * 60)

    df = get_history_data('600519', days=180)
    if not df.empty:
        df = apply_strategy(df)
        signal = get_latest_signal(df)

        print(f"\n最新信号: {signal['signal_name']}")
        print(f"日期: {signal['date']}")
        print(f"收盘价: {signal['close']:.2f}")
        print(f"MA{config.MA_SHORT}: {signal['ma_short']:.2f}")
        print(f"MA{config.MA_LONG}: {signal['ma_long']:.2f}")
        print(f"RSI: {signal['rsi']:.2f}")

        history = get_signal_history(df)
        if not history.empty:
            print(f"\n最近的买卖信号:")
            print(history[['date', 'close', 'signal_name', 'rsi']].tail(10).to_string(index=False))
    else:
        print("获取数据失败!")

