# -*- coding: utf-8 -*-
"""
持仓监控引擎 V2.0
- 每日检查所有持仓股
- 检测止损 / 止盈 / 追踪止损 / 策略卖出信号
- ★ 时间衰减退出机制: 止损随持有时间自动收紧
- 生成卖出建议
- 支持邮件推送
- 支持盘中实时价格监控

时间衰减退出模型 (4阶段):
    Phase 1 (0~60% 预估持有期):  正常持有, 标准ATR止损
    Phase 2 (60~100%):          警戒期, 止损开始收紧
    Phase 3 (100~150%):         超期宽限, 盈利保本/亏损强卖
    Phase 4 (>150%):            强制退出, 不论盈亏
"""

import os
import sys
import numpy as np
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
                           realtime_price: float = None,
                           ai_score_at_buy: float = 0,
                           ai_score_current: float = 0) -> dict:
    """
    检查单只持仓股的卖出条件

    参数:
        use_realtime: 是否使用实时价格（盘中监控时设为True）
        realtime_price: 外部传入的实时价格（批量获取后传入，避免逐只请求）
        ai_score_at_buy: 买入时的 AI 评分（用于技术面卖出确认）
        ai_score_current: 当前最新 AI 评分（用于技术面卖出确认）

    返回:
        dict: {
            stock_code, stock_name, buy_price, current_price,
            pnl_pct, stop_price, target_price, trailing_stop_price,
            alerts: list[str],  # 触发的卖出原因
            advice: str,  # '持有' / '建议卖出' / '立即卖出'
            sell_signals: list  # 策略卖出信号
        }
    """
    default_stop = round(buy_price * (1 - config.STOP_LOSS_PCT), 2)
    default_target = round(buy_price * (1 + config.TAKE_PROFIT_PCT), 2)
    
    result = {
        'stock_code': stock_code,
        'stock_name': stock_name,
        'buy_price': buy_price,
        'buy_date': buy_date,
        'shares': shares,
        'current_price': 0,
        'pnl_pct': 0,
        'stop_price': default_stop,
        'target_price': default_target,
        'trailing_stop_price': 0,
        'high_since_buy': 0,
        'alerts': [],
        'advice': '持有',
        'sell_signals': [],
        'price_source': 'close',
        'stop_method': 'fixed',
        'ai_score_at_buy': ai_score_at_buy,
        'ai_score_current': ai_score_current,
        # 时间衰减相关字段
        'days_held': 0,
        'est_hold_days': 10,
        'time_ratio': 0,
        'time_phase': 1,
        'time_phase_name': '正常持有',
        'original_stop': default_stop,
    }

    try:
        df = get_history_data(stock_code, days=120, use_cache=True)
        if df.empty:
            result['alerts'].append('无法获取行情数据')
            return result

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
            except Exception as exc:
                import logging as _pm_log
                _pm_log.getLogger(__name__).warning("获取实时价格失败 %s: %s", stock_code, exc)

        result['current_price'] = current_price

        pnl_pct = (current_price - buy_price) / buy_price * 100
        result['pnl_pct'] = round(pnl_pct, 2)

        # 买入以来的交易日数
        buy_date_str = buy_date[:10]
        latest_kline_date = str(df['date'].astype(str).iloc[-1])[:10]
        today_str = datetime.now().strftime('%Y-%m-%d')
        mask = df['date'].astype(str) >= buy_date_str
        df_since = df[mask]

        if df_since.empty:
            # 当天买入但日线尚未落库时，不能错误回退到近30天，
            # 否则会把新仓误判成“已持有很久”。
            if buy_date_str >= latest_kline_date or buy_date_str == today_str:
                days_held = 1
                high_since = max(float(current_price), float(buy_price))
            else:
                df_since = df.tail(1)
                days_held = 1
                high_since = max(float(df_since['high'].iloc[-1]), float(current_price), float(buy_price))
        else:
            days_held = max(len(df_since), 1)
            high_since = float(df_since['high'].max())

        result['days_held'] = days_held
        result['high_since_buy'] = high_since

        # ================================================================
        # ATR计算 + 动态止损/止盈
        # ================================================================
        atr_14 = buy_price * 0.03  # 默认值
        vol20 = 0.3
        stop_multi = 1.5
        
        try:
            close_s = df['close'].astype(float)
            high_s = df['high'].astype(float)
            low_s = df['low'].astype(float)
            
            tr = np.maximum(
                high_s - low_s,
                np.maximum(
                    (high_s - close_s.shift(1)).abs(),
                    (low_s - close_s.shift(1)).abs()
                )
            )
            _atr = float(tr.rolling(14).mean().iloc[-1])
            if _atr > 0 and not np.isnan(_atr):
                atr_14 = _atr
            
            ret = close_s.pct_change()
            _vol = float(ret.rolling(20).std().iloc[-1]) * np.sqrt(252)
            if not np.isnan(_vol) and _vol > 0:
                vol20 = _vol
            
            # 趋势强度
            ma5_v = float(close_s.tail(5).mean())
            ma10_v = float(close_s.tail(10).mean())
            ma20_v = float(close_s.tail(20).mean())
            ma_align = int(ma5_v > ma10_v) + int(ma10_v > ma20_v)
            trend_str = ma_align / 2.0
            
            vol_factor = np.clip(vol20 / 0.3, 0.7, 2.0)
            trend_factor = 1.0 + trend_str * 0.5
            stop_multi = 1.5 * trend_factor * vol_factor
            stop_multi = np.clip(stop_multi, 1.0, 3.5)
            
            # 价格效率 (趋势型 vs 震荡型)
            net_move = (close_s - close_s.shift(10)).abs()
            gross_move = close_s.diff().abs().rolling(10).sum()
            _eff = float((net_move / (gross_move + 1e-10)).iloc[-1])
            efficiency = np.clip(_eff if not np.isnan(_eff) else 0.5, 0.1, 0.9)
            
            result['stop_method'] = 'atr_dynamic'
        except Exception as exc:
            import logging as _pm_log
            _pm_log.getLogger(__name__).warning("ATR计算降级 %s: %s", stock_code, exc)
            efficiency = 0.5

        # ================================================================
        # 预估持有天数 (与 ai_engine_v2 保持一致的逻辑)
        # ================================================================
        target_multi = stop_multi * 1.8
        target_distance = atr_14 * target_multi
        daily_avg_move = atr_14 * 0.6 * (efficiency / 0.5)
        
        if daily_avg_move > 0:
            est_hold_days = np.clip(target_distance / daily_avg_move, 2, 30)
        else:
            est_hold_days = 10
        
        result['est_hold_days'] = round(float(est_hold_days), 1)
        
        # ================================================================
        # 基础ATR止损/止盈 (时间衰减前的初始值)
        # ================================================================
        base_stop = round(buy_price - atr_14 * stop_multi, 2)
        base_stop = max(base_stop, buy_price * 0.80)
        
        base_target = round(buy_price + atr_14 * target_multi, 2)
        
        result['original_stop'] = base_stop
        result['stop_price'] = base_stop
        result['target_price'] = base_target
        
        # ================================================================
        # ★ 预测有效期 — 止损随时间渐进收紧 (价格为王, 时间兜底)
        #
        # 核心原则: 永远不因为"时间到了"直接卖出
        #           而是通过收紧止损, 让价格自己触发退出
        # ================================================================
        validity_days = max(est_hold_days + 1, est_hold_days * 1.5)
        time_ratio = days_held / est_hold_days if est_hold_days > 0 else 0
        result['time_ratio'] = round(time_ratio, 2)
        
        is_profitable = current_price > buy_price
        
        # 正常追踪止损 (始终生效)
        trailing_atr_multi = max(stop_multi * 0.8, 1.0)
        trailing_stop = round(high_since - atr_14 * trailing_atr_multi, 2)
        
        if time_ratio <= 0.7:
            # ---- 预测有效期内 (0~70%): 标准持有 ----
            time_phase = 1
            phase_name = '价格主导'
            effective_stop = base_stop
            
        elif time_ratio <= 1.0:
            # ---- 接近有效期 (70~100%): 轻微收紧止损 ----
            time_phase = 2
            phase_name = '渐进收紧'
            
            tighten_progress = (time_ratio - 0.7) / 0.3  # 0→1
            
            if is_profitable:
                # 盈利: 止损向买入价方向上移 (锁定部分利润)
                profit_lock = buy_price + (current_price - buy_price) * 0.2
                effective_stop = base_stop + (profit_lock - base_stop) * tighten_progress
            else:
                # 亏损: 止损只轻微收紧 (让价格自己说话)
                tighten_target = base_stop + (current_price - base_stop) * 0.15
                effective_stop = base_stop + (tighten_target - base_stop) * tighten_progress
            
            trailing_atr_multi = max(trailing_atr_multi * (1 - tighten_progress * 0.2), 0.8)
            trailing_stop = round(high_since - atr_14 * trailing_atr_multi, 2)
            
        elif time_ratio <= 1.5:
            # ---- 超过有效期 (100~150%): 止损进一步收紧, 但仍由价格决定 ----
            time_phase = 3
            phase_name = '止损收紧'
            
            exceed_progress = (time_ratio - 1.0) / 0.5  # 0→1
            
            if is_profitable:
                # 盈利: 止损上移到至少买入价(保本), 然后继续收紧
                breakeven = buy_price
                tight_stop = current_price - atr_14 * 1.2  # 留1.2ATR空间
                effective_stop = breakeven + (tight_stop - breakeven) * exceed_progress
                effective_stop = max(effective_stop, buy_price)
            else:
                # 亏损: 止损收紧到只留1ATR空间 (让价格做最后裁判)
                tight_stop = current_price - atr_14 * 1.0
                effective_stop = base_stop + (tight_stop - base_stop) * exceed_progress
                effective_stop = max(effective_stop, base_stop)
            
            trailing_atr_multi = 0.8
            trailing_stop = round(high_since - atr_14 * trailing_atr_multi, 2)
            
        else:
            # ---- 远超有效期 (>150%): 最紧止损, 但仍由价格触发 ----
            time_phase = 4
            phase_name = '极紧止损'
            
            # 只留 0.5ATR 空间, 任何微小下跌都会触发止损退出
            effective_stop = current_price - atr_14 * 0.5
            effective_stop = max(effective_stop, base_stop)
            
            trailing_atr_multi = 0.5
            trailing_stop = round(high_since - atr_14 * trailing_atr_multi, 2)
        
        # 止损只会收紧(上移), 不会放松(下移)
        time_decay_stop = round(max(effective_stop, base_stop), 2)
        result['stop_price'] = time_decay_stop
        result['trailing_stop_price'] = trailing_stop
        result['time_phase'] = time_phase
        result['time_phase_name'] = phase_name

        # ================================================================
        # 检查卖出条件 — 优先级: ❶止损 ❷止盈 ❸追踪止损 ❹策略信号
        # 注意: 没有"时间到了直接卖"的逻辑, 全部由价格触发
        # ================================================================
        urgency = 0

        # ❶ 止损 (最高优先级, 含时间收紧后的止损)
        if current_price <= result['stop_price']:
            if time_phase >= 3:
                stop_desc = f"ATR止损(已收紧, 持有{days_held}天)"
            elif time_phase == 2:
                stop_desc = f"ATR止损(渐进收紧中)"
            else:
                stop_desc = "ATR动态止损" if result['stop_method'] == 'atr_dynamic' else "固定止损"
            result['alerts'].append(f"❶触发{stop_desc}（止损价 {result['stop_price']:.2f}）")
            urgency = max(urgency, 2)

        # ❷ 止盈
        if current_price >= result['target_price']:
            result['alerts'].append(f"❷触发止盈（目标价 {result['target_price']:.2f}）")
            urgency = max(urgency, 1)

        # ❸ 追踪止损
        if pnl_pct > 2 and current_price <= result['trailing_stop_price']:
            result['alerts'].append(
                f"❸触发追踪止损（最高 {high_since:.2f} → 回落至 {current_price:.2f}）"
            )
            urgency = max(urgency, 2)
        
        # 时间状态提示 (纯信息, 不直接触发卖出)
        if time_phase == 2 and urgency == 0:
            result['alerts'].append(
                f"📋 预测有效期已过{time_ratio:.0%}, 止损渐进收紧至{result['stop_price']:.2f}"
            )
        elif time_phase >= 3 and urgency == 0:
            result['alerts'].append(
                f"📋 已超预测有效期(持有{days_held}天/{est_hold_days:.0f}天), "
                f"止损收紧至{result['stop_price']:.2f}, 等待价格触发退出"
            )

        # ❹ 策略卖出信号（有 AI 评分时需衰减确认或多信号共振，无 AI 评分时保持原始行为）
        try:
            sigs = run_all_strategies(df)
            sell_sigs = [s for s in sigs if s['signal'] == 'sell']
            if sell_sigs:
                result['sell_signals'] = sell_sigs
                names = ', '.join([s['strategy'] for s in sell_sigs])
                n_sell_sigs = len(sell_sigs)
                has_ai_scores = ai_score_at_buy > 0 and ai_score_current > 0

                if not has_ai_scores:
                    result['alerts'].append(f"策略卖出信号（{names}）")
                    urgency = max(urgency, 1)
                else:
                    score_drop = ai_score_at_buy - ai_score_current
                    drop_threshold = getattr(config, 'AI_SELL_SCORE_DROP', 15)
                    has_score_drop = score_drop >= drop_threshold
                    has_multi_resonance = n_sell_sigs >= 2
                    score_info = f" (AI评分:{ai_score_at_buy:.0f}→{ai_score_current:.0f},降{score_drop:.0f}分)"

                    if has_multi_resonance or has_score_drop:
                        if has_multi_resonance:
                            result['alerts'].append(f"❹策略卖出信号×{n_sell_sigs}共振（{names}）{score_info}")
                        else:
                            result['alerts'].append(f"❹策略卖出+AI评分衰减（{names}）{score_info}")
                        urgency = max(urgency, 1)
                    else:
                        result['alerts'].append(
                            f"📋策略卖出信号（{names}）{score_info}[AI未确认,仅提醒]"
                        )
        except Exception as exc:
            import logging as _pm_log
            _pm_log.getLogger(__name__).warning("策略卖出信号计算失败 %s: %s", stock_code, exc)

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
        import logging as _pm_log
        _pm_log.getLogger(__name__).warning("批量获取实时价格失败，将逐只获取: %s", e)

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
