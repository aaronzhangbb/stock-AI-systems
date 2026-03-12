import pandas as pd
import streamlit as st


def render_auto_trade_result(last_result: dict, *, from_file: bool = False) -> None:
    is_plan = last_result.get("plan_only", False)
    hint = " (从上次执行恢复)" if from_file else ""
    st.markdown(
        f"<div class='signal-card' style='padding:10px 16px;'><span style='color:#5b8def;font-weight:700;'>{last_result.get('summary', '')}</span> "
        f"<span style='color:#7a869a;'>({last_result.get('timestamp', '')}){hint}</span></div>",
        unsafe_allow_html=True,
    )

    if is_plan:
        _render_plan_result(last_result)
    else:
        _render_exec_result(last_result)

    if last_result.get("skipped"):
        with st.expander(f"跳过的候选 ({len(last_result['skipped'])}个)"):
            for sk in last_result["skipped"]:
                st.markdown(f"- **{sk.get('name','')}**({sk.get('code','')}): {sk.get('reason','')}")


def _render_plan_result(result: dict) -> None:
    """渲染作战计划模式的结果（买卖建议，非已执行交易）"""
    sell_suggestions = result.get("sell_suggestions", [])
    buy_suggestions = result.get("buy_suggestions", [])

    if sell_suggestions:
        st.markdown("###### 🔴 卖出建议（明日盘中执行）")
        for ss in sell_suggestions:
            pnl_pct = ss.get("pnl_pct", 0)
            pnl_c = "#e06060" if pnl_pct >= 0 else "#5eba7d"
            phase = ss.get("time_phase", "")
            phase_tag = f' · {phase}' if phase else ''
            st.markdown(
                f"""
<div style="background:rgba(224,96,96,0.06);border-left:3px solid #f0a050;border-radius:6px;padding:8px 14px;margin-bottom:6px;">
<div style="display:flex;align-items:center;gap:8px;">
  <span style="background:#f0a050;color:#fff;border-radius:4px;padding:1px 8px;font-size:11px;font-weight:600;">建议卖出</span>
  <b style="color:#e2e8f0;">{ss['stock_name']}({ss['stock_code']})</b>
  <span style="color:#7a869a;">收盘价 ¥{ss['price']:.2f} × {ss['shares']}股</span>
  <span style="color:{pnl_c};font-weight:700;">{pnl_pct:+.1f}%</span>
</div>
<div style="color:#94a3b8;font-size:12px;margin-top:4px;">
  止损 ¥{ss.get('sell_stop', 0):.2f} · 止盈 ¥{ss.get('sell_target', 0):.2f}{phase_tag}<br>
  {ss.get('reason', '')}
</div>
</div>
""",
                unsafe_allow_html=True,
            )

    if result.get("hold_alerts"):
        st.markdown("###### 📋 持仓提醒（继续监控）")
        for ha in result["hold_alerts"]:
            pnl_pct = ha.get("pnl_pct", 0)
            pnl_c = "#e06060" if pnl_pct >= 0 else "#5eba7d"
            alert_text = "; ".join(ha.get("alerts", []))
            ai_buy = ha.get("ai_score_at_buy", 0)
            ai_now = ha.get("ai_score_current", 0)
            ai_info = f"AI评分: {ai_buy:.0f}→{ai_now:.0f}" if ai_buy > 0 and ai_now > 0 else ""
            st.markdown(
                f"""
<div style="background:rgba(91,141,239,0.06);border-left:3px solid #f0a050;border-radius:6px;padding:8px 14px;margin-bottom:6px;">
<b style="color:#e2e8f0;">{ha['stock_name']}({ha['stock_code']})</b>
<span style="color:#7a869a;margin-left:12px;">现价 ¥{ha['price']:.2f}</span>
<span style="color:{pnl_c};margin-left:12px;">{pnl_pct:+.1f}%</span>
<span style="color:#f0a050;margin-left:12px;">{ai_info}</span>
<br><span style="color:#f0a050;font-size:12px;">⚠️ {alert_text}</span>
</div>
""",
                unsafe_allow_html=True,
            )

    if buy_suggestions:
        st.markdown("###### 🟢 买入建议（明日盘中狙击）")
        for bs in buy_suggestions:
            rr = bs.get("risk_reward", 0) or 0
            st.markdown(
                f"""
<div style="background:rgba(94,186,125,0.06);border-left:3px solid #5eba7d;border-radius:6px;padding:8px 14px;margin-bottom:6px;">
<div style="display:flex;align-items:center;gap:8px;">
  <span style="background:#5eba7d;color:#fff;border-radius:4px;padding:1px 8px;font-size:11px;font-weight:600;">建议买入</span>
  <b style="color:#e2e8f0;">{bs['stock_name']}({bs['stock_code']})</b>
  <span style="color:#7a869a;">AI评分 {bs.get('ai_score', 0)}</span>
</div>
<div style="color:#94a3b8;font-size:12px;margin-top:4px;">
  买入区间 [¥{bs['buy_price']:.2f}, ¥{bs['buy_upper']:.2f}] · {bs['shares']}股 ≈ ¥{bs.get('est_cost', 0):,.0f}<br>
  止损 ¥{bs.get('sell_stop', 0):.2f} · 目标 ¥{bs.get('sell_target', 0):.2f} · 盈亏比 {rr:.1f}
</div>
</div>
""",
                unsafe_allow_html=True,
            )

    if sell_suggestions or buy_suggestions:
        st.markdown(
            '<div style="background:rgba(91,141,239,0.08);border-radius:6px;padding:8px 14px;margin-top:8px;">'
            '<span style="color:#5b8def;font-size:12px;">💡 以上为盘后分析建议，已写入作战计划。明日盘中狙击引擎将根据实时价格自动执行。</span>'
            '</div>',
            unsafe_allow_html=True,
        )


def _render_exec_result(result: dict) -> None:
    """渲染直接执行模式的结果（后向兼容）"""
    if result.get("sell_actions"):
        st.markdown("###### 卖出操作")
        for sa in result["sell_actions"]:
            pnl_c = "#e06060" if sa["pnl"] >= 0 else "#5eba7d"
            st.markdown(
                f"""
<div style="background:rgba(224,96,96,0.06);border-left:3px solid {pnl_c};border-radius:6px;padding:8px 14px;margin-bottom:6px;">
<b style="color:#e2e8f0;">{sa['stock_name']}({sa['stock_code']})</b>
<span style="color:#7a869a;margin-left:12px;">@{sa['price']:.2f} × {sa['shares']}股</span>
<span style="color:{pnl_c};margin-left:12px;font-weight:700;">盈亏 {sa['pnl']:+,.0f}({sa['pnl_pct']:+.1f}%)</span>
<br><span style="color:#94a3b8;font-size:12px;">{sa['reason']}</span>
</div>
""",
                unsafe_allow_html=True,
            )

    if result.get("hold_alerts"):
        st.markdown("###### 📋 持仓提醒（AI未确认，暂不卖出）")
        for ha in result["hold_alerts"]:
            pnl_pct = ha.get("pnl_pct", 0)
            pnl_c = "#e06060" if pnl_pct >= 0 else "#5eba7d"
            alert_text = "; ".join(ha.get("alerts", []))
            ai_buy = ha.get("ai_score_at_buy", 0)
            ai_now = ha.get("ai_score_current", 0)
            ai_info = f"AI评分: {ai_buy:.0f}→{ai_now:.0f}" if ai_buy > 0 and ai_now > 0 else ""
            st.markdown(
                f"""
<div style="background:rgba(91,141,239,0.06);border-left:3px solid #f0a050;border-radius:6px;padding:8px 14px;margin-bottom:6px;">
<b style="color:#e2e8f0;">{ha['stock_name']}({ha['stock_code']})</b>
<span style="color:#7a869a;margin-left:12px;">现价 ¥{ha['price']:.2f}</span>
<span style="color:{pnl_c};margin-left:12px;">{pnl_pct:+.1f}%</span>
<span style="color:#f0a050;margin-left:12px;">{ai_info}</span>
<br><span style="color:#f0a050;font-size:12px;">⚠️ {alert_text}</span>
</div>
""",
                unsafe_allow_html=True,
            )

    if result.get("buy_actions"):
        st.markdown("###### 买入操作")
        for ba in result["buy_actions"]:
            st.markdown(
                f"""
<div style="background:rgba(94,186,125,0.06);border-left:3px solid #5eba7d;border-radius:6px;padding:8px 14px;margin-bottom:6px;">
<b style="color:#e2e8f0;">{ba['stock_name']}({ba['stock_code']})</b>
<span style="color:#7a869a;margin-left:12px;">@{ba['price']:.2f} × {ba['shares']}股 = ¥{ba['cost']:,.0f}</span>
<br><span style="color:#94a3b8;font-size:12px;">AI评分 {ba['ai_score']} · 止损 {ba['sell_stop']:.2f} · 目标 {ba['sell_target']:.2f} · 盈亏比 {ba['risk_reward']:.1f}</span>
</div>
""",
                unsafe_allow_html=True,
            )


def build_ai_positions_df(equity: dict) -> pd.DataFrame:
    rows = []
    for p in equity.get("positions", []):
        pnl_sign = "+" if p["profit_pct"] >= 0 else ""
        rows.append(
            {
                "代码": p["code"],
                "名称": p["name"],
                "数量": f"{p['shares']}股",
                "成本": f"¥{p['avg_cost']:.2f}",
                "现价": f"¥{p['current_price']:.2f}",
                "盈亏": f"¥{p['profit']:,.0f}",
                "收益率": f"{pnl_sign}{p['profit_pct']:.1f}%",
                "市值": f"¥{p['value']:,.0f}",
            }
        )
    return pd.DataFrame(rows)
