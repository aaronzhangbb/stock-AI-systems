"""
QuantX - A股量化交易辅助系统 v5.0
AI策略 + 每日信号 + 持仓管理 + 模拟交易
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

import config
from src.data.data_fetcher import get_history_data, get_stock_name
from src.data.stock_pool import StockPool
from src.strategy.strategies import STRATEGY_REGISTRY, run_all_strategies
from src.strategy.ai_scoring import score_stock, compute_price_targets
from src.strategy.strategy_validator import validate_all_strategies, compute_composite_score
from src.strategy.scanner import MarketScanner
from src.strategy.strategy_discovery import train_model, load_learned_rules, get_discovery_summary, apply_learned_rules
from src.strategy.ai_strategies import AI_STRATEGIES, AI_COMBO_STRATEGIES, scan_stock_signals, get_strategy_summary
from src.strategy.stock_categories import get_stock_style, STYLE_STRATEGY_CONFIG, get_category_stats
from src.backtest.backtester import run_backtest
from src.trading.paper_trading import PaperTradingAccount
from src.trading.position_monitor import check_all_manual_positions, get_sell_alerts
from src.strategy.strategy_lab import StrategyLab, DIMENSIONS as LAB_DIMENSIONS

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="QuantX - A股量化系统",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== 样式 ====================
st.markdown("""
<style>
.stApp { background-color: #121620; }
section[data-testid="stSidebar"] { background-color: #161b26; border-right: 1px solid #252d3d; }
.stApp, .stMarkdown, p, span, li, label, div { color: #cbd5e1 !important; }
h1, h2, h3, h4 { color: #e8edf5 !important; }

.header-glow {
    font-size: 28px; font-weight: 700; color: #e8edf5 !important;
    border-left: 4px solid #5b8def; padding-left: 14px; margin-bottom: 2px;
}
.header-sub {
    color: #7a869a !important; font-size: 13px; letter-spacing: 1px; padding-left: 18px;
}
.signal-card {
    background: #1b2231; border: 1px solid #252d3d; border-radius: 10px;
    padding: 16px 20px; margin: 6px 0;
}
.signal-card:hover { border-color: #5b8def; }
.signal-card-buy {
    background: #221a1e; border: 1px solid #c0544e; border-radius: 10px;
    padding: 16px 20px; margin: 6px 0;
}
.signal-card-sell {
    background: #192220; border: 1px solid #3ea06c; border-radius: 10px;
    padding: 16px 20px; margin: 6px 0;
}
.signal-card-warn {
    background: #2a2218; border: 1px solid #d4a74e; border-radius: 10px;
    padding: 16px 20px; margin: 6px 0;
}
.metric-value {
    font-size: 26px; font-weight: 700; color: #e8edf5 !important;
    font-variant-numeric: tabular-nums;
}
.metric-label {
    color: #7a869a !important; font-size: 13px; letter-spacing: 0.5px;
    text-transform: uppercase; margin-bottom: 4px;
}
.tag-buy {
    background: #c0544e; color: #fff !important;
    padding: 4px 12px; border-radius: 6px; font-size: 13px; font-weight: 600;
    display: inline-block;
}
.tag-sell {
    background: #3ea06c; color: #fff !important;
    padding: 4px 12px; border-radius: 6px; font-size: 13px; font-weight: 600;
    display: inline-block;
}
.tag-strategy {
    background: rgba(91,141,239,0.12); color: #7aadff !important;
    padding: 3px 10px; border-radius: 5px; font-size: 13px;
    display: inline-block; border: 1px solid rgba(91,141,239,0.25);
}
.tag-grade-a {
    background: #1a3328; color: #5eba7d !important;
    padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;
    display: inline-block; border: 1px solid #3ea06c;
}
.tag-grade-b {
    background: #2a2818; color: #d4a74e !important;
    padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;
    display: inline-block; border: 1px solid #d4a74e;
}
.strength-bar {
    height: 7px; border-radius: 4px; background: #252d3d; overflow: hidden; margin-top: 5px;
}
.strength-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #5b8def, #8b5cf6);
}
.divider { height: 1px; background: #252d3d; margin: 24px 0; }
.stDataFrame { font-size: 14px !important; }
.stDataFrame td, .stDataFrame th { color: #cbd5e1 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background-color: #1b2231; border-radius: 8px; padding: 10px 20px;
    color: #8a95a8 !important; font-size: 14px;
}
.stTabs [aria-selected="true"] {
    background-color: #252d3d !important;
    color: #e8edf5 !important; border-bottom: 2px solid #5b8def;
}
button[kind="primary"], div.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #3b6fd4, #5b8def) !important;
    color: white !important; border: none !important;
    font-size: 15px !important; font-weight: 600 !important; border-radius: 8px !important;
}
.stButton > button:not([kind="primary"]) {
    background: #1b2231 !important; color: #cbd5e1 !important;
    border: 1px solid #333d50 !important; border-radius: 8px !important;
}
.stMultiSelect [data-baseweb="tag"] {
    background-color: #2a3548 !important; color: #cbd5e1 !important; border: none !important;
}
.stTextInput input, .stNumberInput input, .stSelectbox > div > div {
    background-color: #1b2231 !important; color: #e8edf5 !important; border-color: #333d50 !important;
}
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stRadio label, .stRadio div[role="radiogroup"] label span { color: #cbd5e1 !important; font-size: 15px !important; }
section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] { color: #cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)


# ==================== 初始化 ====================
@st.cache_resource
def get_paper_account():
    return PaperTradingAccount()

@st.cache_resource
def get_stock_pool():
    return StockPool()

@st.cache_resource
def get_scanner():
    return MarketScanner()

@st.cache_resource
def get_strategy_lab():
    return StrategyLab()

@st.cache_data(ttl=300)
def load_data(code, d):
    df = get_history_data(code, days=d)
    return df

@st.cache_data(ttl=300)
def load_stock_name(code):
    return get_stock_name(code)


# ==================== 侧边栏 ====================
st.sidebar.markdown('<p class="header-glow" style="font-size:22px;">⚡ QuantX</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="header-sub">A股量化交易系统 v5.0</p>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)

pages = ["📋 策略方案", "🔬 策略发现", "📡 每日信号", "💼 我的持仓", "🎮 模拟交易", "⚙️ 系统设置"]
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = pages[0]

page = st.sidebar.radio("功能导航", pages, index=pages.index(st.session_state["nav_page"]), key="nav_page")

pool = get_stock_pool()
pool_stats = pool.get_stats()
st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
if pool_stats['board_count'] > 0:
    tradeable_n = pool_stats.get('tradeable_count', pool_stats['stock_count'])
    excluded_n = pool_stats['stock_count'] - tradeable_n
    st.sidebar.markdown(f"""
<div class="signal-card" style="padding:10px 14px;">
<div class="metric-label">股票池状态</div>
<div style="color:#cbd5e1;font-size:14px;margin-top:4px;">
{pool_stats['board_count']} 行业 · <span style="color:#5eba7d;font-weight:600;">{tradeable_n}</span> 只可交易
</div>
<div style="color:#7a869a;font-size:12px;margin-top:2px;">排除{excluded_n}只(ST/B股/北交所) · {pool_stats['last_update']}</div>
</div>
""", unsafe_allow_html=True)
else:
    st.sidebar.warning("股票池为空 → 系统设置中同步")


# ================================================================
#   PAGE 1: 📋 策略方案
# ================================================================
if page == "📋 策略方案":
    st.markdown('<p class="header-glow">📋 AI 策略方案</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-sub">STRATEGY · 基于5008只可交易A股全量回测验证（V3无采样偏差）</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # --- 概要卡片 ---
    ai_sum = get_strategy_summary()
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="signal-card"><div class="metric-label">单策略</div><div class="metric-value">{ai_sum["total_strategies"]}</div><div style="color:#7a869a;font-size:12px;">精选{ai_sum["tiers"]["精选"]} 均衡{ai_sum["tiers"]["均衡"]} 广谱{ai_sum["tiers"]["广谱"]}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="signal-card"><div class="metric-label">组合策略</div><div class="metric-value">{ai_sum["combo_strategies"]}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="signal-card"><div class="metric-label">最佳胜率</div><div class="metric-value" style="color:#5eba7d;">{ai_sum["best_win_rate"]:.1f}%</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="signal-card"><div class="metric-label">最佳夏普</div><div class="metric-value">{ai_sum["best_sharpe"]:.2f}</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="signal-card"><div class="metric-label">持有周期</div><div class="metric-value">{ai_sum["hold_days"]}天</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # --- 核心发现（V3全量验证） ---
    st.markdown("""
<div class="signal-card" style="padding:14px 18px;border-left:4px solid #5eba7d;">
<span style="color:#5eba7d;font-weight:700;font-size:15px;">🏆 V3全量验证结论（5008只A股完整回测，无采样偏差）</span><br>
<div style="margin-top:8px;">
<span style="color:#e8edf5;font-size:14px;">最优策略：<strong style="color:#5eba7d;font-size:16px;">「布林带底部放量 + MA60斜率探底」组合</strong></span><br>
<span style="color:#e8edf5;">胜率 <strong style="color:#5eba7d;">79.0%</strong> · 夏普 <strong style="color:#5eba7d;">4.24</strong> · 每笔收益 <strong style="color:#5eba7d;">+14.92%</strong> · 4913次交易 · 覆盖2524只股票</span>
</div>
<div style="margin-top:8px;color:#8a95a8;font-size:13px;">
条件：布林带位置≤0.1 + 量比≥1.5 + MA60斜率在-8.1%~-2.1% → 持有10天卖出<br>
最优单策略：「布林带底部放量」胜率58.6%,夏普1.46,27724次交易,覆盖4862只
</div>
</div>
""", unsafe_allow_html=True)

    # --- Tab：策略列表 / 分类推荐 / 投资建议 ---
    tab_strat, tab_class, tab_advice = st.tabs(["🏆 策略列表", "📊 分类推荐", "💰 投资建议"])

    with tab_strat:
        # 单策略
        st.markdown("#### AI挖掘策略（精选 + 均衡 + 广谱）")
        tier_emoji = {'精选': '🥇', '均衡': '🥈', '广谱': '🥉'}
        tier_card = {'精选': 'signal-card-sell', '均衡': 'signal-card-warn', '广谱': 'signal-card'}

        for strat in AI_STRATEGIES:
            bt = strat['backtest']
            v2 = strat.get('v2_fullmarket', {})
            tier = strat['tier']
            emoji = tier_emoji.get(tier, '📊')
            card = tier_card.get(tier, 'signal-card')
            wr_color = "#5eba7d" if bt['win_rate'] >= 65 else ("#e0a84e" if bt['win_rate'] >= 55 else "#cbd5e1")
            ret_color = "#e06060" if bt['avg_return'] > 0 else "#5eba7d"

            # V2验证标签
            v2_badge = ''
            if v2:
                v2_wr = v2.get('win_rate', 0)
                wr_diff = v2_wr - bt['win_rate']
                if wr_diff >= -3:
                    v2_badge = '<span style="background:#1a3a2a;color:#5eba7d;padding:2px 8px;border-radius:4px;font-size:11px;margin-left:6px;">✅ V2验证通过</span>'
                elif wr_diff >= -10:
                    v2_badge = f'<span style="background:#3a2a1a;color:#e0a84e;padding:2px 8px;border-radius:4px;font-size:11px;margin-left:6px;">⚠️ V2胜率{wr_diff:+.0f}%</span>'
                else:
                    v2_badge = f'<span style="background:#3a1a1a;color:#e06060;padding:2px 8px;border-radius:4px;font-size:11px;margin-left:6px;">🔻 V2胜率{wr_diff:+.0f}%</span>'

            # V2对比行
            v2_row = ''
            if v2:
                v2_wr_color = "#5eba7d" if v2.get('win_rate', 0) >= 60 else ("#e0a84e" if v2.get('win_rate', 0) >= 55 else "#8a95a8")
                v2_note = v2.get('note', '')
                v2_row = f'''
<div style="margin-top:8px;padding:6px 10px;background:#111620;border-radius:6px;border-left:3px solid #3a4a6a;">
<div style="display:flex;gap:28px;flex-wrap:wrap;align-items:center;">
<span style="color:#5b8def;font-size:11px;font-weight:600;">V2全市场</span>
<div><span style="color:#5a6580;font-size:11px;">胜率</span> <span style="color:{v2_wr_color};font-weight:600;font-size:13px;">{v2.get("win_rate", 0):.1f}%</span></div>
<div><span style="color:#5a6580;font-size:11px;">夏普</span> <span style="color:#cbd5e1;font-size:13px;">{v2.get("sharpe", 0):.2f}</span></div>
<div><span style="color:#5a6580;font-size:11px;">收益</span> <span style="color:#cbd5e1;font-size:13px;">{v2.get("avg_return", 0):+.2f}%</span></div>
<div><span style="color:#5a6580;font-size:11px;">交易</span> <span style="color:#cbd5e1;font-size:13px;">{v2.get("trades", 0):,}次</span></div>
</div>
<div style="color:#7a869a;font-size:12px;margin-top:4px;">{v2_note}</div>
</div>'''

            st.markdown(f"""
<div class="{card}">
<div style="display:flex;justify-content:space-between;align-items:flex-start;">
<div>
<span style="font-size:16px;">{emoji}</span>
<span style="color:#e8edf5;font-weight:700;font-size:15px;margin-left:4px;">{strat['name']}</span>
<span class="tag-strategy" style="margin-left:8px;">{strat['type']}</span>
<span style="background:#252d3d;color:#8a95a8;padding:2px 8px;border-radius:4px;font-size:12px;margin-left:6px;">{tier}级</span>
{v2_badge}
</div>
<span style="color:#7a869a;font-size:12px;">持有 {strat['hold_days']} 天 · {bt['trades']} 次交易</span>
</div>
<div style="color:#8a95a8;font-size:13px;margin-top:6px;padding:4px 8px;background:#161b26;border-radius:6px;">{strat['description']}</div>
<div style="display:flex;gap:28px;margin-top:10px;flex-wrap:wrap;">
<div><span style="color:#7a869a;font-size:12px;">V1胜率</span><br><span style="color:{wr_color};font-weight:700;font-size:18px;">{bt['win_rate']:.1f}%</span></div>
<div><span style="color:#7a869a;font-size:12px;">V1夏普</span><br><span style="color:#e8edf5;font-weight:600;">{bt['sharpe']:.2f}</span></div>
<div><span style="color:#7a869a;font-size:12px;">V1收益</span><br><span style="color:{ret_color};font-weight:600;">{bt['avg_return']:+.2f}%</span></div>
<div><span style="color:#7a869a;font-size:12px;">盈亏比</span><br><span style="color:#e8edf5;font-weight:600;">{bt['profit_loss_ratio']:.2f}</span></div>
<div><span style="color:#7a869a;font-size:12px;">最大回撤</span><br><span style="color:#e06060;font-weight:600;">{f"{bt['max_drawdown']:.1f}%" if bt.get('max_drawdown') is not None else "N/A"}</span></div>
</div>
{v2_row}
</div>
""", unsafe_allow_html=True)

        # 组合策略
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("#### 🔗 AI组合策略")
        for combo in AI_COMBO_STRATEGIES:
            bt = combo['backtest']
            st.markdown(f"""
<div class="signal-card-sell">
<div style="display:flex;justify-content:space-between;align-items:center;">
<div>
<span style="font-size:16px;">🏆</span>
<span style="color:#e8edf5;font-weight:700;font-size:15px;margin-left:4px;">{combo['name']}</span>
<span style="background:#1a3328;color:#5eba7d;padding:2px 8px;border-radius:4px;font-size:12px;margin-left:8px;border:1px solid #3ea06c;">{combo['tier']}</span>
</div>
<span style="color:#7a869a;font-size:12px;">{bt['trades']} 次交易</span>
</div>
<div style="color:#8a95a8;font-size:13px;margin-top:6px;">{combo['description']}</div>
<div style="display:flex;gap:30px;margin-top:10px;">
<div><span style="color:#7a869a;font-size:12px;">胜率</span><br><span style="color:#5eba7d;font-weight:700;font-size:20px;">{bt['win_rate']:.1f}%</span></div>
<div><span style="color:#7a869a;font-size:12px;">夏普</span><br><span style="color:#e8edf5;font-weight:600;font-size:18px;">{bt['sharpe']:.2f}</span></div>
<div><span style="color:#7a869a;font-size:12px;">每笔收益</span><br><span style="color:#e06060;font-weight:600;font-size:18px;">{bt['avg_return']:+.2f}%</span></div>
<div><span style="color:#7a869a;font-size:12px;">盈亏比</span><br><span style="color:#e8edf5;font-weight:600;">{bt['profit_loss_ratio']:.2f}</span></div>
<div><span style="color:#7a869a;font-size:12px;">最大回撤</span><br><span style="color:#e06060;">{f"{bt['max_drawdown']:.1f}%" if bt.get('max_drawdown') is not None else "N/A"}</span></div>
{f'<div><span style="color:#7a869a;font-size:12px;">覆盖</span><br><span style="color:#5b8def;font-weight:600;">{combo["stocks_hit"]}只</span></div>' if combo.get('stocks_hit') else ''}
{f'<div style="margin-left:auto;"><span style="background:#1a3a2a;color:#5eba7d;padding:3px 10px;border-radius:4px;font-size:11px;">V3全量验证</span></div>' if combo.get('v3_fullmarket') else ''}
</div>
</div>
""", unsafe_allow_html=True)

    with tab_class:
        st.markdown("#### 📊 分类策略推荐（不同类型股票最适合的策略）")
        st.markdown("AI验证发现：**不同行业类型的股票，同一策略表现差异巨大**")

        for style, cfg in STYLE_STRATEGY_CONFIG.items():
            perf = cfg.get('verified_performance', {})
            is_best = '★★★' in cfg.get('note', '')
            card_cls = "signal-card-sell" if is_best else "signal-card"
            star_html = '<span style="color:#ffd700;font-size:14px;margin-left:6px;">★ 全场最佳</span>' if is_best else ''

            perf_tags = []
            for pname, pdata in perf.items():
                wr = pdata.get('win_rate', 0)
                ar = pdata.get('avg_return', 0)
                sp = pdata.get('sharpe', 0)
                perf_tags.append(
                    f'<span style="display:inline-flex;gap:8px;background:#161b26;border-radius:6px;padding:4px 10px;margin:2px 4px;">'
                    f'<span style="color:#5b8def;font-weight:600;font-size:12px;">{pname}</span>'
                    f'<span style="color:#e8edf5;font-size:12px;">胜率{wr:.0f}%</span>'
                    f'<span style="color:#e06060;font-size:12px;">{ar:+.1f}%</span>'
                    f'<span style="color:#7a869a;font-size:12px;">夏普{sp:.1f}</span></span>'
                )
            perf_html = ''.join(perf_tags)

            html = (
                f'<div class="{card_cls}" style="margin:6px 0;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<div><span style="color:#e8edf5;font-weight:700;font-size:15px;">{style}</span>'
                f'{star_html}'
                f'<span style="color:#7a869a;font-size:12px;margin-left:8px;">({cfg.get("stock_count", 0)} 只股票)</span></div>'
                f'<div style="font-size:12px;">'
                f'<span style="color:#7a869a;">止损</span> <span style="color:#e06060;">{cfg["stop_loss"]*100:.0f}%</span>'
                f'<span style="color:#7a869a;margin-left:8px;">仓位</span> <span style="color:#e8edf5;">{cfg["position_ratio"]*100:.0f}%</span>'
                f'</div></div>'
                f'<div style="color:#8a95a8;font-size:13px;margin-top:4px;">{cfg["description"]}</div>'
                f'<div style="margin-top:6px;display:flex;flex-wrap:wrap;">{perf_html}</div>'
                f'<div style="color:#5b8def;font-size:12px;margin-top:6px;">💡 {cfg["note"]}</div>'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)

    with tab_advice:
        st.markdown("#### 💰 投资方案建议")

        st.markdown("""
<div class="signal-card" style="padding:16px 20px;">
<div style="color:#e8edf5;font-weight:700;font-size:16px;margin-bottom:10px;">📌 核心操作策略</div>
<div style="color:#cbd5e1;line-height:1.8;">
<strong>1. 触发条件：</strong>股价偏离60日均线超过-13%时买入，持有10天卖出<br>
<strong>2. 精选策略优先：</strong>超跌MA30 + MA60 + 均线企稳，三者同时触发时信号最强（胜率78%+）<br>
<strong>3. 分类操作：</strong>周期股(化工/有色/钢铁)回报最高，消费白马最稳定<br>
<strong>4. 风控纪律：</strong>单票仓位不超过总资金的25-35%，严格止损（蓝筹5%、科技10%）
</div>
</div>

<div class="signal-card-warn" style="padding:16px 20px;">
<div style="color:#d4a74e;font-weight:700;font-size:16px;margin-bottom:10px;">⚠️ 风险提示</div>
<div style="color:#cbd5e1;line-height:1.8;">
· 最大回撤可达68-77%，需要严格止损纪律<br>
· 策略基于历史数据，未来市场可能发生变化<br>
· 建议先用模拟盘跟踪1-3个月再实盘操作<br>
· 不要All-in单只股票，分散持仓降低风险
</div>
</div>

<div class="signal-card" style="padding:16px 20px;">
<div style="color:#e8edf5;font-weight:700;font-size:16px;margin-bottom:10px;">📐 资金分配建议</div>
<div style="color:#cbd5e1;line-height:1.8;">
以<strong>10万元</strong>为例：<br>
· 周期制造类（D类）：3万元（30%）— 超跌MA60策略，回报最高<br>
· 大盘稳健类（A类）：2.5万元（25%）— 超跌MA60策略，最稳定<br>
· 制造装备类（E类）：2万元（20%）— 超跌MA30策略<br>
· 消费医药类（C类）：1.5万元（15%）— 超跌MA30策略<br>
· 预留现金：1万元（10%）— 应对极端超跌加仓机会
</div>
</div>
""", unsafe_allow_html=True)

        # 根据用户资金动态生成建议
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("#### 🧮 按你的资金量生成方案")
        user_capital = st.number_input("输入你的可投资资金（元）", value=100000.0, step=10000.0, format="%.0f")
        if st.button("生成配置方案", type="primary"):
            alloc = [
                ('D-周期制造', 0.30, '超跌MA60均值回归', '化工/有色/钢铁/煤炭'),
                ('A-大盘稳健', 0.25, '超跌MA60均值回归', '银行/金融/公用事业'),
                ('E-制造装备', 0.20, '超跌MA30均值回归', '机械/专用设备/汽车'),
                ('C-消费医药', 0.15, '超跌MA30均值回归', '食品饮料/医药/家电'),
                ('预留现金', 0.10, '—', '极端超跌加仓机会'),
            ]
            rows = []
            for name, ratio, strat_name, desc in alloc:
                amt = user_capital * ratio
                rows.append({'分类': name, '比例': f'{ratio*100:.0f}%', '金额': f'¥{amt:,.0f}', '策略': strat_name, '行业': desc})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ================================================================
#   PAGE 2: 🔬 策略发现
# ================================================================
elif page == "🔬 策略发现":
    st.markdown('<p class="header-glow">🔬 策略发现实验室</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-sub">STRATEGY LAB · 多维度历史数据分析 · AI策略挖掘 · 参数优化</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    lab = get_strategy_lab()

    # --- 维度选择器 ---
    st.markdown("#### 选择分析维度")
    dim_cols = st.columns(len(LAB_DIMENSIONS))
    for idx, (dim_key, dim_info) in enumerate(LAB_DIMENSIONS.items()):
        with dim_cols[idx]:
            st.markdown(f"""
<div class="signal-card" style="text-align:center;padding:12px 8px;">
<div style="font-size:24px;">{dim_info['icon']}</div>
<div style="color:#e8edf5;font-weight:600;font-size:14px;margin-top:4px;">{dim_info['name']}</div>
<div style="color:#7a869a;font-size:11px;margin-top:2px;">{dim_info['desc'][:20]}...</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Tab布局
    tab_single, tab_full, tab_optimize, tab_history = st.tabs([
        "📊 单维度分析", "🌐 全维度扫描", "🔧 参数优化", "📜 历史结果"
    ])

    # ==== Tab1: 单维度分析 ====
    with tab_single:
        col_dim, col_sample, col_hold = st.columns([2, 1, 1])
        with col_dim:
            dim_options = {v['name']: k for k, v in LAB_DIMENSIONS.items()}
            selected_dim_name = st.selectbox("分析维度", list(dim_options.keys()), key="lab_dim")
            selected_dim = dim_options[selected_dim_name]
        with col_sample:
            sample_n = st.slider("每组采样", 20, 80, 40, step=10, key="lab_sample",
                                  help="每个分组随机采样的股票数量")
        with col_hold:
            hold_d = st.slider("持有天数", 3, 20, 10, step=1, key="lab_hold")

        if st.button("🚀 开始分析", type="primary", use_container_width=True, key="lab_run"):
            bar = st.progress(0)
            status_txt = st.empty()
            def on_lab_progress(cur, total, gname, sname):
                bar.progress(min(cur / max(total, 1), 1.0))
                status_txt.text(f"[{cur}/{total}] {gname} → {sname}")

            with st.spinner(f"正在对「{selected_dim_name}」维度进行分析..."):
                result = lab.run_dimension_analysis(
                    selected_dim, max_per_group=sample_n, hold_days=hold_d,
                    progress_callback=on_lab_progress
                )
            bar.progress(1.0)
            status_txt.empty()

            if 'error' in result:
                st.error(result['error'])
            else:
                st.session_state['lab_result'] = result
                st.success(f"分析完成！{len(result.get('groups', {}))} 个分组 × {len(AI_STRATEGIES)} 个策略")
                st.rerun()

        # 展示结果
        result = st.session_state.get('lab_result')
        if result and result.get('matrix') is not None and not result['matrix'].empty:
            matrix_df = result['matrix']
            best_by_group = result.get('best_by_group', {})
            insights = result.get('insights', [])

            # --- 洞察摘要 ---
            if insights:
                insights_html = ''.join([f'<div style="margin:4px 0;color:#cbd5e1;font-size:14px;">{ins}</div>' for ins in insights])
                st.markdown(f"""
<div class="signal-card" style="padding:14px 18px;">
<div style="color:#5b8def;font-weight:700;font-size:15px;margin-bottom:8px;">💡 AI 洞察</div>
{insights_html}
</div>
""", unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # --- 最佳策略汇总 ---
            st.markdown("##### 🏆 各分组最优策略")
            best_rows = []
            for gname, binfo in best_by_group.items():
                if binfo:
                    best_rows.append({
                        '分组': gname,
                        '最优策略': binfo.get('name', '-'),
                        '胜率': f"{binfo.get('win_rate', 0):.1f}%",
                        '夏普': f"{binfo.get('sharpe', 0):.2f}",
                        '每笔收益': f"{binfo.get('avg_return', 0):.2f}%",
                        '交易数': binfo.get('trades', 0),
                    })
            if best_rows:
                st.dataframe(pd.DataFrame(best_rows), use_container_width=True, hide_index=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # --- 性能热力图 ---
            st.markdown("##### 🗺️ 策略×分组 性能矩阵")
            metric_sel = st.radio("指标", ['胜率', '夏普', '收益', '盈亏比'], horizontal=True, key="hm_metric")

            # 构建pivot表
            try:
                pivot = matrix_df.pivot_table(index='分组', columns='策略', values=metric_sel, aggfunc='first')
                if not pivot.empty:
                    # Plotly heatmap
                    colorscale = 'RdYlGn' if metric_sel in ['胜率', '夏普', '收益', '盈亏比'] else 'RdYlGn_r'
                    fig = go.Figure(data=go.Heatmap(
                        z=pivot.values,
                        x=[str(c)[:8] for c in pivot.columns],
                        y=[str(r)[:12] for r in pivot.index],
                        colorscale=colorscale,
                        text=np.round(pivot.values, 1),
                        texttemplate="%{text}",
                        textfont={"size": 11, "color": "#e8edf5"},
                        hoverongaps=False,
                        colorbar=dict(title=metric_sel, tickfont=dict(color='#8a95a8')),
                    ))
                    fig.update_layout(
                        height=max(300, len(pivot) * 35 + 80),
                        template="plotly_dark",
                        paper_bgcolor='#121620', plot_bgcolor='#161b26',
                        font=dict(color='#8a95a8', size=11),
                        margin=dict(l=120, r=20, t=10, b=60),
                        xaxis=dict(side='bottom', tickangle=-30),
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"热力图生成失败: {e}")

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # --- 详细数据表 ---
            st.markdown("##### 📋 完整回测数据")
            display_df = matrix_df.copy()
            for col in ['胜率', '收益', '回撤']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%")
            for col in ['夏普', '盈亏比']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

            # --- 分组对比图 ---
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("##### 📊 分组策略对比")
            chart_metric = st.selectbox("对比指标", ['胜率', '夏普', '收益'], key="chart_metric")
            try:
                chart_pivot = matrix_df.pivot_table(index='分组', columns='策略', values=chart_metric, aggfunc='first')
                if not chart_pivot.empty:
                    fig2 = go.Figure()
                    colors = ['#5b8def', '#e06060', '#5eba7d', '#d4a74e', '#8b5cf6',
                              '#ef5b8d', '#5bd4ef', '#efa75b', '#7def5b']
                    for i, col in enumerate(chart_pivot.columns):
                        fig2.add_trace(go.Bar(
                            name=str(col)[:10],
                            x=[str(r)[:10] for r in chart_pivot.index],
                            y=chart_pivot[col].values,
                            marker_color=colors[i % len(colors)],
                        ))
                    fig2.update_layout(
                        barmode='group', height=400,
                        template="plotly_dark",
                        paper_bgcolor='#121620', plot_bgcolor='#161b26',
                        font=dict(color='#8a95a8', size=11),
                        margin=dict(l=0, r=0, t=30, b=0),
                        yaxis_title=chart_metric,
                        legend=dict(font=dict(size=10)),
                        xaxis=dict(gridcolor='#252d3d'),
                        yaxis=dict(gridcolor='#252d3d'),
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                pass

        else:
            st.markdown("""
<div class="signal-card" style="text-align:center;padding:40px;">
<div style="font-size:48px;margin-bottom:16px;">🔬</div>
<div style="color:#cbd5e1;font-size:16px;">选择维度后点击「开始分析」</div>
<div style="color:#7a869a;font-size:14px;margin-top:8px;">
系统将对每个分组运行所有AI策略并进行回测<br>
分析时间取决于采样数量，通常3-15分钟
</div>
</div>
""", unsafe_allow_html=True)

    # ==== Tab2: 全维度扫描 ====
    with tab_full:
        st.markdown("#### 🌐 一键全维度分析")
        st.markdown("对所有维度（行业/市值/波动率/价格/趋势）同时运行策略分析，全面发现最优策略组合")

        fc1, fc2 = st.columns(2)
        with fc1:
            full_sample = st.slider("每组采样数", 15, 60, 25, step=5, key="full_sample")
        with fc2:
            full_dims = st.multiselect(
                "选择维度",
                ['industry', 'market_cap', 'volatility', 'price_range', 'trend'],
                default=['market_cap', 'volatility', 'price_range', 'trend'],
                format_func=lambda x: LAB_DIMENSIONS[x]['name'],
                key="full_dims"
            )

        if st.button("🌐 开始全维度扫描", type="primary", use_container_width=True, key="full_run"):
            bar = st.progress(0)
            status = st.empty()
            total_estimate = len(full_dims) * 5 * len(AI_STRATEGIES)
            step_count = [0]

            def on_full_prog(cur, total, gname, sname):
                step_count[0] += 1
                bar.progress(min(step_count[0] / max(total_estimate, 1), 0.99))
                status.text(f"{gname} → {sname}")

            with st.spinner("全维度扫描中..."):
                full_results = lab.run_full_analysis(
                    dimensions=full_dims, max_per_group=full_sample,
                    progress_callback=on_full_prog
                )
            bar.progress(1.0)
            status.empty()
            st.session_state['full_lab_results'] = full_results
            st.success(f"全维度扫描完成！共分析 {len(full_dims)} 个维度")
            st.rerun()

        full_results = st.session_state.get('full_lab_results')
        if full_results:
            # 汇总所有维度的洞察
            all_insights = []
            for dim_key, res in full_results.items():
                if isinstance(res, dict) and 'insights' in res:
                    dim_name = LAB_DIMENSIONS.get(dim_key, {}).get('name', dim_key)
                    for ins in res['insights']:
                        all_insights.append(f"[{dim_name}] {ins}")

            if all_insights:
                insights_html = ''.join([f'<div style="margin:3px 0;color:#cbd5e1;font-size:13px;">{ins}</div>' for ins in all_insights[:15]])
                st.markdown(f"""
<div class="signal-card" style="padding:14px 18px;">
<div style="color:#5b8def;font-weight:700;font-size:15px;margin-bottom:8px;">💡 全维度洞察汇总</div>
{insights_html}
</div>
""", unsafe_allow_html=True)

            # 每个维度一个expander
            for dim_key, res in full_results.items():
                if isinstance(res, dict) and 'matrix' in res and not res['matrix'].empty:
                    dim_name = LAB_DIMENSIONS.get(dim_key, {}).get('name', dim_key)
                    dim_icon = LAB_DIMENSIONS.get(dim_key, {}).get('icon', '📊')
                    best = res.get('best_by_group', {})
                    n_groups = len(best)

                    with st.expander(f"{dim_icon} {dim_name} — {n_groups}个分组", expanded=False):
                        # 最佳策略表
                        b_rows = []
                        for g, info in best.items():
                            if info:
                                b_rows.append({
                                    '分组': g, '最优策略': info.get('name', '-'),
                                    '胜率': f"{info.get('win_rate', 0):.1f}%",
                                    '夏普': f"{info.get('sharpe', 0):.2f}",
                                    '收益': f"{info.get('avg_return', 0):.2f}%",
                                    '交易': info.get('trades', 0),
                                })
                        if b_rows:
                            st.dataframe(pd.DataFrame(b_rows), use_container_width=True, hide_index=True)

                        # 简版热力图
                        try:
                            piv = res['matrix'].pivot_table(index='分组', columns='策略', values='胜率', aggfunc='first')
                            if not piv.empty:
                                fig_mini = go.Figure(data=go.Heatmap(
                                    z=piv.values,
                                    x=[str(c)[:8] for c in piv.columns],
                                    y=[str(r)[:12] for r in piv.index],
                                    colorscale='RdYlGn',
                                    text=np.round(piv.values, 1),
                                    texttemplate="%{text}",
                                    textfont={"size": 10, "color": "#e8edf5"},
                                ))
                                fig_mini.update_layout(
                                    height=max(200, len(piv) * 30 + 60),
                                    template="plotly_dark",
                                    paper_bgcolor='#121620', plot_bgcolor='#161b26',
                                    font=dict(color='#8a95a8', size=10),
                                    margin=dict(l=100, r=10, t=10, b=50),
                                    xaxis=dict(side='bottom', tickangle=-30),
                                )
                                st.plotly_chart(fig_mini, use_container_width=True)
                        except Exception:
                            pass

        else:
            st.info("选择维度后点击「开始全维度扫描」")

    # ==== Tab3: 参数优化 ====
    with tab_optimize:
        st.markdown("#### 🔧 策略参数优化")
        st.markdown("选择一个策略，在指定分组中自动搜索最优参数")

        opt_c1, opt_c2 = st.columns(2)
        with opt_c1:
            strat_names = {s['name']: s['id'] for s in AI_STRATEGIES}
            opt_strat_name = st.selectbox("策略", list(strat_names.keys()), key="opt_strat")
            opt_strat_id = strat_names[opt_strat_name]
        with opt_c2:
            opt_sample = st.slider("采样股票数", 20, 100, 50, step=10, key="opt_sample")

        if st.button("🔧 开始优化", type="primary", use_container_width=True, key="opt_run"):
            # 获取采样股票
            from src.data.data_cache import DataCache
            cache = DataCache()
            all_cached = cache.get_all_cached_stocks()
            if all_cached.empty:
                st.error("没有缓存数据，请先预热缓存")
            else:
                stock_list = [(r['stock_code'], r.get('stock_name', ''))
                              for _, r in all_cached.iterrows()]
                if len(stock_list) > opt_sample:
                    np.random.seed(42)
                    indices = np.random.choice(len(stock_list), opt_sample, replace=False)
                    stock_list = [stock_list[i] for i in indices]

                bar = st.progress(0)
                stxt = st.empty()
                def on_opt_prog(c, t, pname, pval):
                    bar.progress(min(c / max(t, 1), 1.0))
                    stxt.text(f"[{c}/{t}] {pname} = {pval}")

                with st.spinner("参数优化中..."):
                    opt_results = lab.optimize_parameters(
                        stock_list, base_strategy_id=opt_strat_id,
                        progress_callback=on_opt_prog
                    )
                bar.progress(1.0)
                stxt.empty()

                if opt_results:
                    st.session_state['opt_results'] = opt_results
                    st.success(f"优化完成！测试了 {len(opt_results)} 组参数")
                    st.rerun()

        opt_results = st.session_state.get('opt_results')
        if opt_results:
            st.markdown("##### 📊 参数优化结果（按夏普排序）")
            opt_rows = []
            for r in opt_results[:20]:
                opt_rows.append({
                    '参数': r['param_name'],
                    '原始值': f"{r['original_value']:.4f}",
                    '测试值': f"{r['param_value']:.4f}",
                    '胜率': f"{r['win_rate']:.1f}%",
                    '夏普': f"{r['sharpe']:.2f}",
                    '收益': f"{r['avg_return']:.2f}%",
                    '盈亏比': f"{r['profit_loss_ratio']:.2f}",
                    '交易数': r['trades'],
                })
            st.dataframe(pd.DataFrame(opt_rows), use_container_width=True, hide_index=True)

            # 参数vs指标图
            if len(opt_results) >= 3:
                fig_opt = go.Figure()
                vals = [r['param_value'] for r in opt_results]
                wrs = [r['win_rate'] for r in opt_results]
                sharpes = [r['sharpe'] for r in opt_results]

                fig_opt.add_trace(go.Scatter(
                    x=vals, y=wrs, name='胜率(%)', mode='lines+markers',
                    line=dict(color='#5eba7d', width=2), yaxis='y1',
                ))
                fig_opt.add_trace(go.Scatter(
                    x=vals, y=sharpes, name='夏普', mode='lines+markers',
                    line=dict(color='#5b8def', width=2), yaxis='y2',
                ))
                fig_opt.update_layout(
                    height=350, template="plotly_dark",
                    paper_bgcolor='#121620', plot_bgcolor='#161b26',
                    font=dict(color='#8a95a8', size=11),
                    margin=dict(l=0, r=60, t=30, b=0),
                    xaxis=dict(title='参数值', gridcolor='#252d3d'),
                    yaxis=dict(title='胜率(%)', side='left', gridcolor='#252d3d'),
                    yaxis2=dict(title='夏普', side='right', overlaying='y'),
                    legend=dict(x=0, y=1.1, orientation='h'),
                )
                st.plotly_chart(fig_opt, use_container_width=True)

            # 最优参数建议
            if opt_results:
                best_opt = opt_results[0]
                st.markdown(f"""
<div class="signal-card-sell" style="padding:14px 18px;">
<div style="color:#5eba7d;font-weight:700;font-size:15px;">✅ 最优参数建议</div>
<div style="color:#cbd5e1;font-size:14px;margin-top:6px;">
参数 <code>{best_opt['param_name']}</code> 从 {best_opt['original_value']:.4f} 调整为 <strong>{best_opt['param_value']:.4f}</strong><br>
胜率 {best_opt['win_rate']:.1f}% · 夏普 {best_opt['sharpe']:.2f} · 收益 {best_opt['avg_return']:.2f}% · {best_opt['trades']}次交易
</div>
</div>
""", unsafe_allow_html=True)

    # ==== Tab4: 历史结果 ====
    with tab_history:
        st.markdown("#### 📜 历史分析记录")
        history_df = lab.get_all_run_history()
        if not history_df.empty:
            disp_h = history_df[['dimension', 'status', 'total_groups', 'sample_per_group',
                                  'started_at', 'completed_at']].copy()
            disp_h.columns = ['维度', '状态', '分组数', '采样/组', '开始时间', '完成时间']
            disp_h['维度'] = disp_h['维度'].map(lambda x: LAB_DIMENSIONS.get(x, {}).get('name', x))
            st.dataframe(disp_h, use_container_width=True, hide_index=True)

            # 查看历史结果
            st.markdown("##### 查看历史分析结果")
            hist_dim_opts = {LAB_DIMENSIONS.get(d, {}).get('name', d): d
                            for d in history_df[history_df['status'] == 'completed']['dimension'].unique()}
            if hist_dim_opts:
                sel_hist = st.selectbox("选择维度", list(hist_dim_opts.keys()), key="hist_dim")
                if st.button("📂 加载结果", key="load_hist"):
                    cached_result = lab.get_latest_results(hist_dim_opts[sel_hist])
                    if cached_result:
                        st.session_state['lab_result'] = cached_result
                        st.success("已加载！切换到「单维度分析」标签查看")
                    else:
                        st.warning("未找到对应结果")
        else:
            st.info("暂无历史分析记录，请先运行分析")


# ================================================================
#   PAGE 3: 📡 每日信号
# ================================================================
elif page == "📡 每日信号":
    st.markdown('<p class="header-glow">📡 每日信号</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-sub">DAILY SIGNALS · AI评分推荐 + 规则策略扫描 + 邮件推送</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    scanner = get_scanner()
    account = get_paper_account()

    # ===== 顶层Tab: AI评分 / 规则信号 =====
    main_tab_ai, main_tab_rules = st.tabs(["🤖 AI评分推荐", "📡 规则策略信号"])

    # ============================================================
    # Tab 1: AI评分推荐
    # ============================================================
    with main_tab_ai:
        st.markdown("#### 🤖 AI评分推荐 (XGBoost V2)")
        st.markdown("基于88个高阶特征的机器学习模型，给全市场股票评分。**评分越高，未来5天涨>3%的概率越大。**")

        ai_scan_btn = st.button("🧠 运行AI评分扫描", type="primary", use_container_width=True)

        if ai_scan_btn:
            try:
                from src.strategy.ai_engine_v2 import AIScorer
                from src.data.data_cache import DataCache as DC2
                from src.data.stock_pool import StockPool as SP2
                ai_scorer = AIScorer()
                ai_cache = DC2()
                ai_pool = SP2()
                bar2 = st.progress(0)
                txt2 = st.empty()
                def ai_prog(c, t):
                    bar2.progress(min(c / t, 1.0))
                    txt2.text(f"AI评分: {c}/{t} ({c/t*100:.0f}%)")
                with st.spinner("AI正在评分全市场（约3分钟）..."):
                    ai_df = ai_scorer.scan_market(ai_cache, ai_pool, top_n=50, progress_callback=ai_prog)
                bar2.progress(1.0)
                txt2.empty()
                st.session_state['ai_scores'] = ai_df
                # 同时保存到文件
                import json as _json2
                output2 = {
                    'scan_date': time.strftime('%Y-%m-%d') if 'time' in dir() else '',
                    'scan_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_scored': len(ai_df),
                    'score_distribution': {
                        'above_90': int(len(ai_df[ai_df['ai_score'] >= 90])),
                        'above_80': int(len(ai_df[ai_df['ai_score'] >= 80])),
                    },
                    'top50': ai_df.head(50).to_dict(orient='records'),
                }
                score_out = os.path.join('data', 'ai_daily_scores.json')
                with open(score_out, 'w', encoding='utf-8') as f:
                    _json2.dump(output2, f, ensure_ascii=False, indent=2, default=str)
                st.success(f"AI评分完成！共评分 {len(ai_df)} 只股票")
            except Exception as e:
                st.error(f"AI评分失败: {e}")

        # 加载已有结果
        ai_df = st.session_state.get('ai_scores')
        if ai_df is None:
            try:
                import json as _json
                score_path = os.path.join('data', 'ai_daily_scores.json')
                if os.path.exists(score_path):
                    with open(score_path, 'r', encoding='utf-8') as f:
                        cached_scores = _json.load(f)
                    if cached_scores.get('top50'):
                        ai_df = pd.DataFrame(cached_scores['top50'])
                        st.info(f"📂 显示缓存结果（扫描时间: {cached_scores.get('scan_time', 'N/A')}）· 点击上方按钮更新")
            except Exception:
                pass

        if ai_df is not None and not ai_df.empty:
            # 评分分布
            c1, c2, c3, c4 = st.columns(4)
            above90 = len(ai_df[ai_df['ai_score'] >= 90]) if 'ai_score' in ai_df.columns else 0
            above80 = len(ai_df[ai_df['ai_score'] >= 80]) if 'ai_score' in ai_df.columns else 0
            above70 = len(ai_df[ai_df['ai_score'] >= 70]) if 'ai_score' in ai_df.columns else 0
            avg_score = ai_df['ai_score'].mean() if 'ai_score' in ai_df.columns else 0
            with c1:
                st.markdown(f'<div class="signal-card-buy"><div class="metric-label">90+ 强烈推荐</div><div class="metric-value" style="color:#e06060;">{above90}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="signal-card"><div class="metric-label">80+ 推荐</div><div class="metric-value">{above80}</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="signal-card"><div class="metric-label">70+ 关注</div><div class="metric-value">{above70}</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="signal-card"><div class="metric-label">平均分</div><div class="metric-value">{avg_score:.1f}</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # Top 10 详细卡片
            st.markdown("##### ⭐ Top 10 AI推荐")
            for _, row in ai_df.head(10).iterrows():
                score = row.get('ai_score', 0)
                score_color = '#e06060' if score >= 90 else ('#f0a050' if score >= 80 else '#5eba7d')
                vol20 = f"{row['volatility_20d']:.2f}" if row.get('volatility_20d') is not None else "N/A"
                bb = f"{row['bb_pos']:.3f}" if row.get('bb_pos') is not None else "N/A"
                rsi = f"{row['rsi_14']:.0f}" if row.get('rsi_14') is not None else "N/A"
                ret5 = f"{row['ret_5d']:+.1f}%" if row.get('ret_5d') is not None else "N/A"
                ma60 = f"{row['ma60_diff']:+.1f}%" if row.get('ma60_diff') is not None else "N/A"
                st.markdown(f"""
<div class="signal-card" style="margin-bottom:8px;">
<div style="display:flex;justify-content:space-between;align-items:center;">
<div>
<span style="color:#e2e8f0;font-weight:700;font-size:16px;">{row.get('stock_code','')} {row.get('stock_name','')}</span>
<span style="color:#7a869a;margin-left:12px;">{row.get('board_name','')}</span>
</div>
<div style="color:{score_color};font-weight:900;font-size:22px;">AI {score:.1f}分</div>
</div>
<div style="display:flex;gap:20px;margin-top:8px;color:#94a3b8;font-size:13px;">
<span>收盘 <b style="color:#e2e8f0;">{row.get('close',0):.2f}</b></span>
<span>波动率 <b>{vol20}</b></span>
<span>布林 <b>{bb}</b></span>
<span>RSI <b>{rsi}</b></span>
<span>5日 <b>{ret5}</b></span>
<span>MA60 <b>{ma60}</b></span>
</div>
</div>
""", unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # Top 30 表格
            top_n_show = min(30, len(ai_df))
            st.markdown(f"##### 📊 AI评分 Top {top_n_show} 完整表格")
            display_cols = ['stock_code', 'stock_name', 'board_name', 'ai_score', 'close',
                            'volatility_20d', 'bb_pos', 'rsi_14', 'ret_5d', 'vol_ratio', 'ma60_diff']
            available = [c for c in display_cols if c in ai_df.columns]
            show_df = ai_df.head(top_n_show)[available].copy()
            col_rename = {
                'stock_code': '代码', 'stock_name': '名称', 'board_name': '行业',
                'ai_score': 'AI评分', 'close': '收盘价', 'volatility_20d': '波动率',
                'bb_pos': '布林位置', 'rsi_14': 'RSI', 'ret_5d': '5日涨跌%',
                'vol_ratio': '量比', 'ma60_diff': 'MA60偏离%'
            }
            show_df = show_df.rename(columns={k: v for k, v in col_rename.items() if k in show_df.columns})

            st.dataframe(
                show_df,
                use_container_width=True,
                height=min(40 * top_n_show + 40, 800),
                column_config={
                    'AI评分': st.column_config.ProgressColumn(
                        'AI评分', min_value=0, max_value=100, format="%.1f"
                    ),
                }
            )
        else:
            st.markdown("""
<div class="signal-card" style="text-align:center;padding:40px;">
<div style="font-size:48px;margin-bottom:16px;">🤖</div>
<div style="color:#cbd5e1;font-size:16px;">点击「运行AI评分扫描」生成今日推荐</div>
<div style="color:#7a869a;font-size:14px;margin-top:8px;">
基于XGBoost GPU模型 · 88个V2高阶特征 · 测试集Top50精度96%<br>
全市场5008只股票评分，约3分钟完成
</div>
</div>
""", unsafe_allow_html=True)

    # ============================================================
    # Tab 2: 规则策略信号 (原有逻辑)
    # ============================================================
    with main_tab_rules:

        # --- 操作按钮 ---
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn1:
            scan_clicked = st.button("🚀 扫描全市场", type="primary", use_container_width=True)
        with col_btn2:
            daily_clicked = st.button("📧 执行每日任务（含邮件推送）", use_container_width=True)
        with col_btn3:
            warmup_clicked = st.button("📥 预热缓存（首次需要）", use_container_width=True)

    # 预热缓存
    if warmup_clicked:
        bar = st.progress(0)
        txt = st.empty()
        def on_warmup(c, t, n, s):
            bar.progress(min(c / t, 1.0))
            txt.text(f"下载中 [{c}/{t}] {n} ({s})")
        with st.spinner("首次下载历史数据..."):
            result = scanner.warmup_cache(days=730, progress_callback=on_warmup)
        bar.progress(1.0)
        txt.empty()
        st.success(f"缓存预热完成！成功 {result['success']}/{result['total']}")

    # 每日任务
    if daily_clicked:
        from daily_job import run_daily_job
        with st.spinner("正在执行每日闭环任务..."):
            job_result = run_daily_job()
        if job_result:
            buy_n = len(job_result.get('buy_recs', []))
            sell_n = len(job_result.get('sell_alerts', []))
            email_ok = job_result.get('email_sent', False)
            st.success(f"任务完成！推荐 {buy_n} 只 · 卖出提醒 {sell_n} 只 · 邮件{'已发送' if email_ok else '未发送'}")

    # 扫描
    if scan_clicked:
        bar = st.progress(0)
        txt = st.empty()
        def on_progress(c, t, n):
            bar.progress(min(c / t, 1.0))
            txt.text(f"扫描 [{c}/{t}] {n}")
        with st.spinner("扫描全市场..."):
            result = scanner.scan_market(days=730, progress_callback=on_progress, max_workers=2)
        bar.progress(1.0)
        txt.empty()
        st.session_state['scan_result'] = result

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # --- 展示信号 ---
    result = st.session_state.get('scan_result')
    if not result:
        today_signals = scanner.get_today_signals()
        last_task = scanner.get_latest_scan_task()
        if last_task and not today_signals.empty:
            buy_df = today_signals[today_signals['signal_type'] == 'buy']
            sell_df = today_signals[today_signals['signal_type'] == 'sell']
            buy_raw = buy_df.to_dict('records')
            sell_raw = sell_df.to_dict('records')
            result = {
                'buy_signals': buy_raw, 'sell_signals': sell_raw,
                'buy_recommendations': scanner.aggregate_recommendations(buy_raw, min_strategies=2),
                'sell_recommendations': scanner.aggregate_recommendations(sell_raw, min_strategies=2),
                'stats': {
                    'total': last_task.get('total_stocks', 0),
                    'scanned': last_task.get('scanned_stocks', 0),
                    'buy_count': last_task.get('buy_signals', 0),
                    'sell_count': last_task.get('sell_signals', 0),
                    'buy_rec_count': 0, 'sell_rec_count': 0,
                    'duration': last_task.get('duration_seconds', 0),
                    'scan_time': last_task.get('scan_time', ''),
                }
            }
            result['stats']['buy_rec_count'] = len(result['buy_recommendations'])
            result['stats']['sell_rec_count'] = len(result['sell_recommendations'])

    if result and result.get('stats'):
        stats = result['stats']
        buy_recs = result.get('buy_recommendations', [])

        # 统计
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="signal-card"><div class="metric-label">扫描股票</div><div class="metric-value">{stats.get("scanned", 0)}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="signal-card"><div class="metric-label">原始信号</div><div class="metric-value">{stats.get("buy_count", 0) + stats.get("sell_count", 0)}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="signal-card-buy"><div class="metric-label">买入推荐</div><div class="metric-value" style="color:#e06060;">{len(buy_recs)}</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="signal-card"><div class="metric-label">扫描耗时</div><div class="metric-value">{stats.get("duration", 0)}s</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # 买入信号 Tab
        tab_buy, tab_sell = st.tabs(["🔴 买入信号", "🟢 持仓卖出提醒"])

        with tab_buy:
            if buy_recs:
                df_rec = pd.DataFrame(buy_recs)
                rec_cols = {
                    'stock_code': '代码', 'stock_name': '名称', 'board_name': '行业',
                    'strategy_count': '策略数', 'validated_count': '验证通过',
                    'strategies': '策略组合', 'composite_score': '评分',
                    'buy_price': '建议买入价', 'target_price': '目标价', 'stop_price': '止损价',
                    'close_price': '最新价',
                }
                disp_cols = [c for c in rec_cols.keys() if c in df_rec.columns]
                display = df_rec[disp_cols].rename(columns=rec_cols)
                for pcol in ['建议买入价', '目标价', '止损价', '最新价']:
                    if pcol in display.columns:
                        display[pcol] = display[pcol].apply(lambda x: f"{x:.2f}" if x > 0 else "-")
                if '评分' in display.columns:
                    display['评分'] = display['评分'].apply(lambda x: f"{x:.0f}")

                st.dataframe(display, use_container_width=True, hide_index=True, height=500)
            else:
                st.info("暂无买入推荐，请先执行扫描")

        with tab_sell:
            st.markdown("**卖出提醒仅针对「我的持仓」中录入的股票**")
            manual_df = account.list_manual_positions()
            if not manual_df.empty:
                with st.spinner("检测持仓卖出条件..."):
                    monitor_results = check_all_manual_positions(account)
                alerts = get_sell_alerts(monitor_results)
                if alerts:
                    for a in alerts:
                        advice_color = "#e06060" if a['advice'] == '立即卖出' else "#d4a74e"
                        card_class = "signal-card-buy" if a['advice'] == '立即卖出' else "signal-card-warn"
                        pnl_sign = "+" if a['pnl_pct'] >= 0 else ""
                        alert_html = "".join([f"<div style='color:#8a95a8;font-size:13px;'>· {msg}</div>" for msg in a['alerts']])
                        st.markdown(f"""
<div class="{card_class}">
<div style="display:flex;justify-content:space-between;align-items:center;">
<div>
<span style="font-size:16px;font-weight:600;color:#e8edf5;">{a['stock_name']}({a['stock_code']})</span>
<span style="color:#7a869a;margin-left:12px;">买入:{a['buy_price']:.2f} → 现价:{a['current_price']:.2f}</span>
<span style="color:{'#e06060' if a['pnl_pct']>=0 else '#5eba7d'};margin-left:8px;">({pnl_sign}{a['pnl_pct']:.1f}%)</span>
</div>
<div style="color:{advice_color};font-weight:700;font-size:15px;">{a['advice']}</div>
</div>
{alert_html}
</div>
""", unsafe_allow_html=True)
                else:
                    st.success("所有持仓状态良好，暂无卖出提醒")
            else:
                st.info("请先在「我的持仓」中录入你买入的股票，系统才会监控并推送卖出信号")

    else:
        st.markdown("""
<div class="signal-card" style="text-align:center;padding:40px;">
<div style="font-size:48px;margin-bottom:16px;">📡</div>
<div style="color:#cbd5e1;font-size:16px;">点击「扫描全市场」或「执行每日任务」开始</div>
<div style="color:#7a869a;font-size:14px;margin-top:8px;">
首次使用请先点击「预热缓存」下载历史数据（约10-30分钟）<br>
之后每次扫描约5-10分钟
</div>
</div>
""", unsafe_allow_html=True)


# ================================================================
#   PAGE 4: 💼 我的持仓
# ================================================================
elif page == "💼 我的持仓":
    st.markdown('<p class="header-glow">💼 我的持仓</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-sub">MY PORTFOLIO · 资金管理 + 买入记录 + 盈亏跟踪 + 仓位建议</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    account = get_paper_account()

    # --- Tab: 账户总览 / 录入买入 / 仓位建议 ---
    tab_overview, tab_input, tab_sizing = st.tabs(["📊 账户总览", "✏️ 录入买入", "📐 仓位建议"])

    with tab_overview:
        # 获取持仓和当前价格
        manual_df = account.list_manual_positions()
        monitor_results = []
        if not manual_df.empty:
            with st.spinner("获取最新行情..."):
                monitor_results = check_all_manual_positions(account)

        # 账户概况
        total_cost = 0
        total_market = 0
        total_pnl = 0
        pos_rows = []

        for r in monitor_results:
            cost = r['buy_price'] * r.get('shares', 0) if r.get('shares', 0) > 0 else 0
            market = r['current_price'] * r.get('shares', 0) if r.get('shares', 0) > 0 and r['current_price'] > 0 else 0
            pnl = market - cost if cost > 0 else 0
            total_cost += cost
            total_market += market
            total_pnl += pnl

            pnl_sign = "+" if r['pnl_pct'] >= 0 else ""
            pos_rows.append({
                '代码': r['stock_code'],
                '名称': r['stock_name'],
                '买入价': f"{r['buy_price']:.2f}",
                '现价': f"{r['current_price']:.2f}" if r['current_price'] > 0 else "-",
                '数量': r.get('shares', 0),
                '盈亏%': f"{pnl_sign}{r['pnl_pct']:.1f}%" if r['current_price'] > 0 else "-",
                '止损价': f"{r['stop_price']:.2f}",
                '止盈价': f"{r['target_price']:.2f}",
                '建议': r['advice'],
                '买入日期': r['buy_date'],
            })

        # 概要卡片
        pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        pnl_color = "#e06060" if total_pnl >= 0 else "#5eba7d"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="signal-card"><div class="metric-label">持仓数量</div><div class="metric-value">{len(pos_rows)}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="signal-card"><div class="metric-label">总成本</div><div class="metric-value">¥{total_cost:,.0f}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="signal-card"><div class="metric-label">总市值</div><div class="metric-value">¥{total_market:,.0f}</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="signal-card"><div class="metric-label">总盈亏</div><div class="metric-value" style="color:{pnl_color};">{"+" if total_pnl>=0 else ""}{pnl_pct:.2f}%</div><div style="color:{pnl_color};font-size:14px;">¥{total_pnl:,.0f}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        if pos_rows:
            st.markdown("#### 📋 持仓明细")
            st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

            # 关闭持仓
            st.markdown("##### 关闭已卖出的持仓")
            close_col1, close_col2 = st.columns([3, 1])
            with close_col1:
                close_options = [f"{row['stock_code']} - {row['stock_name']} ({row['buy_date']})" for _, row in manual_df.iterrows()]
                close_sel = st.selectbox("选择要关闭的持仓", close_options, key="close_sel")
            with close_col2:
                if st.button("关闭此持仓", use_container_width=True):
                    parts = close_sel.split(" - ")
                    c_code = parts[0]
                    c_date = parts[1].split("(")[1].rstrip(")")
                    account.remove_manual_position(c_code, c_date)
                    st.success("已关闭")
                    st.rerun()
        else:
            st.markdown("""
<div class="signal-card" style="text-align:center;padding:40px;">
<div style="font-size:48px;margin-bottom:16px;">💼</div>
<div style="color:#cbd5e1;font-size:16px;">暂无持仓记录</div>
<div style="color:#7a869a;font-size:14px;margin-top:8px;">请到「✏️ 录入买入」标签页录入你的买入操作</div>
</div>
""", unsafe_allow_html=True)

    with tab_input:
        st.markdown("#### ✏️ 录入买入信息")
        st.markdown("根据每日信号的买入推荐，手动执行买入后在此录入")

        col_a, col_b = st.columns(2)
        with col_a:
            m_code = st.text_input("股票代码", value="", max_chars=6, key="m_code", placeholder="如 600519")
        with col_b:
            m_name = ""
            if m_code and len(m_code) == 6:
                m_name = load_stock_name(m_code)
                st.text_input("股票名称", value=m_name, disabled=True, key="m_name_disp")

        col_c, col_d, col_e = st.columns(3)
        with col_c:
            m_price = st.number_input("买入价格（元）", value=0.0, step=0.01, min_value=0.0, key="m_price")
        with col_d:
            m_shares = st.number_input("买入股数", value=100, step=100, min_value=0, key="m_shares")
        with col_e:
            m_date = st.date_input("买入日期", key="m_date")

        m_note = st.text_input("备注（可选）", key="m_note", placeholder="例如：根据超跌MA60信号买入")

        if st.button("✅ 确认录入", type="primary", use_container_width=True, key="add_manual"):
            if m_code and m_price > 0:
                r = account.add_manual_position(m_code, m_name, m_price, m_date.strftime('%Y-%m-%d'), m_shares, m_note)
                if r['success']:
                    st.success(f"已录入 {m_name}({m_code}) {m_shares}股 @ {m_price:.2f}")
                    st.rerun()
                else:
                    st.error(r['message'])
            else:
                st.warning("请填写股票代码和买入价格")

    with tab_sizing:
        st.markdown("#### 📐 智能仓位建议")
        st.markdown("输入你的总资金，系统根据AI策略和当前持仓为你推荐配置方案")

        total_fund = st.number_input("你的总投资资金（元）", value=100000.0, step=10000.0, format="%.0f", key="total_fund")
        if st.button("生成仓位配置", type="primary", key="gen_sizing"):
            # 按分类配置
            alloc = [
                ('D-周期制造', 0.30, '超跌MA60', '化工/有色/钢铁/煤炭', '胜率86%, 夏普3.7'),
                ('A-大盘稳健', 0.25, '超跌MA60', '银行/金融/公用事业', '胜率76%, 夏普3.1'),
                ('E-制造装备', 0.20, '超跌MA30', '机械/设备/汽车', '胜率75%, 夏普2.5'),
                ('C-消费医药', 0.15, '超跌MA30', '食品饮料/医药', '胜率70%, 夏普2.2'),
                ('预留现金', 0.10, '—', '极端超跌加仓', '—'),
            ]
            rows = []
            for name, ratio, strat_name, sector, perf in alloc:
                amt = total_fund * ratio
                rows.append({
                    '分类': name, '比例': f'{ratio*100:.0f}%', '配置金额': f'¥{amt:,.0f}',
                    '推荐策略': strat_name, '行业': sector, '历史表现': perf
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown(f"""
<div class="signal-card" style="padding:12px 16px;">
<div style="color:#5b8def;font-weight:600;">操作建议：</div>
<div style="color:#cbd5e1;font-size:14px;margin-top:4px;">
· 每类最多买入2-3只股票，分散风险<br>
· 单只股票不超过 ¥{total_fund*0.15:,.0f}（总资金15%）<br>
· 等待AI信号触发后再买入，不要追高<br>
· 建议持有10天，到期不管盈亏都卖出
</div>
</div>
""", unsafe_allow_html=True)


# ================================================================
#   PAGE 5: 🎮 模拟交易
# ================================================================
elif page == "🎮 模拟交易":
    st.markdown('<p class="header-glow">🎮 模拟交易</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-sub">PAPER TRADING · 虚拟资金模拟买卖 · 验证策略效果</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    account = get_paper_account()

    # 获取持仓和价格
    positions = account.get_positions()
    current_prices = {}
    if not positions.empty:
        for _, pos in positions.iterrows():
            try:
                from src.data.data_fetcher import get_realtime_price
                pi = get_realtime_price(pos['stock_code'])
                if pi:
                    current_prices[pos['stock_code']] = pi['close']
            except Exception:
                pass

    equity = account.get_total_equity(current_prices)

    # 账户概览
    pnl_color = "#e06060" if equity['total_profit'] >= 0 else "#5eba7d"
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1, "初始资金", f"¥{equity['initial_capital']:,.0f}"),
        (c2, "可用现金", f"¥{equity['cash']:,.0f}"),
        (c3, "总资产", f"¥{equity['total_equity']:,.0f}"),
        (c4, "总收益率", f"{equity['total_profit_pct']:.2f}%"),
    ]:
        with col:
            color = pnl_color if label in ['总收益率', '总资产'] else '#e8edf5'
            st.markdown(f'<div class="signal-card"><div class="metric-label">{label}</div><div class="metric-value" style="color:{color};font-size:22px;">{val}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Tab: 买卖操作 / 持仓 / 交易记录
    trade_tabs = st.tabs(["🔄 模拟买卖", "📦 模拟持仓", "📜 交易记录"])

    with trade_tabs[0]:
        stock_code_t = st.text_input("股票代码", value="", max_chars=6, key="trade_code", placeholder="输入6位代码")
        if stock_code_t and len(stock_code_t) == 6:
            stock_name_t = load_stock_name(stock_code_t)
            df_t = load_data(stock_code_t, 30)
            if not df_t.empty:
                curr_price = float(df_t.iloc[-1]['close'])
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"#### 🔴 买入 {stock_name_t}")
                    bp = st.number_input("买入价格", value=curr_price, step=0.01, key="bp")
                    bs = st.number_input("买入股数", value=100, step=100, min_value=100, key="bs")
                    if st.button("确认买入", type="primary", use_container_width=True):
                        r = account.buy(stock_code_t, stock_name_t, bp, bs)
                        st.success(r['message']) if r['success'] else st.error(r['message'])
                        if r['success']:
                            st.rerun()
                with col2:
                    st.markdown(f"#### 🟢 卖出 {stock_name_t}")
                    sp = st.number_input("卖出价格", value=curr_price, step=0.01, key="sp")
                    ss = st.number_input("卖出股数", value=100, step=100, min_value=100, key="ss")
                    if st.button("确认卖出", use_container_width=True):
                        r = account.sell(stock_code_t, stock_name_t, sp, ss)
                        st.success(r['message']) if r['success'] else st.error(r['message'])
                        if r['success']:
                            st.rerun()

    with trade_tabs[1]:
        if equity['positions']:
            pos_data = [{'代码': p['code'], '名称': p['name'], '持仓': f"{p['shares']}股",
                         '成本': f"¥{p['avg_cost']:.2f}", '现价': f"¥{p['current_price']:.2f}",
                         '盈亏': f"¥{p['profit']:,.2f}", '收益率': f"{p['profit_pct']:.2f}%"} for p in equity['positions']]
            st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
        else:
            st.info("模拟盘暂无持仓")

    with trade_tabs[2]:
        trades = account.get_trades()
        if not trades.empty:
            dt = trades[['created_at', 'stock_code', 'stock_name', 'action', 'price', 'shares', 'profit']].copy()
            dt.columns = ['时间', '代码', '名称', '操作', '价格', '数量', '盈亏']
            st.dataframe(dt, use_container_width=True, hide_index=True)

            # 收益曲线
            sell_trades = trades[trades['action'] == '卖出'].copy()
            if not sell_trades.empty:
                sell_trades = sell_trades.sort_values('created_at')
                sell_trades['cum_profit'] = sell_trades['profit'].cumsum()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sell_trades['created_at'], y=sell_trades['cum_profit'],
                    fill='tozeroy', name='累计盈亏',
                    line=dict(color='#5b8def', width=2), fillcolor='rgba(91,141,239,0.08)',
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="#7a869a", opacity=0.5)
                fig.update_layout(
                    height=300, template="plotly_dark", paper_bgcolor='#121620', plot_bgcolor='#161b26',
                    yaxis_title="累计盈亏 (¥)", margin=dict(l=0, r=0, t=10, b=0),
                    font=dict(color='#8a95a8', size=12),
                    xaxis=dict(gridcolor='#252d3d'), yaxis=dict(gridcolor='#252d3d'),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无交易记录")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if st.button("⚠️ 重置模拟账户", key="reset_acct"):
        account.reset_account()
        st.rerun()


# ================================================================
#   PAGE 6: ⚙️ 系统设置
# ================================================================
elif page == "⚙️ 系统设置":
    st.markdown('<p class="header-glow">⚙️ 系统设置</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-sub">SETTINGS · 数据同步 · 邮件配置 · 参数管理</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    tab_pool, tab_email, tab_params = st.tabs(["📦 股票池", "📧 邮件配置", "📐 参数配置"])

    with tab_pool:
        st.markdown("#### 📦 股票池管理")
        tradeable_n = pool_stats.get('tradeable_count', pool_stats['stock_count'])
        excluded_n = pool_stats['stock_count'] - tradeable_n
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="signal-card"><div class="metric-label">行业板块</div><div class="metric-value" style="font-size:20px;">{pool_stats["board_count"]}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="signal-card"><div class="metric-label">股票总数</div><div class="metric-value" style="font-size:20px;">{pool_stats["stock_count"]}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="signal-card"><div class="metric-label">可交易</div><div class="metric-value" style="font-size:20px;color:#5eba7d;">{tradeable_n}</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="signal-card"><div class="metric-label">已排除</div><div class="metric-value" style="font-size:20px;color:#e06060;">{excluded_n}</div></div>', unsafe_allow_html=True)

        # 排除详情
        t_stats = pool.get_tradeable_stats()
        if t_stats.get('excluded_detail'):
            detail_tags = ' · '.join([f'{reason}: {cnt}只' for reason, cnt in t_stats['excluded_detail'].items()])
            market_tags = ' · '.join([f'{m}: {c}只' for m, c in t_stats.get('market_distribution', {}).items()])
            st.markdown(f"""
<div class="signal-card" style="padding:10px 14px;">
<div style="color:#7a869a;font-size:13px;">
<strong style="color:#e06060;">排除原因：</strong>{detail_tags}<br>
<strong style="color:#5eba7d;">可交易分布：</strong>{market_tags}
</div>
</div>
""", unsafe_allow_html=True)

        col_sync, col_mark = st.columns(2)
        with col_sync:
            if st.button("🔄 同步股票池（申万行业分类）", type="primary", use_container_width=True):
                bar = st.progress(0)
                txt = st.empty()
                def on_p(c, t, n):
                    bar.progress(c / t)
                    txt.text(f"[{c}/{t}] {n}")
                with st.spinner("同步中..."):
                    pool.update_industry_boards(progress_callback=on_p)
                bar.progress(1.0)
                txt.text("完成！")
                st.rerun()
        with col_mark:
            if st.button("🏷️ 重新标记可交易状态", use_container_width=True):
                result = pool.mark_tradeable_status()
                st.success(f"标记完成！可交易 {result['tradeable']} 只，排除 {result['excluded']} 只")
                st.rerun()

        boards = pool.get_industry_boards()
        if not boards.empty:
            st.dataframe(boards.rename(columns={'board_code': '代码', 'board_name': '名称', 'stock_count': '个股数'}),
                         use_container_width=True, hide_index=True, height=300)

        # 显示被排除的股票
        excluded_df = pool.get_excluded_stocks()
        if not excluded_df.empty:
            with st.expander(f"查看被排除的 {len(excluded_df)} 只股票", expanded=False):
                st.dataframe(excluded_df.rename(columns={
                    'stock_code': '代码', 'stock_name': '名称',
                    'board_name': '行业', 'exclude_reason': '排除原因'
                }), use_container_width=True, hide_index=True, height=300)

    with tab_email:
        st.markdown("#### 📧 邮件推送配置")
        email_status = '✅ 已启用' if config.EMAIL_ENABLE else '❌ 未启用'
        email_to_str = ', '.join(config.EMAIL_TO)
        st.markdown(f"""
<div class="signal-card" style="padding:14px 18px;">
<div style="color:#cbd5e1;font-size:14px;">
当前配置（修改请编辑 <code>config.py</code>）：<br>
<strong>SMTP服务器：</strong>{config.SMTP_HOST}<br>
<strong>发件邮箱：</strong>{config.SMTP_USER}<br>
<strong>收件邮箱：</strong>{email_to_str}<br>
<strong>推送状态：</strong>{email_status}
</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="signal-card" style="padding:12px 16px;">
<div style="color:#5b8def;font-weight:600;">邮件内容包含：</div>
<div style="color:#cbd5e1;font-size:14px;margin-top:4px;">
· 今日AI策略买入推荐（代码、名称、策略、评分、建议价格）<br>
· 持仓卖出提醒（止损/止盈/追踪止损触发）<br>
· 每日执行「📡 每日信号 → 执行每日任务」后自动发送
</div>
</div>
""", unsafe_allow_html=True)

    with tab_params:
        st.markdown("#### 📐 系统参数")
        st.markdown(f"""
<div class="signal-card">
<div style="display:flex;gap:40px;flex-wrap:wrap;">
<div><span class="metric-label">短期均线</span><br><span style="color:#e8edf5;">MA{config.MA_SHORT}</span></div>
<div><span class="metric-label">长期均线</span><br><span style="color:#e8edf5;">MA{config.MA_LONG}</span></div>
<div><span class="metric-label">RSI周期</span><br><span style="color:#e8edf5;">{config.RSI_PERIOD}日</span></div>
<div><span class="metric-label">初始资金</span><br><span style="color:#e8edf5;">¥{config.INITIAL_CAPITAL:,.0f}</span></div>
<div><span class="metric-label">佣金</span><br><span style="color:#e8edf5;">万{config.COMMISSION_RATE*10000:.0f}</span></div>
<div><span class="metric-label">仓位</span><br><span style="color:#e8edf5;">{config.POSITION_RATIO*100:.0f}%</span></div>
<div><span class="metric-label">止损</span><br><span style="color:#e06060;">{config.STOP_LOSS_PCT*100:.0f}%</span></div>
<div><span class="metric-label">止盈</span><br><span style="color:#5eba7d;">{config.TAKE_PROFIT_PCT*100:.0f}%</span></div>
<div><span class="metric-label">追踪止损</span><br><span style="color:#d4a74e;">{config.TRAILING_STOP_PCT*100:.0f}%</span></div>
<div><span class="metric-label">推荐Top</span><br><span style="color:#e8edf5;">{config.RECOMMEND_TOP_N}</span></div>
</div>
</div>
""", unsafe_allow_html=True)

        with st.expander("🔧 高级：重新训练策略模型"):
            st.markdown("上方AI策略已经过充分验证并固化。如需基于最新数据重训练，可在此操作。")
            c_btn1, c_btn2 = st.columns([1, 1])
            with c_btn1:
                train_stocks = st.slider("训练采样股票数", 50, 500, 200, step=50)
            with c_btn2:
                force_retrain = st.checkbox("强制重新训练", value=False)
            if st.button("🚀 开始训练", type="primary", use_container_width=True, key="train_btn"):
                with st.spinner("训练中..."):
                    result = train_model(max_stocks=train_stocks, force=force_retrain)
                if 'error' in result:
                    st.error(f"训练失败: {result['error']}")
                else:
                    st.success(f"训练完成！发现 {len(result.get('learned_rules', []))} 条策略")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
<div class="signal-card" style="text-align:center;">
<div style="color:#7a869a;font-size:13px;letter-spacing:1px;">
QUANTX v5.0 · AI策略 + 每日信号 + 持仓管理 + 模拟交易 · 申万行业 · 5000+A股<br>
仅供学习研究，不构成投资建议
</div>
</div>
""", unsafe_allow_html=True)
