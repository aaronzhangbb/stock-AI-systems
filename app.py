"""
QuantX - A股量化交易辅助系统 v5.1
AI评分信号 + 策略概览 + 持仓管理 + 模拟交易
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
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
from src.strategy.strategy_discovery import train_model
from src.strategy.ai_strategies import AI_STRATEGIES, AI_COMBO_STRATEGIES, scan_stock_signals, get_strategy_summary
from src.strategy.stock_categories import get_stock_style, STYLE_STRATEGY_CONFIG, get_category_stats
from src.backtest.backtester import run_backtest
from src.trading.paper_trading import PaperTradingAccount
from src.trading.position_monitor import check_all_manual_positions, get_sell_alerts
# from src.strategy.strategy_lab import StrategyLab, DIMENSIONS as LAB_DIMENSIONS  # 已移除策略发现功能

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
span[style*="color"] { color: unset !important; }
span.eval-low { color: #4ade80 !important; font-weight: 700; }
span.eval-ok { color: #94a3b8 !important; font-weight: 700; }
span.eval-high { color: #f97316 !important; font-weight: 700; }
span.eval-danger { color: #ef4444 !important; font-weight: 700; }
span.eval-info { color: #60a5fa !important; font-weight: 700; }
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

@st.cache_data(ttl=300)
def load_data(code, d):
    df = get_history_data(code, days=d)
    return df

@st.cache_data(ttl=300)
def load_stock_name(code):
    """获取股票名称：优先从股票池查，再走 data_fetcher"""
    try:
        pool_df = get_stock_pool().get_all_stocks()
        row = pool_df[pool_df['stock_code'] == code]
        if not row.empty:
            return row.iloc[0]['stock_name']
    except Exception:
        pass
    name = get_stock_name(code)
    # 如果返回的还是代码本身，说明没查到名称
    if name == code:
        return ""
    return name


# ==================== 侧边栏 ====================
st.sidebar.markdown('<p class="header-glow" style="font-size:22px;">⚡ QuantX</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="header-sub">A股量化交易系统 v5.1</p>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)

pages = ["📡 每日信号", "💼 我的持仓", "🎮 模拟交易", "⚙️ 系统设置"]
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = pages[0]

page = st.sidebar.radio("功能导航", pages, index=pages.index(st.session_state["nav_page"]), key="nav_page")

pool = get_stock_pool()
pool_stats = pool.get_stats()
st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
if pool_stats['board_count'] > 0:
    tradeable_n = pool_stats.get('tradeable_count', pool_stats['stock_count'])
    excluded_n = pool_stats['stock_count'] - tradeable_n

    # 侧边栏: 情绪简报
    _sidebar_sentiment_path = os.path.join('data', 'market_sentiment.json')
    if os.path.exists(_sidebar_sentiment_path):
        try:
            import json as _jss
            with open(_sidebar_sentiment_path, 'r', encoding='utf-8') as _fss:
                _sd = _jss.load(_fss)
            _ss = _sd.get('sentiment_score', 50)
            _sl = _sd.get('sentiment_level', '未知')
            _st = _sd.get('fetch_time', '')[:10]
            _sc = '#4ade80' if _ss <= 35 else ('#fbbf24' if _ss <= 65 else '#ef4444')
            st.sidebar.markdown(f"""
    <div class="signal-card" style="padding:10px 14px;">
        <div class="metric-label">投资总览</div>
        <div style="color:#cbd5e1;font-size:14px;margin-top:4px;">
{pool_stats['board_count']} 行业 · <span style="color:#5eba7d;font-weight:600;">{tradeable_n}</span> 只可交易
        </div>
<div style="color:#7a869a;font-size:12px;margin-top:2px;">排除{excluded_n}只(ST/B股/北交所) · {pool_stats['last_update']}</div>
<div style="margin-top:6px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.06);">
<span style="color:#7a869a;font-size:11px;">情绪({_st}):</span>
<span style="color:{_sc};font-weight:700;font-size:13px;"> {_ss}分 ({_sl})</span>
</div>
    </div>
    """, unsafe_allow_html=True)
        except Exception:
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
#   PAGE 1: 📡 每日信号 (整合AI评分 + 规则信号 + 策略概览)
# ================================================================
if page == "📡 每日信号":
    st.markdown('<p class="header-glow">📡 每日信号</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-sub">DAILY SIGNALS · AI评分推荐 + 策略概览</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    account = get_paper_account()

    # ===== 顶层Tab: AI评分 / 策略概览 =====
    main_tab_ai, main_tab_strat = st.tabs(["🤖 AI评分推荐", "📋 策略概览"])

    # ============================================================
    # Tab 3: 策略概览（原策略方案页面精简整合）
    # ============================================================
    with main_tab_strat:
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

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # --- 策略列表 ---
        st.markdown("#### 🏆 AI挖掘策略（精选 + 均衡 + 广谱）")
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

        with st.expander("📊 分类策略推荐", expanded=False):
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

        with st.expander("💰 投资建议", expanded=False):
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

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("""
<div class="signal-card" style="padding:12px 16px;">
<div style="color:#5b8def;font-weight:600;">💡 仓位管理已迁移</div>
<div style="color:#cbd5e1;font-size:14px;margin-top:4px;">
具体的仓位配置方案请到「💼 我的持仓 → 📐 仓位建议」页面生成，<br>
系统会根据最新的AI操作清单和你的实际持仓情况，自动计算每只股票的配置金额和建议股数。
</div>
</div>
""", unsafe_allow_html=True)


    # ============================================================
    # Tab 1: AI评分推荐 (原每日信号核心功能)
    # ============================================================

    with main_tab_ai:
        st.markdown("#### 🧠 AI超级策略（三层融合）")
        st.markdown("**三层AI引擎联合评分**: XGBoost(量价特征) + 形态聚类(走势指纹) + Transformer(时序上下文)")
        st.markdown("```最终评分 = 0.5 × XGBoost + 0.3 × 形态胜率 + 0.2 × Transformer```")

        # ---- 大盘情绪 + 板块热度展示 ----
        _sentiment_path = os.path.join('data', 'market_sentiment.json')
        if os.path.exists(_sentiment_path):
            try:
                import json as _json_s
                with open(_sentiment_path, 'r', encoding='utf-8') as _f_s:
                    _sentiment_cache = _json_s.load(_f_s)
                _s_score = _sentiment_cache.get('sentiment_score', 50)
                _s_level = _sentiment_cache.get('sentiment_level', '未知')
                _s_advice = _sentiment_cache.get('sentiment_advice', '')
                _s_time = _sentiment_cache.get('fetch_time', '')
                _sub = _sentiment_cache.get('sub_scores', {})

                # 情绪颜色
                if _s_score <= 20:
                    _s_color = '#4ade80'  # 极度恐慌(绿=可能抄底)
                elif _s_score <= 35:
                    _s_color = '#60a5fa'
                elif _s_score <= 50:
                    _s_color = '#94a3b8'
                elif _s_score <= 65:
                    _s_color = '#fbbf24'
                elif _s_score <= 80:
                    _s_color = '#f97316'
                else:
                    _s_color = '#ef4444'  # 极度贪婪(红=注意风险)

                _bar_pct = max(5, min(95, _s_score))

                with st.expander(f"📊 大盘情绪: {_s_score}分 ({_s_level}) · {_s_time}", expanded=False):
                    # 情绪进度条
                    st.markdown(f"""
<div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:14px 18px;margin-bottom:10px;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
<span style="color:#e2e8f0;font-weight:700;font-size:15px;">情绪温度计</span>
<span style="color:{_s_color};font-weight:800;font-size:22px;">{_s_score}</span>
</div>
<div style="background:#1e293b;border-radius:10px;height:12px;overflow:hidden;">
<div style="background:linear-gradient(90deg, #4ade80, #fbbf24, #ef4444);width:{_bar_pct}%;height:100%;border-radius:10px;transition:width 0.5s;"></div>
</div>
<div style="display:flex;justify-content:space-between;margin-top:4px;">
<span style="color:#4ade80;font-size:10px;">恐慌(抄底)</span>
<span style="color:#94a3b8;font-size:10px;">中性</span>
<span style="color:#ef4444;font-size:10px;">贪婪(风险)</span>
</div>
<div style="color:#94a3b8;font-size:12px;margin-top:8px;">建议: {_s_advice}</div>
</div>""", unsafe_allow_html=True)

                    # 各维度分数
                    _dim_names = {'activity': '涨跌活跃', 'volume': '成交额', 'fund_flow': '主力资金', 'northbound': '北向资金', 'margin': '融资余额'}
                    _dim_cols = st.columns(5)
                    for _i, (_key, _name) in enumerate(_dim_names.items()):
                        _val = _sub.get(_key, 50)
                        _dc = '#5eba7d' if _val >= 60 else ('#f0a050' if _val >= 40 else '#e06060')
                        _dim_cols[_i].markdown(f'<div style="text-align:center;background:rgba(255,255,255,0.03);border-radius:8px;padding:8px 4px;"><div style="color:#7a869a;font-size:11px;">{_name}</div><div style="color:{_dc};font-size:20px;font-weight:800;">{_val}</div></div>', unsafe_allow_html=True)
            except Exception:
                pass

        col_ai_btn1, col_ai_btn2 = st.columns([1, 1])
        with col_ai_btn1:
            ai_scan_btn = st.button("🧠 运行AI超级策略扫描（三层融合）", type="primary", width='stretch')
        with col_ai_btn2:
            daily_push_btn = st.button("📧 执行每日任务（含邮件推送）", width='stretch')

        # 执行每日任务（含邮件推送）
        if daily_push_btn:
            from daily_job import run_daily_job
            with st.spinner("正在执行每日闭环任务（更新数据 → AI扫描 → 持仓检查 → 邮件推送）..."):
                job_result = run_daily_job()
            if job_result:
                n_ai = len(job_result.get('ai_picks', []))
                n_sell = len(job_result.get('sell_alerts', []))
                email_ok = job_result.get('email_sent', False)
                st.success(f"任务完成！AI精选 {n_ai} 只 · 卖出提醒 {n_sell} 只 · 邮件{'已发送' if email_ok else '未发送'}")
                # 刷新页面显示最新结果
                st.session_state.pop('ai_scores', None)
                st.rerun()

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

                # 先增量更新缓存，确保使用最新K线数据
                txt2.text("[0/3] 增量更新缓存（确保数据最新）...")
                with st.spinner("更新缓存中..."):
                    from src.strategy.scanner import MarketScanner as MS2
                    _scanner2 = MS2()
                    _scanner2.warmup_cache(days=730)
                bar2.progress(0.1)
                # 更新后重新加载缓存
                ai_cache = DC2()

                def ai_prog(c, t):
                    bar2.progress(0.1 + min(c / t * 0.35, 0.35))
                    txt2.text(f"[1/3] XGBoost评分: {c}/{t} ({c/t*100:.0f}%)")
                with st.spinner("第1步: XGBoost评分全市场..."):
                    ai_df = ai_scorer.scan_market(ai_cache, ai_pool, top_n=100, progress_callback=ai_prog)
                bar2.progress(0.50)

                # 第二步: 形态匹配
                txt2.text("[2/3] 形态匹配中...")
                pattern_scores = {}
                try:
                    from src.strategy.pattern_engine import PatternEngine
                    pe_model_path = os.path.join('data', 'pattern_engine.pkl')
                    if os.path.exists(pe_model_path):
                        pe = PatternEngine.load(pe_model_path)
                        top_codes = ai_df['stock_code'].tolist() if not ai_df.empty else []
                        matched = 0
                        for code in top_codes:
                            try:
                                kdf = ai_cache.load_kline(code)
                                if kdf is not None:
                                    pr = pe.predict_single(kdf)
                                    if pr and pr['is_valid']:
                                        pattern_scores[code] = pr
                                        matched += 1
                            except Exception:
                                pass
                        txt2.text(f"[2/3] 形态匹配完成: {matched}/{len(top_codes)} 只")
                except Exception:
                    pass
                bar2.progress(0.70)

                # 第三步: Transformer时序评分
                txt2.text("[3/3] Transformer时序评分中...")
                tf_scores = {}
                try:
                    from src.strategy.transformer_engine import StockTransformer
                    tf_model_path = os.path.join('data', 'transformer_model.pt')
                    if os.path.exists(tf_model_path):
                        tf_engine = StockTransformer.load(tf_model_path)
                        top_codes = ai_df['stock_code'].tolist() if not ai_df.empty else []
                        tf_matched = 0
                        for code in top_codes:
                            try:
                                kdf = ai_cache.load_kline(code)
                                if kdf is not None:
                                    ts = tf_engine.predict_single(kdf)
                                    if ts is not None:
                                        tf_scores[code] = ts
                                        tf_matched += 1
                            except Exception:
                                pass
                        txt2.text(f"[3/3] Transformer完成: {tf_matched}/{len(top_codes)} 只")
                except Exception:
                    pass
                bar2.progress(0.9)

                # 三层融合: final = 0.5 × XGBoost + 0.3 × 形态胜率 + 0.2 × Transformer
                if not ai_df.empty:
                    final_scores = []
                    pat_win_rates = []
                    pat_descs = []
                    pat_confs = []
                    tf_score_list = []
                    for _, row in ai_df.iterrows():
                        code = row['stock_code']
                        xgb_score = row['ai_score']
                        
                        # 形态分
                        pr = pattern_scores.get(code)
                        if pr:
                            pat_wr = pr['win_rate']
                            pat_win_rates.append(pat_wr)
                            pat_descs.append(pr.get('pattern_desc', ''))
                            pat_confs.append(pr['confidence'])
                        else:
                            pat_wr = 52.6  # 平均胜率
                            pat_win_rates.append(None)
                            pat_descs.append('')
                            pat_confs.append(None)
                        
                        # Transformer分
                        ts = tf_scores.get(code)
                        if ts is not None:
                            tf_s = ts
                            tf_score_list.append(tf_s)
                        else:
                            tf_s = 52.9  # 平均概率
                            tf_score_list.append(None)
                        
                        # 超级策略融合
                        fused = xgb_score * 0.5 + pat_wr * 0.3 + tf_s * 0.2
                        final_scores.append(round(fused, 1))
                    
                    ai_df['pattern_win_rate'] = pat_win_rates
                    ai_df['pattern_desc'] = pat_descs
                    ai_df['pattern_confidence'] = pat_confs
                    ai_df['transformer_score'] = tf_score_list
                    ai_df['final_score'] = final_scores
                    ai_df = ai_df.sort_values('final_score', ascending=False).reset_index(drop=True)
                
                bar2.progress(1.0)
                txt2.empty()
                st.session_state['ai_scores'] = ai_df

                # 保存到文件
                import json as _json2
                output2 = {
                    'scan_date': time.strftime('%Y-%m-%d') if 'time' in dir() else '',
                    'scan_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_scored': len(ai_df),
                    'pattern_matched': len(pattern_scores),
                    'transformer_matched': len(tf_scores),
                    'fusion': '0.5*XGBoost + 0.3*Pattern + 0.2*Transformer',
                    'score_distribution': {
                        'above_90': int(len(ai_df[ai_df['final_score'] >= 90])) if 'final_score' in ai_df.columns else 0,
                        'above_80': int(len(ai_df[ai_df['final_score'] >= 80])) if 'final_score' in ai_df.columns else 0,
                    },
                    'top50': ai_df.head(50).to_dict(orient='records'),
                }
                score_out = os.path.join('data', 'ai_daily_scores.json')
                with open(score_out, 'w', encoding='utf-8') as f:
                    _json2.dump(output2, f, ensure_ascii=False, indent=2, default=str)
                n_pat = len(pattern_scores)
                n_tf = len(tf_scores)
                st.success(f"AI超级策略扫描完成！XGB {len(ai_df)}只 + 形态 {n_pat}只 + Transformer {n_tf}只")
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
            # 评分分布 (使用融合分 or AI评分)
            score_col = 'final_score' if 'final_score' in ai_df.columns else 'ai_score'
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            above90 = len(ai_df[ai_df[score_col] >= 90]) if score_col in ai_df.columns else 0
            above80 = len(ai_df[ai_df[score_col] >= 80]) if score_col in ai_df.columns else 0
            above70 = len(ai_df[ai_df[score_col] >= 70]) if score_col in ai_df.columns else 0
            avg_score = ai_df[score_col].mean() if score_col in ai_df.columns else 0
            n_pattern = len(ai_df[ai_df['pattern_win_rate'].notna()]) if 'pattern_win_rate' in ai_df.columns else 0
            n_tf = len(ai_df[ai_df['transformer_score'].notna()]) if 'transformer_score' in ai_df.columns else 0
            with c1:
                st.markdown(f'<div class="signal-card-buy"><div class="metric-label">90+ 强烈推荐</div><div class="metric-value" style="color:#e06060;">{above90}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="signal-card"><div class="metric-label">80+ 推荐</div><div class="metric-value">{above80}</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="signal-card"><div class="metric-label">70+ 关注</div><div class="metric-value">{above70}</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="signal-card"><div class="metric-label">超级均分</div><div class="metric-value">{avg_score:.1f}</div></div>', unsafe_allow_html=True)
            with c5:
                st.markdown(f'<div class="signal-card"><div class="metric-label">形态匹配</div><div class="metric-value">{n_pattern}</div></div>', unsafe_allow_html=True)
            with c6:
                st.markdown(f'<div class="signal-card"><div class="metric-label">TF匹配</div><div class="metric-value">{n_tf}</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # ============================================================
            # 今日操作清单 (极简版)
            # ============================================================
            score_col_s = 'final_score' if 'final_score' in ai_df.columns else 'ai_score'
            strong_picks = ai_df[ai_df[score_col_s] >= 85].head(5)
            
            if not strong_picks.empty:
                st.markdown("##### 📋 今日操作清单")
                st.markdown('<div style="color:#94a3b8;font-size:13px;margin-bottom:12px;">只看这里就够了 — AI筛选出最值得关注的股票，直接给出操作价格</div>', unsafe_allow_html=True)

                # 预加载基本面数据(带缓存) + 行业估值基准
                _pick_codes = strong_picks['stock_code'].tolist()
                _fund_map = {}
                _benchmarks = {}
                try:
                    from src.data.fundamental import get_valuation_for_stocks, evaluate_pe, evaluate_pb, get_industry_benchmarks
                    _fund_map = get_valuation_for_stocks(_pick_codes)
                    _benchmarks = get_industry_benchmarks()
                except Exception:
                    pass

                for idx_pick, row_pick in strong_picks.iterrows():
                    p_code = row_pick.get('stock_code', '')
                    p_name = row_pick.get('stock_name', '')
                    p_close = row_pick.get('close', 0)
                    p_final = row_pick.get('final_score', row_pick.get('ai_score', 0))
                    p_buy = row_pick.get('buy_price')
                    p_buy_up = row_pick.get('buy_upper')
                    p_sell_tgt = row_pick.get('sell_target')
                    p_sell_stp = row_pick.get('sell_stop')
                    p_hold = row_pick.get('hold_days', '5天')
                    p_rr = row_pick.get('risk_reward')
                    p_pos = row_pick.get('position_pct', '10%')
                    
                    # 格式化
                    buy_s = f"{p_buy:.2f}" if pd.notna(p_buy) else f"{p_close:.2f}"
                    buy_up_s = f"{p_buy_up:.2f}" if pd.notna(p_buy_up) else "N/A"
                    tgt_s = f"{p_sell_tgt:.2f}" if pd.notna(p_sell_tgt) else "N/A"
                    stp_s = f"{p_sell_stp:.2f}" if pd.notna(p_sell_stp) else "N/A"
                    tgt_pct_s = f"+{(p_sell_tgt/p_close-1)*100:.1f}%" if pd.notna(p_sell_tgt) and p_close > 0 else ""
                    stp_pct_s = f"-{(1-p_sell_stp/p_close)*100:.1f}%" if pd.notna(p_sell_stp) and p_close > 0 else ""
                    rr_s = f"{p_rr:.1f}:1" if pd.notna(p_rr) else "N/A"
                    
                    # 星级
                    if p_final >= 90:
                        stars = "⭐⭐⭐"
                        level = "强烈推荐"
                        level_color = "#e06060"
                    elif p_final >= 85:
                        stars = "⭐⭐"
                        level = "推荐"
                        level_color = "#f0a050"
                    else:
                        stars = "⭐"
                        level = "关注"
                        level_color = "#5eba7d"
                    
                    # 退出规则 — 价格为王, 时间兜底
                    exit_rules = row_pick.get('exit_rules', '')
                    validity_d = row_pick.get('validity_days')
                    est_d = row_pick.get('est_hold_days')
                    
                    if exit_rules:
                        expire_action = (
                            f"<b>退出优先级</b> (价格为王, 时间兜底):<br>"
                            f"&nbsp;❶ <b style='color:#e06060;'>止损</b>: 跌破止损价 → 无条件卖出 (最高优先级)<br>"
                            f"&nbsp;❷ <b style='color:#5eba7d;'>止盈</b>: 触及目标价 → 卖出锁利<br>"
                            f"&nbsp;❸ <b style='color:#f0a050;'>追踪止损</b>: 从高点回撤超1ATR → 保护利润<br>"
                            f"&nbsp;❹ <b style='color:#94a3b8;'>超有效期</b>: 超{validity_d}天以上都没触发 → 止损自动收紧至0.5ATR, 让价格做最终裁判"
                        )
                    else:
                        expire_action = f"❶止损 ❷止盈 ❸追踪止损 ❹超有效期止损收紧"

                    # 基本面标签(含行业相对估值)
                    _fv = _fund_map.get(p_code, {})
                    _f_pe = _fv.get('pe')
                    _f_pb = _fv.get('pb')
                    _f_mv = _fv.get('total_mv')
                    _board = row_pick.get('board_name', '')
                    _fund_parts = []
                    def _eval_css_class(color_hex):
                        """颜色hex转CSS class"""
                        _map = {'#4ade80': 'eval-low', '#94a3b8': 'eval-ok', '#f97316': 'eval-high', '#ef4444': 'eval-danger', '#60a5fa': 'eval-info'}
                        return _map.get(color_hex, 'eval-ok')

                    if _f_pe is not None:
                        try:
                            _pe_eval, _pe_clr = evaluate_pe(_f_pe, _board, _benchmarks)
                            _pe_cls = _eval_css_class(_pe_clr)
                            if _f_pe < 0:
                                _fund_parts.append(f'PE <span class="{_pe_cls}">亏损</span>')
                            else:
                                _fund_parts.append(f'PE {_f_pe:.1f}<span class="{_pe_cls}">({_pe_eval})</span>')
                        except Exception:
                            _fund_parts.append(f"PE {'亏损' if _f_pe < 0 else f'{_f_pe:.1f}'}")
                    if _f_pb is not None:
                        try:
                            _pb_eval, _pb_clr = evaluate_pb(_f_pb, _board, _benchmarks)
                            _pb_cls = _eval_css_class(_pb_clr)
                            _fund_parts.append(f'PB {_f_pb:.2f}<span class="{_pb_cls}">({_pb_eval})</span>')
                        except Exception:
                            _fund_parts.append(f"PB {_f_pb:.2f}")
                    if _f_mv is not None:
                        if _f_mv >= 10000:
                            _fund_parts.append(f"市值{_f_mv/10000:.0f}万亿")
                        elif _f_mv >= 100:
                            _fund_parts.append(f"市值{_f_mv:.0f}亿")
                        else:
                            _fund_parts.append(f"市值{_f_mv:.1f}亿")
                    _fund_label = " | ".join(_fund_parts) if _fund_parts else ""
                    _fund_html = f'<div style="color:#94a3b8;font-size:11px;margin-top:6px;">{_fund_label}</div>' if _fund_label else ""

                    st.markdown(f"""
<div style="background:linear-gradient(135deg, rgba(30,40,60,0.95), rgba(20,30,50,0.95));border:1px solid rgba(94,186,125,0.25);border-radius:12px;padding:16px 20px;margin-bottom:10px;">
<div style="display:flex;justify-content:space-between;align-items:center;">
<div>
<span style="font-size:18px;font-weight:800;color:#e2e8f0;">{p_code} {p_name}</span>
<span style="color:{level_color};font-size:13px;margin-left:10px;font-weight:700;">{stars} {level}</span>
<span style="color:#7a869a;font-size:12px;margin-left:8px;">综合{p_final:.0f}分</span>
{_fund_html}
</div>
<div style="color:#e2e8f0;font-size:13px;">当前价 <b style="font-size:16px;">{p_close:.2f}</b></div>
</div>

<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:16px;margin-top:14px;">
<div style="text-align:center;background:rgba(46,139,87,0.15);border-radius:8px;padding:10px;">
<div style="color:#7a869a;font-size:11px;">买入价</div>
<div style="color:#5eba7d;font-size:20px;font-weight:900;">{buy_s}</div>
<div style="color:#7a869a;font-size:11px;">最高 {buy_up_s}</div>
</div>
<div style="text-align:center;background:rgba(224,96,96,0.12);border-radius:8px;padding:10px;">
<div style="color:#7a869a;font-size:11px;">止盈卖出</div>
<div style="color:#e06060;font-size:20px;font-weight:900;">{tgt_s}</div>
<div style="color:#f0a050;font-size:11px;">{tgt_pct_s}</div>
</div>
<div style="text-align:center;background:rgba(224,96,96,0.08);border-radius:8px;padding:10px;">
<div style="color:#7a869a;font-size:11px;">止损卖出</div>
<div style="color:#94a3b8;font-size:20px;font-weight:900;">{stp_s}</div>
<div style="color:#94a3b8;font-size:11px;">{stp_pct_s}</div>
</div>
<div style="text-align:center;background:rgba(255,255,255,0.04);border-radius:8px;padding:10px;">
<div style="color:#7a869a;font-size:11px;">仓位/有效期</div>
<div style="color:#e2e8f0;font-size:16px;font-weight:700;">{p_pos}</div>
<div style="color:#7a869a;font-size:11px;">有效期 {p_hold}</div>
</div>
</div>

<div style="margin-top:10px;padding:8px 12px;background:rgba(240,160,80,0.08);border-radius:6px;border-left:3px solid #f0a050;">
<div style="color:#f0a050;font-size:12px;font-weight:700;">⏰ 到期未达标怎么办？</div>
<div style="color:#cbd5e1;font-size:12px;margin-top:2px;">{expire_action}</div>
</div>
</div>""", unsafe_allow_html=True)
                
                # 操作规则说明
                st.markdown("""
<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:14px 18px;margin-top:4px;margin-bottom:16px;">
<div style="color:#e2e8f0;font-weight:700;font-size:14px;margin-bottom:8px;">📖 操作规则（必读）</div>
<div style="color:#94a3b8;font-size:13px;line-height:1.8;">
<b style="color:#e06060;">❶ 止损(最高优先):</b> 股价跌破止损价 → 无条件卖出, 这是铁律<br>
<b style="color:#5eba7d;">❷ 止盈:</b> 股价触及止盈目标 → 卖出锁利<br>
<b style="color:#f0a050;">❸ 追踪止损:</b> 盈利后从高点回落超1ATR → 卖出保护利润<br>
<b style="color:#94a3b8;">❹ 有效期兜底:</b> 以上都没触发? 超过有效期后止损自动收紧, 让价格做最终裁判<br>
<b style="color:#5eba7d;">买入:</b> 在"买入价"附近挂限价单, 不追高超过"最高可接受价"<br>
<b style="color:#7a869a;">仓位:</b> 单只股票不超过建议仓位(Kelly公式), 总持仓不超过3~5只
</div>
</div>""", unsafe_allow_html=True)
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # Top 10 详细卡片（含买卖建议 + 形态匹配）
            with st.expander("📊 Top 10 详细分析（展开查看完整技术指标）", expanded=False):
                for _, row in ai_df.head(10).iterrows():
                    final = row.get('final_score', row.get('ai_score', 0))
                    score = row.get('ai_score', 0)
                    score_color = '#e06060' if final >= 90 else ('#f0a050' if final >= 80 else '#5eba7d')
                    close_val = row.get('close', 0)
                    vol20 = f"{row['volatility_20d']:.2f}" if pd.notna(row.get('volatility_20d')) else "N/A"
                    bb = f"{row['bb_pos']:.3f}" if pd.notna(row.get('bb_pos')) else "N/A"
                    rsi = f"{row['rsi_14']:.0f}" if pd.notna(row.get('rsi_14')) else "N/A"
                    ret5 = f"{row['ret_5d']:+.1f}%" if pd.notna(row.get('ret_5d')) else "N/A"
                    ma60 = f"{row['ma60_diff']:+.1f}%" if pd.notna(row.get('ma60_diff')) else "N/A"

                    # 买卖建议字段
                    buy_p = row.get('buy_price')
                    buy_up = row.get('buy_upper')
                    buy_cond = row.get('buy_condition', '')
                    buy_time = row.get('buy_timing', '')
                    sell_tgt = row.get('sell_target')
                    sell_stp = row.get('sell_stop')
                    hold_d = row.get('hold_days', '5~10天')
                    rr = row.get('risk_reward')
                    pos_pct = row.get('position_pct', '10%')
                    pos_adv = row.get('position_advice', '')

                    # 格式化
                    buy_str = f"{buy_p:.2f}" if pd.notna(buy_p) else "N/A"
                    buy_up_str = f"{buy_up:.2f}" if pd.notna(buy_up) else "N/A"
                    sell_tgt_str = f"{sell_tgt:.2f}" if pd.notna(sell_tgt) else "N/A"
                    sell_stp_str = f"{sell_stp:.2f}" if pd.notna(sell_stp) else "N/A"
                    rr_str = f"{rr:.1f}" if pd.notna(rr) else "N/A"
                    tgt_pct = f"+{(sell_tgt / close_val - 1) * 100:.1f}%" if pd.notna(sell_tgt) and close_val > 0 else ""
                    stp_pct = f"-{(1 - sell_stp / close_val) * 100:.1f}%" if pd.notna(sell_stp) and close_val > 0 else ""

                    # 盈亏比颜色
                    rr_color = '#5eba7d' if pd.notna(rr) and rr >= 2 else ('#f0a050' if pd.notna(rr) and rr >= 1.5 else '#e06060')

                    # 形态信息
                    pat_wr = row.get('pattern_win_rate')
                    pat_desc = row.get('pattern_desc', '')
                    pat_conf = row.get('pattern_confidence')
                    has_pattern = pd.notna(pat_wr) if pat_wr is not None else False
                    pat_wr_str = f"{pat_wr:.1f}%" if has_pattern else "N/A"
                    tf_sc = row.get('transformer_score')
                    has_tf = pd.notna(tf_sc) if tf_sc is not None else False
                    tf_str = f"{tf_sc:.1f}" if has_tf else "N/A"
                    pat_badge = ""
                    if has_pattern and pat_wr is not None:
                        if pat_wr >= 80:
                            pat_badge = '<span style="background:#1a3a1a;color:#5eba7d;padding:2px 8px;border-radius:4px;font-size:11px;margin-left:8px;">🎯 高胜率形态</span>'
                        elif pat_wr >= 60:
                            pat_badge = '<span style="background:#2a2a1a;color:#e0a84e;padding:2px 8px;border-radius:4px;font-size:11px;margin-left:8px;">📊 中胜率形态</span>'

                    st.markdown(f"""
<div class="signal-card" style="margin-bottom:12px;padding:16px 20px;">
<!-- 标题行 -->
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
<div>
<span style="color:#e2e8f0;font-weight:700;font-size:17px;">{row.get('stock_code','')} {row.get('stock_name','')}</span>
<span style="color:#7a869a;margin-left:12px;font-size:13px;">{row.get('board_name','')}</span>
{pat_badge}
</div>
<div style="text-align:right;">
<div style="color:{score_color};font-weight:900;font-size:22px;">综合 {final:.1f}分</div>
<div style="color:#7a869a;font-size:11px;">XGB {score:.1f} · 形态 {pat_wr_str} · TF {tf_str}</div>
</div>
</div>
<!-- 核心指标行 -->
<div style="display:flex;gap:18px;color:#94a3b8;font-size:13px;margin-bottom:8px;">
<span>收盘 <b style="color:#e2e8f0;">{close_val:.2f}</b></span>
<span>波动率 <b>{vol20}</b></span>
<span>布林 <b>{bb}</b></span>
<span>RSI <b>{rsi}</b></span>
<span>5日 <b>{ret5}</b></span>
<span>MA60 <b>{ma60}</b></span>
</div>
{"<div style='color:#7a869a;font-size:12px;margin-bottom:8px;'>🔍 形态: " + pat_desc + "</div>" if pat_desc else ""}
<!-- 买卖建议区 -->
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
<!-- 买入建议 -->
<div style="background:rgba(46,139,87,0.12);border:1px solid rgba(94,186,125,0.3);border-radius:8px;padding:12px;">
<div style="color:#5eba7d;font-weight:700;font-size:14px;margin-bottom:8px;">📥 买入建议</div>
<div style="display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-size:13px;">
<span style="color:#7a869a;">建议买入价</span>
<span style="color:#5eba7d;font-weight:700;font-size:15px;">{buy_str}</span>
<span style="color:#7a869a;">最高可接受</span>
<span style="color:#94a3b8;">{buy_up_str}</span>
<span style="color:#7a869a;">买入条件</span>
<span style="color:#e2e8f0;font-size:12px;">{buy_cond}</span>
<span style="color:#7a869a;">时机建议</span>
<span style="color:#cbd5e1;font-size:12px;">{buy_time}</span>
</div>
</div>
<!-- 卖出建议 -->
<div style="background:rgba(224,96,96,0.10);border:1px solid rgba(224,96,96,0.3);border-radius:8px;padding:12px;">
<div style="color:#e06060;font-weight:700;font-size:14px;margin-bottom:8px;">📤 卖出建议</div>
<div style="display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-size:13px;">
<span style="color:#7a869a;">止盈目标</span>
<span style="color:#e06060;font-weight:700;font-size:15px;">{sell_tgt_str} <span style="font-size:12px;color:#f0a050;">({tgt_pct})</span></span>
<span style="color:#7a869a;">止损价格</span>
<span style="color:#94a3b8;">{sell_stp_str} <span style="font-size:12px;">({stp_pct})</span></span>
<span style="color:#7a869a;">持有周期</span>
<span style="color:#e2e8f0;">{hold_d}</span>
<span style="color:#7a869a;">盈亏比</span>
<span style="color:{rr_color};font-weight:700;">{rr_str} : 1</span>
</div>
</div>
</div>
<!-- 底部仓位建议 -->
<div style="display:flex;justify-content:space-between;align-items:center;margin-top:10px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.06);">
<span style="color:#7a869a;font-size:12px;">💰 建议仓位: <b style="color:#e2e8f0;">{pos_pct}</b> · {pos_adv}</span>
<span style="color:#7a869a;font-size:12px;">⚠️ 以上为AI模型辅助建议，不构成投资建议，请结合自身判断</span>
</div>
</div>""", unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # Top 30 表格（含融合评分+形态+买卖价格）
            top_n_show = min(30, len(ai_df))
            st.markdown(f"##### 📊 AI综合 Top {top_n_show} 完整表格")
            display_cols = ['stock_code', 'stock_name', 'board_name', 
                            'final_score', 'ai_score', 'pattern_win_rate', 'transformer_score', 'pattern_desc',
                            'close', 'buy_price', 'sell_target', 'sell_stop', 'risk_reward', 'hold_days',
                            'volatility_20d', 'bb_pos', 'rsi_14', 'ret_5d', 'vol_ratio', 'ma60_diff']
            available = [c for c in display_cols if c in ai_df.columns]
            show_df = ai_df.head(top_n_show)[available].copy()
            col_rename = {
                'stock_code': '代码', 'stock_name': '名称', 'board_name': '行业',
                'final_score': '超级评分', 'ai_score': 'XGB评分', 
                'pattern_win_rate': '形态胜率%', 'transformer_score': 'TF评分',
                'pattern_desc': '形态描述',
                'close': '收盘价',
                'buy_price': '建议买入', 'sell_target': '止盈目标', 'sell_stop': '止损价',
                'risk_reward': '盈亏比', 'hold_days': '持有周期',
                'volatility_20d': '波动率', 'bb_pos': '布林位置', 'rsi_14': 'RSI',
                'ret_5d': '5日涨跌%', 'vol_ratio': '量比', 'ma60_diff': 'MA60偏离%'
            }
            show_df = show_df.rename(columns={k: v for k, v in col_rename.items() if k in show_df.columns})

            col_cfg = {}
            if '超级评分' in show_df.columns:
                col_cfg['超级评分'] = st.column_config.ProgressColumn(
                    '超级评分', min_value=0, max_value=100, format="%.1f"
                )
            if 'XGB评分' in show_df.columns:
                col_cfg['XGB评分'] = st.column_config.ProgressColumn(
                    'XGB评分', min_value=0, max_value=100, format="%.1f"
                )
            if 'TF评分' in show_df.columns:
                col_cfg['TF评分'] = st.column_config.ProgressColumn(
                    'TF评分', min_value=0, max_value=100, format="%.1f"
                )
            st.dataframe(
                show_df,
                width='stretch',
                height=min(40 * top_n_show + 40, 800),
                column_config=col_cfg,
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
</div>""", unsafe_allow_html=True)


# ================================================================
#   PAGE 4: 💼 我的持仓
# ================================================================
elif page == "💼 我的持仓":
    st.markdown('<p class="header-glow">💼 我的持仓</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-sub">MY PORTFOLIO · 资金管理 + 买入记录 + 盈亏跟踪 + 仓位建议</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    account = get_paper_account()

    # --- Tab: 账户总览 / 录入买入 / 录入卖出 / 历史交易 / 仓位建议 ---
    tab_overview, tab_input, tab_sell, tab_history, tab_sizing = st.tabs(["📊 账户总览", "✏️ 录入买入", "📤 录入卖出", "📜 历史交易", "📐 仓位建议"])

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
            
            # 预测有效期状态
            days_held = r.get('days_held', 0)
            est_days = r.get('est_hold_days', 10)
            time_phase = r.get('time_phase', 1)
            phase_name = r.get('time_phase_name', '价格主导')
            phase_icons = {1: '🟢', 2: '🟡', 3: '🟠', 4: '🔴'}
            phase_icon = phase_icons.get(time_phase, '⚪')
            time_display = f"{phase_icon}{days_held}/{est_days:.0f}天"
            
            pos_rows.append({
                '代码': r['stock_code'],
                '名称': r['stock_name'],
                '买入价': f"{r['buy_price']:.2f}",
                '现价': f"{r['current_price']:.2f}" if r['current_price'] > 0 else "-",
                '数量': r.get('shares', 0),
                '盈亏%': f"{pnl_sign}{r['pnl_pct']:.1f}%" if r['current_price'] > 0 else "-",
                '当前止损': f"{r['stop_price']:.2f}",
                '止盈价': f"{r['target_price']:.2f}",
                '有效期': time_display,
                '止损状态': phase_name,
                '建议': r['advice'],
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
            st.dataframe(pd.DataFrame(pos_rows), width='stretch', hide_index=True)
            
            # 卖出提醒 (逐只展示)
            alerts_exist = any(r.get('alerts') for r in monitor_results)
            if alerts_exist:
                st.markdown("#### 🔔 退出信号 & 止损状态")
                for r in monitor_results:
                    alerts = r.get('alerts', [])
                    if not alerts:
                        continue
                    
                    days_held = r.get('days_held', 0)
                    est_days = r.get('est_hold_days', 10)
                    time_phase = r.get('time_phase', 1)
                    phase_name = r.get('time_phase_name', '')
                    original_stop = r.get('original_stop', r.get('stop_price', 0))
                    current_stop = r.get('stop_price', 0)
                    advice = r.get('advice', '')
                    
                    # 颜色根据紧急程度
                    if advice == '立即卖出':
                        border_color = '#e06060'
                        bg_color = 'rgba(224,96,96,0.08)'
                    elif advice == '建议卖出':
                        border_color = '#f0a050'
                        bg_color = 'rgba(240,160,80,0.08)'
                    else:
                        border_color = '#5eba7d'
                        bg_color = 'rgba(94,186,125,0.06)'
                    
                    alert_html = "<br>".join([f"· {a}" for a in alerts])
                    
                    # 止损收紧幅度
                    stop_tighten = ""
                    if time_phase >= 2 and original_stop > 0 and current_stop > original_stop:
                        tighten_pct = (current_stop - original_stop) / r.get('buy_price', 1) * 100
                        stop_tighten = f"<br><span style='color:#f0a050;'>止损已从 {original_stop:.2f} 收紧至 {current_stop:.2f} (上移{tighten_pct:.1f}%)</span>"
                    
                    # 时间进度条
                    progress_pct = min(days_held / est_days * 100, 100) if est_days > 0 else 0
                    bar_color = '#5eba7d' if time_phase <= 1 else ('#f0a050' if time_phase <= 2 else '#e06060')
                    
                    st.markdown(f"""
<div style="background:{bg_color};border-left:3px solid {border_color};border-radius:8px;padding:12px 16px;margin-bottom:8px;">
<div style="display:flex;justify-content:space-between;align-items:center;">
<span style="color:#e2e8f0;font-weight:700;">{r['stock_name']}({r['stock_code']})</span>
<span style="color:{border_color};font-weight:700;font-size:14px;">{advice}</span>
</div>
<div style="margin:8px 0;">
<div style="background:rgba(255,255,255,0.08);border-radius:4px;height:6px;overflow:hidden;">
<div style="background:{bar_color};height:100%;width:{progress_pct}%;border-radius:4px;transition:width 0.3s;"></div>
</div>
<div style="display:flex;justify-content:space-between;margin-top:4px;">
<span style="color:#7a869a;font-size:11px;">持有 {days_held}天 / 有效期 {est_days:.0f}天</span>
<span style="color:#7a869a;font-size:11px;">{phase_name}</span>
</div>
</div>
<div style="color:#cbd5e1;font-size:12px;line-height:1.7;">{alert_html}{stop_tighten}</div>
</div>""", unsafe_allow_html=True)

            # 关闭持仓
            st.markdown("##### 关闭已卖出的持仓")
            close_col1, close_col2 = st.columns([3, 1])
            with close_col1:
                close_options = [f"{row['stock_code']} - {row['stock_name']} ({row['buy_date']})" for _, row in manual_df.iterrows()]
                close_sel = st.selectbox("选择要关闭的持仓", close_options, key="close_sel")
            with close_col2:
                if st.button("关闭此持仓", width='stretch'):
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

        # 使用 session_state 持久化表单数据（页面切换不丢失）
        if '_buy_code' not in st.session_state:
            st.session_state['_buy_code'] = ""
        if '_buy_price' not in st.session_state:
            st.session_state['_buy_price'] = 0.0
        if '_buy_shares' not in st.session_state:
            st.session_state['_buy_shares'] = 100
        if '_buy_note' not in st.session_state:
            st.session_state['_buy_note'] = ""

        col_a, col_b = st.columns(2)
        with col_a:
            m_code = st.text_input("股票代码", value=st.session_state['_buy_code'], max_chars=6, key="m_code", placeholder="如 600519")
            st.session_state['_buy_code'] = m_code
        with col_b:
            m_name = ""
            if m_code and len(m_code.strip()) == 6:
                m_name = load_stock_name(m_code.strip())
            st.markdown("股票名称")
            if m_name:
                st.markdown(f"""<div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:8px 12px;font-size:15px;color:#e8edf5;">{m_name}</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:8px 12px;font-size:14px;color:#64748b;">{'请输入6位股票代码' if not m_code or len(m_code.strip()) < 6 else '未查到名称'}</div>""", unsafe_allow_html=True)

        col_c, col_d, col_e = st.columns(3)
        with col_c:
            m_price = st.number_input("买入价格（元）", value=st.session_state['_buy_price'], step=0.01, min_value=0.0, key="m_price")
            st.session_state['_buy_price'] = m_price
        with col_d:
            m_shares = st.number_input("买入股数", value=st.session_state['_buy_shares'], step=100, min_value=0, key="m_shares")
            st.session_state['_buy_shares'] = m_shares
        with col_e:
            m_date = st.date_input("买入日期", key="m_date")

        m_note = st.text_input("备注（可选）", value=st.session_state['_buy_note'], key="m_note", placeholder="例如：根据超跌MA60信号买入")
        st.session_state['_buy_note'] = m_note

        if st.button("✅ 确认录入", type="primary", width='stretch', key="add_manual"):
            if m_code and m_price > 0:
                r = account.add_manual_position(m_code.strip(), m_name, m_price, m_date.strftime('%Y-%m-%d'), m_shares, m_note)
                if r['success']:
                    st.success(f"已录入 {m_name}({m_code}) {m_shares}股 @ {m_price:.2f}")
                    # 录入成功后清空表单
                    st.session_state['_buy_code'] = ""
                    st.session_state['_buy_price'] = 0.0
                    st.session_state['_buy_shares'] = 100
                    st.session_state['_buy_note'] = ""
                    st.rerun()
                else:
                    st.error(r['message'])
            else:
                st.warning("请填写股票代码和买入价格")

    with tab_sell:
        st.markdown("#### 📤 录入卖出信息")
        st.markdown("当你卖出股票后，在此录入卖出信息，系统将自动计算盈亏并关闭持仓")

        manual_df_sell = account.list_manual_positions()
        if manual_df_sell.empty:
            st.markdown("""
<div class="signal-card" style="text-align:center;padding:40px;">
<div style="font-size:48px;margin-bottom:16px;">📤</div>
<div style="color:#cbd5e1;font-size:16px;">暂无持仓记录</div>
<div style="color:#7a869a;font-size:14px;margin-top:8px;">没有需要录入卖出的持仓</div>
</div>
""", unsafe_allow_html=True)
        else:
            # 持久化卖出价格
            if '_sell_price' not in st.session_state:
                st.session_state['_sell_price'] = 0.0

            sell_options = [
                f"{row['stock_code']} - {row.get('stock_name', '')} (买入@{row['buy_price']:.2f} {row['buy_date']})"
                for _, row in manual_df_sell.iterrows()
            ]
            sell_sel = st.selectbox("选择要卖出的持仓", sell_options, key="sell_sel")

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                s_price = st.number_input("卖出价格（元）", value=st.session_state['_sell_price'], step=0.01, min_value=0.0, key="s_price")
                st.session_state['_sell_price'] = s_price
            with col_s2:
                s_date = st.date_input("卖出日期", key="s_date")

            if st.button("✅ 确认卖出", type="primary", width='stretch', key="confirm_sell"):
                if s_price > 0:
                    parts = sell_sel.split(" - ")
                    s_code = parts[0]
                    # 从选项中提取买入日期
                    s_buy_date = sell_sel.split("(买入@")[1].split(" ")[1].rstrip(")")
                    result = account.sell_manual_position(s_code, s_buy_date, s_price, s_date.strftime('%Y-%m-%d'))
                    if result['success']:
                        pnl_color = "#e06060" if result['pnl_pct'] >= 0 else "#5eba7d"
                        st.success(result['message'])
                        st.markdown(f"""
<div class="signal-card" style="padding:12px 16px;">
<div style="color:{pnl_color};font-weight:700;font-size:18px;">盈亏: {"+" if result['pnl_pct']>=0 else ""}{result['pnl_pct']:.1f}%</div>
<div style="color:#cbd5e1;font-size:14px;margin-top:4px;">金额: ¥{result['pnl']:,.2f}</div>
</div>
""", unsafe_allow_html=True)
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.warning("请输入卖出价格")

    with tab_history:
        st.markdown("#### 📜 历史交易记录")
        closed_df = account.list_closed_positions(limit=100)
        if not closed_df.empty:
            # 汇总统计
            sold_df = closed_df[closed_df['status'] == 'sold']
            if not sold_df.empty:
                total_trades = len(sold_df)
                win_trades = len(sold_df[sold_df['actual_pnl_pct'] > 0])
                loss_trades = len(sold_df[sold_df['actual_pnl_pct'] < 0])
                avg_pnl = sold_df['actual_pnl_pct'].mean()
                total_pnl_amt = sold_df['actual_pnl'].sum()
                win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0

                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1:
                    st.markdown(f'<div class="signal-card"><div class="metric-label">总交易</div><div class="metric-value">{total_trades}</div></div>', unsafe_allow_html=True)
                with sc2:
                    wr_color = "#5eba7d" if win_rate >= 50 else "#e06060"
                    st.markdown(f'<div class="signal-card"><div class="metric-label">胜率</div><div class="metric-value" style="color:{wr_color};">{win_rate:.0f}%</div><div style="color:#7a869a;font-size:12px;">赢{win_trades} 亏{loss_trades}</div></div>', unsafe_allow_html=True)
                with sc3:
                    avg_color = "#e06060" if avg_pnl >= 0 else "#5eba7d"
                    st.markdown(f'<div class="signal-card"><div class="metric-label">平均盈亏</div><div class="metric-value" style="color:{avg_color};">{"+" if avg_pnl>=0 else ""}{avg_pnl:.1f}%</div></div>', unsafe_allow_html=True)
                with sc4:
                    tp_color = "#e06060" if total_pnl_amt >= 0 else "#5eba7d"
                    st.markdown(f'<div class="signal-card"><div class="metric-label">总盈亏</div><div class="metric-value" style="color:{tp_color};">¥{total_pnl_amt:,.0f}</div></div>', unsafe_allow_html=True)

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # 明细表
            hist_rows = []
            for _, row in closed_df.iterrows():
                pnl_str = f"{row.get('actual_pnl_pct', 0):+.1f}%" if row['status'] == 'sold' else "-"
                hist_rows.append({
                    '代码': row['stock_code'],
                    '名称': row.get('stock_name', ''),
                    '买入价': f"{row['buy_price']:.2f}",
                    '卖出价': f"{row.get('sell_price', 0):.2f}" if row['status'] == 'sold' else "-",
                    '数量': row.get('shares', 0),
                    '盈亏%': pnl_str,
                    '盈亏额': f"¥{row.get('actual_pnl', 0):,.0f}" if row['status'] == 'sold' else "-",
                    '买入日': row['buy_date'],
                    '卖出日': row.get('sell_date', '') if row['status'] == 'sold' else "-",
                    '状态': '已卖出' if row['status'] == 'sold' else '已关闭',
                })
            st.dataframe(pd.DataFrame(hist_rows), width='stretch', hide_index=True)
        else:
            st.info("暂无历史交易记录")

    with tab_sizing:
        st.markdown("#### 📐 智能仓位建议")
        st.markdown("输入你的总资金，系统根据**AI三层融合策略**的操作清单为你推荐配置方案")

        total_fund = st.number_input("你的总投资资金（元）", value=100000.0, step=10000.0, format="%.0f", key="total_fund")
        if st.button("生成仓位配置", type="primary", key="gen_sizing"):
            # 读取最新AI操作清单
            ai_picks = []
            try:
                action_path = os.path.join('data', 'ai_action_list.json')
                if os.path.exists(action_path):
                    with open(action_path, 'r', encoding='utf-8') as f:
                        action_data = json.load(f)
                    ai_picks = action_data.get('picks', [])
            except Exception:
                pass

            # 读取当前持仓
            manual_df = account.list_manual_positions()
            held_codes = set(manual_df['stock_code'].tolist()) if not manual_df.empty else set()
            held_count = len(held_codes)

            if ai_picks:
                # ---- 基于AI操作清单生成仓位配置 ----
                max_positions = 5  # 最多同时持有5只
                avail_slots = max(0, max_positions - held_count)
                max_per_stock_pct = 0.20  # 单只最大仓位20%
                reserve_pct = 0.15  # 预留15%现金
                invest_pct = 1.0 - reserve_pct

                # 过滤掉已持有的
                new_picks = [p for p in ai_picks if p['stock_code'] not in held_codes][:avail_slots]

                if new_picks:
                    # 按评分分配权重
                    total_score = sum(p['final_score'] for p in new_picks)
                    rows = []
                    for p in new_picks:
                        weight = p['final_score'] / total_score * invest_pct if total_score > 0 else invest_pct / len(new_picks)
                        weight = min(weight, max_per_stock_pct)  # 单只上限
                        amt = total_fund * weight
                        shares = max(100, int(amt / p['buy_price'] / 100) * 100) if p.get('buy_price', 0) > 0 else 0

                        rows.append({
                            '代码': p['stock_code'],
                            '名称': p['stock_name'],
                            '行业': p.get('board_name', ''),
                            'AI评分': f"{p['final_score']:.0f}分",
                            '仓位比例': f"{weight*100:.0f}%",
                            '配置金额': f"¥{amt:,.0f}",
                            '买入价': f"¥{p['buy_price']:.2f}",
                            '建议股数': f"{shares}股",
                            '止盈价': f"¥{p.get('sell_target', 0):.2f}" if p.get('sell_target') else "—",
                            '止损价': f"¥{p.get('sell_stop', 0):.2f}" if p.get('sell_stop') else "—",
                            '持有周期': p.get('hold_days', '3~5天'),
                        })

                    # 已持仓汇总
                    if held_count > 0:
                        rows.append({
                            '代码': '—', '名称': f'已持仓({held_count}只)', '行业': '—',
                            'AI评分': '—', '仓位比例': '—', '配置金额': '—',
                            '买入价': '—', '建议股数': '—', '止盈价': '—', '止损价': '—', '持有周期': '—',
                        })

                    # 预留现金
                    reserve_amt = total_fund * reserve_pct
                    rows.append({
                        '代码': '—', '名称': '预留现金', '行业': '—',
                        'AI评分': '—', '仓位比例': f'{reserve_pct*100:.0f}%',
                        '配置金额': f'¥{reserve_amt:,.0f}',
                        '买入价': '—', '建议股数': '—', '止盈价': '—', '止损价': '—', '持有周期': '—',
                    })

                    scan_time = action_data.get('time', 'N/A')
                    st.markdown(f'<div style="color:#7a869a;font-size:12px;margin-bottom:8px;">基于AI操作清单（{scan_time}）· 三层融合: XGBoost×0.5 + 形态×0.3 + Transformer×0.2</div>', unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
                else:
                    if held_count >= max_positions:
                        st.warning(f"当前已持仓 {held_count} 只，已达上限（{max_positions}只），建议等待卖出信号后再新增")
                    else:
                        st.info("AI操作清单中的股票你都已持有，无需新增仓位")

                # 操作规则
                max_single = total_fund * max_per_stock_pct
                st.markdown(f"""
<div class="signal-card" style="padding:12px 16px;">
<div style="color:#5b8def;font-weight:600;">操作规则（AI策略配套）：</div>
<div style="color:#cbd5e1;font-size:14px;margin-top:4px;">
· 同时持有不超过 <b>5只</b> 股票，分散风险<br>
· 单只股票不超过 <b>¥{max_single:,.0f}</b>（总资金{max_per_stock_pct*100:.0f}%）<br>
· 严格按AI操作清单的 <b style="color:#5eba7d;">买入价</b> 挂限价单，不追高超过「最高可接受价」<br>
· 触及 <b style="color:#e06060;">止盈价</b> 立即卖出，跌破 <b style="color:#94a3b8;">止损价</b> 无条件卖出<br>
· 退出优先级: ❶止损(铁律) ❷止盈 ❸追踪止损 ❹超有效期止损收紧(价格做最终裁判)<br>
· 保留 <b>{reserve_pct*100:.0f}%</b> 现金应对突发机会或加仓
</div>
</div>
""", unsafe_allow_html=True)

            else:
                st.info("暂无AI操作清单数据，请先在「每日信号」中运行AI扫描或执行每日任务")


# ================================================================
#   PAGE 5: 🎮 模拟交易
# ================================================================
elif page == "🎮 模拟交易":
    st.markdown('<p class="header-glow">🎮 模拟交易</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-sub">PAPER TRADING · 虚拟资金模拟买卖 · 验证策略效果</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    account = get_paper_account()

    # 获取持仓和价格（批量一次请求）
    positions = account.get_positions()
    current_prices = {}
    if not positions.empty:
        from src.data.data_fetcher import batch_get_realtime_prices
        codes = positions['stock_code'].tolist()
        with st.spinner(f"获取 {len(codes)} 只持仓实时价格..."):
            price_map = batch_get_realtime_prices(codes)
        for code, pi in price_map.items():
            if pi.get('close', 0) > 0:
                current_prices[code] = pi['close']

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
                    if st.button("确认买入", type="primary", width='stretch'):
                        r = account.buy(stock_code_t, stock_name_t, bp, bs)
                        st.success(r['message']) if r['success'] else st.error(r['message'])
                        if r['success']:
                            st.rerun()
                with col2:
                    st.markdown(f"#### 🟢 卖出 {stock_name_t}")
                    sp = st.number_input("卖出价格", value=curr_price, step=0.01, key="sp")
                    ss = st.number_input("卖出股数", value=100, step=100, min_value=100, key="ss")
                    if st.button("确认卖出", width='stretch'):
                        r = account.sell(stock_code_t, stock_name_t, sp, ss)
                        st.success(r['message']) if r['success'] else st.error(r['message'])
                        if r['success']:
                            st.rerun()

    with trade_tabs[1]:
        if equity['positions']:
            pos_data = [{'代码': p['code'], '名称': p['name'], '持仓': f"{p['shares']}股",
                         '成本': f"¥{p['avg_cost']:.2f}", '现价': f"¥{p['current_price']:.2f}",
                         '盈亏': f"¥{p['profit']:,.2f}", '收益率': f"{p['profit_pct']:.2f}%"} for p in equity['positions']]
            st.dataframe(pd.DataFrame(pos_data), width='stretch', hide_index=True)
        else:
            st.info("模拟盘暂无持仓")

    with trade_tabs[2]:
        trades = account.get_trades()
        if not trades.empty:
            dt = trades[['created_at', 'stock_code', 'stock_name', 'action', 'price', 'shares', 'profit']].copy()
            dt.columns = ['时间', '代码', '名称', '操作', '价格', '数量', '盈亏']
            st.dataframe(dt, width='stretch', hide_index=True)

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
                st.plotly_chart(fig, width='stretch')
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

    tab_pool, tab_train, tab_scheduler, tab_email, tab_params = st.tabs(["📦 股票池", "🧠 AI模型训练", "⏰ 定时任务", "📧 邮件配置", "📐 参数配置"])

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
            if st.button("🔄 同步股票池（申万行业分类）", type="primary", width='stretch'):
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
            if st.button("🏷️ 重新标记可交易状态", width='stretch'):
                result = pool.mark_tradeable_status()
                st.success(f"标记完成！可交易 {result['tradeable']} 只，排除 {result['excluded']} 只")
                st.rerun()

        boards = pool.get_industry_boards()
        if not boards.empty:
            st.dataframe(boards.rename(columns={'board_code': '代码', 'board_name': '名称', 'stock_count': '个股数'}),
                         width='stretch', hide_index=True, height=300)

            # 显示被排除的股票
            excluded_df = pool.get_excluded_stocks()
            if not excluded_df.empty:
                with st.expander(f"查看被排除的 {len(excluded_df)} 只股票", expanded=False):
                    st.dataframe(excluded_df.rename(columns={
                        'stock_code': '代码', 'stock_name': '名称',
                        'board_name': '行业', 'exclude_reason': '排除原因'
                    }), width='stretch', hide_index=True, height=300)

    with tab_train:
        st.markdown("#### 🧠 AI超级策略 — 模型训练")
        st.markdown("当积累了新数据后，重新训练AI模型以适应最新市场。**每日扫描不需要重训**，建议每1~2周训练一次。")

        # 模型状态
        st.markdown("##### 📊 当前模型状态")
        model_files = {
            'XGBoost (第一层)': ('xgb_v2_model.json', '量价特征 → 涨跌概率'),
            '形态聚类 (第二层)': ('pattern_engine.pkl', '走势形状 → 形态胜率'),
            'Transformer (第三层)': ('transformer_model.pt', '时序序列 → 趋势判断'),
        }
        mc1, mc2, mc3 = st.columns(3)
        for col_ui, (name, (fname, desc)) in zip([mc1, mc2, mc3], model_files.items()):
            fpath = os.path.join('data', fname)
            with col_ui:
                if os.path.exists(fpath):
                    mtime = os.path.getmtime(fpath)
                    mdate = pd.Timestamp.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                    fsize = os.path.getsize(fpath) / 1024 / 1024
                    days_ago = (pd.Timestamp.now() - pd.Timestamp.fromtimestamp(mtime)).days
                    fresh = "🟢" if days_ago <= 14 else ("🟡" if days_ago <= 30 else "🔴")
                    st.markdown(f"""
<div class="signal-card" style="padding:12px 16px;">
<div style="font-weight:700;color:#e2e8f0;font-size:14px;">{fresh} {name}</div>
<div style="color:#7a869a;font-size:12px;margin-top:4px;">{desc}</div>
<div style="color:#94a3b8;font-size:12px;margin-top:6px;">
训练时间: <b style="color:#e2e8f0;">{mdate}</b><br>
距今: <b>{days_ago}天</b> · 大小: {fsize:.1f}MB
</div>
</div>""", unsafe_allow_html=True)
        else:
                    st.markdown(f"""
<div class="signal-card" style="padding:12px 16px;">
<div style="font-weight:700;color:#e06060;font-size:14px;">❌ {name}</div>
<div style="color:#7a869a;font-size:12px;margin-top:4px;">{desc}</div>
<div style="color:#e06060;font-size:12px;margin-top:6px;">模型不存在，请训练</div>
</div>""", unsafe_allow_html=True)

        # 上次训练报告
        report_path = os.path.join('data', 'retrain_report.json')
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                last_report = json.load(f)
            with st.expander(f"📋 上次训练报告 ({last_report.get('train_date', 'N/A')})", expanded=False):
                st.json(last_report)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("##### 🚀 启动训练")
        st.markdown("""
<div class="signal-card" style="padding:12px 16px;">
<div style="color:#94a3b8;font-size:13px;">
<b>训练耗时估算 (RTX 5080 GPU):</b><br>
第一层 XGBoost: ~5分钟 · 第二层 形态聚类: ~10分钟 · 第三层 Transformer: ~25分钟<br>
<b>全部三层: ~40分钟</b>
</div>
</div>""", unsafe_allow_html=True)

        train_layer_choice = st.radio(
            "选择训练层",
            ["全部三层 (推荐)", "仅第一层 XGBoost (~5分钟)", "仅第二层 形态聚类 (~10分钟)", 
             "仅第三层 Transformer (~25分钟)", "第一+二层 (~15分钟)"],
            index=0, horizontal=True
        )

        layer_map = {
            "全部三层 (推荐)": [1, 2, 3],
            "仅第一层 XGBoost (~5分钟)": [1],
            "仅第二层 形态聚类 (~10分钟)": [2],
            "仅第三层 Transformer (~25分钟)": [3],
            "第一+二层 (~15分钟)": [1, 2],
        }
        selected_layers = layer_map[train_layer_choice]

        if st.button("🧠 开始训练", type="primary", width='stretch'):
            train_bar = st.progress(0)
            train_txt = st.empty()
            train_log = st.empty()
            log_lines = []

            total_steps = len(selected_layers)
            results = {}

            for step_i, layer_id in enumerate(selected_layers):
                layer_names = {1: 'XGBoost', 2: '形态聚类', 3: 'Transformer'}
                lname = layer_names[layer_id]
                train_bar.progress(step_i / total_steps)
                train_txt.text(f"训练中 [{step_i+1}/{total_steps}]: 第{layer_id}层 {lname}...")

                try:
                    if layer_id == 1:
                        from retrain_all import train_layer1
                        results['layer1_xgboost'] = train_layer1()
                        r = results['layer1_xgboost']
                        log_lines.append(f"✅ 第一层 XGBoost: AUC={r['test_auc']:.4f}  P@50={r['test_p@50']:.3f}  ({r['elapsed']:.0f}秒)")
                    elif layer_id == 2:
                        from retrain_all import train_layer2
                        results['layer2_pattern'] = train_layer2()
                        r = results['layer2_pattern']
                        log_lines.append(f"✅ 第二层 形态聚类: {r['high_wr_clusters']}种高胜率形态  ({r['elapsed']:.0f}秒)")
                    elif layer_id == 3:
                        from retrain_all import train_layer3
                        results['layer3_transformer'] = train_layer3()
                        r = results['layer3_transformer']
                        log_lines.append(f"✅ 第三层 Transformer: AUC={r['test_auc']:.4f}  区分度={r['discrimination']:.4f}  ({r['elapsed']:.0f}秒)")
                except Exception as e:
                    log_lines.append(f"❌ 第{layer_id}层 {lname} 失败: {e}")

                train_log.markdown("\n\n".join(log_lines))

            train_bar.progress(1.0)
            train_txt.empty()

            # 保存报告
            results['train_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            results['layers_trained'] = selected_layers
            rpt_path = os.path.join('data', 'retrain_report.json')
            with open(rpt_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)

            st.success(f"训练完成！共训练 {len(selected_layers)} 层")
            st.balloons()

    with tab_scheduler:
        st.markdown("#### ⏰ 每日定时任务")
        st.markdown("设置收盘后自动运行：更新数据 → AI三层扫描 → 生成操作清单 → 邮件推送。**全自动，无需人工干预。**")

        import subprocess as _sp

        # 读取调度器状态
        scheduler_config_path = os.path.join('data', 'scheduler_config.json')
        scheduler_status_path = os.path.join('data', 'scheduler_status.json')

        # 先刷新状态
        try:
            _sp.run(
                ['powershell', '-ExecutionPolicy', 'Bypass', '-File',
                 os.path.join(os.path.dirname(__file__), 'setup_scheduler.ps1'),
                 '-Action', 'status'],
                capture_output=True, timeout=10
            )
        except Exception:
            pass

        # 读取当前状态
        sched_active = False
        sched_time = "15:30"
        sched_last_run = ""
        sched_next_run = ""
        sched_state = "未注册"

        if os.path.exists(scheduler_config_path):
            try:
                with open(scheduler_config_path, 'r', encoding='utf-8') as f:
                    sc_cfg = json.load(f)
                sched_time = sc_cfg.get('run_time', '15:30')
                sched_active = sc_cfg.get('status', '') == 'active'
            except Exception:
                pass

        if os.path.exists(scheduler_status_path):
            try:
                with open(scheduler_status_path, 'r', encoding='utf-8') as f:
                    sc_st = json.load(f)
                if sc_st.get('exists'):
                    sched_state = sc_st.get('state', '未知')
                    sched_last_run = sc_st.get('last_run', '')
                    sched_next_run = sc_st.get('next_run', '')
                    sched_active = True
                else:
                    sched_active = False
                    sched_state = "未注册"
            except Exception:
                pass

        # 状态显示
        if sched_active:
            status_color = "#5eba7d"
            status_icon = "✅"
            status_text = f"已启用 · 每工作日 {sched_time} 自动运行"
        else:
            status_color = "#7a869a"
            status_icon = "⏸️"
            status_text = "未启用 · 需要手动点击按钮运行"

        st.markdown(f"""
<div class="signal-card" style="padding:16px 20px;">
<div style="display:flex;justify-content:space-between;align-items:center;">
<div>
<span style="font-size:20px;">{status_icon}</span>
<span style="color:{status_color};font-weight:700;font-size:16px;margin-left:8px;">{status_text}</span>
</div>
<div style="text-align:right;color:#7a869a;font-size:12px;">
{"上次运行: " + sched_last_run if sched_last_run else ""}
{"<br>下次运行: " + sched_next_run if sched_next_run else ""}
</div>
</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # 工作原理说明
        st.markdown("""
<div class="signal-card" style="padding:14px 18px;">
<div style="color:#5b8def;font-weight:700;font-size:14px;margin-bottom:8px;">📖 工作原理</div>
<div style="color:#94a3b8;font-size:13px;line-height:1.8;">
<b>1.</b> 使用 Windows 任务计划程序（Task Scheduler）注册定时任务<br>
<b>2.</b> 每个工作日（周一到周五）收盘后自动触发<br>
<b>3.</b> 自动执行完整流程：增量更新K线数据 → AI三层超级策略扫描 → 生成操作清单 → 邮件推送<br>
<b>4.</b> 即使 Streamlit 前端未打开也能正常运行<br>
<b style="color:#f0a050;">⚠️ 前提：电脑在定时时间点必须处于开机状态（不能休眠/关机）</b>
</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # 操作区
        st.markdown("##### 管理定时任务")

        sch_col1, sch_col2 = st.columns(2)
        with sch_col1:
            new_time = st.text_input("执行时间（24小时制）", value=sched_time, key="sched_time_input",
                                     help="建议15:30（收盘后30分钟，等待数据源更新）")
        with sch_col2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)  # 对齐
            time_hint = ""
            if new_time:
                try:
                    h, m = new_time.split(":")
                    h, m = int(h), int(m)
                    if 15 <= h <= 17:
                        time_hint = "✅ 推荐时间段"
                    elif h < 15:
                        time_hint = "⚠️ 收盘前运行，数据可能不完整"
                    else:
                        time_hint = "ℹ️ 可用，但较晚"
                except Exception:
                    time_hint = "❌ 格式错误，请用 HH:MM"
            st.markdown(f"<div style='color:#94a3b8;font-size:13px;margin-top:8px;'>{time_hint}</div>", unsafe_allow_html=True)

        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            enable_btn = st.button("✅ 启用定时任务", type="primary", width='stretch', key="sched_enable")
        with btn_col2:
            disable_btn = st.button("⏸️ 停用定时任务", width='stretch', key="sched_disable")
        with btn_col3:
            run_now_btn = st.button("▶️ 立即执行一次", width='stretch', key="sched_run_now")

        if enable_btn:
            with st.spinner("正在注册 Windows 计划任务..."):
                try:
                    result = _sp.run(
                        ['powershell', '-ExecutionPolicy', 'Bypass', '-File',
                         os.path.join(os.path.dirname(__file__), 'setup_scheduler.ps1'),
                         '-Action', 'register', '-Time', new_time],
                        capture_output=True, text=True, timeout=30, encoding='utf-8'
                    )
                    if result.returncode == 0:
                        st.success(f"定时任务已启用！每工作日 {new_time} 自动运行")
                        st.info("可在 Windows「任务计划程序」中查看任务 QuantX_DailyJob")
                    else:
                        st.error(f"注册失败: {result.stderr or result.stdout}")
                        st.markdown(f"<details><summary>详细输出</summary><pre>{result.stdout}\n{result.stderr}</pre></details>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"执行失败: {e}")
            st.rerun()

        if disable_btn:
            with st.spinner("正在注销计划任务..."):
                try:
                    result = _sp.run(
                        ['powershell', '-ExecutionPolicy', 'Bypass', '-File',
                         os.path.join(os.path.dirname(__file__), 'setup_scheduler.ps1'),
                         '-Action', 'unregister'],
                        capture_output=True, text=True, timeout=15, encoding='utf-8'
                    )
                    if result.returncode == 0:
                        st.success("定时任务已停用")
                    else:
                        st.error(f"注销失败: {result.stderr}")
                except Exception as e:
                    st.error(f"执行失败: {e}")
            st.rerun()

        if run_now_btn:
            st.info("正在后台启动每日任务...（请查看终端窗口了解进度）")
            try:
                _sp.Popen(
                    ['cmd', '/c', os.path.join(os.path.dirname(__file__), 'run_daily.bat')],
                    cwd=os.path.dirname(__file__),
                    creationflags=0x00000010  # CREATE_NEW_CONSOLE
                )
                st.success("任务已在新窗口中启动！完成后结果会自动保存，刷新页面即可查看。")
            except Exception as e:
                st.error(f"启动失败: {e}")

        # 运行日志
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        log_path = os.path.join('data', 'scheduler_log.txt')
        if os.path.exists(log_path):
            with st.expander("📋 运行日志（最近20条）", expanded=False):
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    recent = lines[-20:] if len(lines) > 20 else lines
                    st.code("".join(recent), language="text")
                except Exception:
                    st.info("暂无日志")

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
· 大盘情绪温度计（涨跌活跃/成交额/主力资金/北向/融资）<br>
· AI超级策略操作清单（XGBoost+形态+Transformer三层融合精选）<br>
· 持仓卖出提醒（止损/止盈/追踪止损触发）<br>
· 在「📡 每日信号 → 执行每日任务」中手动触发，或配置定时任务自动发送
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
            if st.button("🚀 开始训练", type="primary", width='stretch', key="train_btn"):
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
