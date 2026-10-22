"""Web App phân tích Pole-Zero & đáp ứng tần số hệ LTI rời rạc (Streamlit).

Chạy:  streamlit run app.py
Phần tính toán nằm trong lti_core.py; file này chỉ lo giao diện.
"""
import html
import inspect
import io

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lti_core import (
    parse_complex_list, ba_from_zpk, zpk_from_ba, roc_options, is_stable,
    is_causal, impulse_response, freq_response, group_points, expand_pair,
)

st.set_page_config(page_title="Pole-Zero LTI Analyzer", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ====================================================================== THEME
C_PRIMARY, C_NAVY, C_TEAL = "#1F5FBF", "#16325C", "#0E9AA7"
C_POLE, C_ZERO, C_ORANGE = "#D64545", "#1E8E5A", "#E08A1E"
C_MUTED, C_GRID, C_AXIS = "#667085", "#EEF1F6", "#C9D1DD"
FONT = "Inter, 'Segoe UI', Roboto, Arial, sans-serif"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp { font-family: 'Inter', 'Segoe UI', Roboto, Arial, sans-serif; background: #F4F6FA; }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1320px; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- Hero ---------- */
.hero { background: linear-gradient(120deg, #16325C 0%, #1F5FBF 62%, #0E9AA7 100%);
        color: #fff; padding: 1.5rem 1.9rem; border-radius: 18px; margin-bottom: 1.1rem;
        box-shadow: 0 10px 28px rgba(22,50,92,.20); }
.hero-title { font-size: 1.65rem; font-weight: 700; line-height: 1.25; letter-spacing: -.01em; }
.hero-sub { margin-top: .4rem; font-size: .95rem; opacity: .92; }
.hero-tags { margin-top: .8rem; }
.hero-tags span { display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28);
        padding: .15rem .65rem; border-radius: 999px; font-size: .75rem; margin-right: .4rem; }

/* ---------- KPI cards ---------- */
.kpi { background: #fff; border: 1px solid #E3E8EF; border-radius: 14px; padding: .85rem 1.1rem;
       box-shadow: 0 1px 3px rgba(16,24,40,.06); height: 100%; }
.kpi-label { font-size: .7rem; text-transform: uppercase; letter-spacing: .07em; color: #667085; font-weight: 600; }
.kpi-value { font-size: 1.3rem; font-weight: 700; margin-top: .2rem; line-height: 1.25; }
.kpi-sub { font-size: .78rem; color: #667085; margin-top: .15rem; }
.ok { color: #12805C; } .bad { color: #C0392B; } .neutral { color: #1F5FBF; } .warn { color: #B7791F; }

/* ---------- Banner ---------- */
.banner { border-radius: 12px; padding: .75rem 1.1rem; margin: .9rem 0 .4rem 0; font-size: .92rem; border: 1px solid; }
.banner.ok { background: #ECFDF3; border-color: #ABEFC6; color: #085D3A; }
.banner.bad { background: #FEF3F2; border-color: #FECDCA; color: #912018; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E3E8EF; }
.sb-title { display: flex; align-items: center; gap: .55rem; font-weight: 700; font-size: .95rem;
            color: #16325C; margin: 1.1rem 0 .4rem 0; }
.sb-title .step { background: #1F5FBF; color: #fff; width: 1.45rem; height: 1.45rem; border-radius: 50%;
            display: inline-flex; align-items: center; justify-content: center; font-size: .75rem; }
.sb-brand { font-weight: 700; color: #16325C; font-size: 1.05rem; padding-bottom: .5rem; border-bottom: 1px solid #E3E8EF; }

/* ---------- Sidebar: khung cho các ô nhập ---------- */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #D5DCE8 !important; border-radius: 14px !important; }
section[data-testid="stSidebar"] div[data-baseweb="input"],
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    border: 1.5px solid #CBD5E1 !important; border-radius: 10px !important; background: #F8FAFC !important; }
section[data-testid="stSidebar"] div[data-baseweb="input"] > div { background: transparent !important; border: none !important; }
section[data-testid="stSidebar"] div[data-baseweb="input"]:focus-within,
section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within {
    border-color: #1F5FBF !important; box-shadow: 0 0 0 3px rgba(31,95,191,.15); background: #fff !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: .4rem; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    border: 1.5px solid #D5DCE8; border-radius: 10px; padding: .45rem .75rem; background: #fff;
    width: 100%; margin: 0; transition: all .15s; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { border-color: #1F5FBF; background: #F5F8FE; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    border-color: #1F5FBF; background: #EAF1FD; }
section[data-testid="stSidebar"] div[data-testid="stExpander"] {
    border: 1.5px solid #D5DCE8; border-radius: 10px; background: #FBFCFE; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: .35rem; border-bottom: 1px solid #DCE2EC; }
.stTabs [data-baseweb="tab"] { padding: .55rem 1.1rem; font-weight: 600; color: #667085; border-radius: 10px 10px 0 0; }
.stTabs [aria-selected="true"] { color: #1F5FBF; background: #fff; }

/* ---------- Bordered containers (card look) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
.section-title { font-weight: 700; color: #16325C; font-size: 1rem; margin-bottom: .3rem; }
.note { color: #667085; font-size: .82rem; }
.footer { text-align: center; color: #98A2B3; font-size: .78rem; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ====================================================================== HELPERS
def show(fig):
    """Hiển thị biểu đồ Plotly, tương thích nhiều phiên bản Streamlit."""
    if "width" in inspect.signature(st.plotly_chart).parameters:
        st.plotly_chart(fig, theme=None, width="stretch")
    else:
        st.plotly_chart(fig, theme=None, use_container_width=True)


def style(fig, height=420, legend_top=True):
    """Áp dụng một phong cách thống nhất cho mọi biểu đồ."""
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family=FONT, size=13, color="#1B2430"),
        margin=dict(l=55, r=20, t=40 if legend_top else 20, b=48),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=C_AXIS, font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)")
        if legend_top else dict(),
    )
    fig.update_xaxes(gridcolor=C_GRID, zerolinecolor=C_AXIS, linecolor=C_AXIS, ticks="outside", tickcolor=C_AXIS)
    fig.update_yaxes(gridcolor=C_GRID, zerolinecolor=C_AXIS, linecolor=C_AXIS, ticks="outside", tickcolor=C_AXIS)
    return fig


def kpi(label, value, tone="neutral", sub=""):
    return (f'<div class="kpi"><div class="kpi-label">{html.escape(label)}</div>'
            f'<div class="kpi-value {tone}">{html.escape(value)}</div>'
            f'<div class="kpi-sub">{html.escape(sub)}</div></div>')


def sb_title(target, n, text):
    target.markdown(f'<div class="sb-title"><span class="step">{n}</span>{text}</div>',
                    unsafe_allow_html=True)


def poly_str(c):
    """Chuỗi LaTeX của đa thức theo z^-1 (None nếu hệ số phức)."""
    if np.iscomplexobj(c):
        return None
    terms = []
    for k, v in enumerate(c):
        if abs(v) < 1e-12:
            continue
        zp = "" if k == 0 else (" z^{-1}" if k == 1 else f" z^{{-{k}}}")
        terms.append(f"{v:+.4g}{zp}")
    return " ".join(terms).lstrip("+") or "0"


PRESETS = {
    "Thông thấp bậc 1: 1/(1-0.8z⁻¹)": ("1", "1, -0.8"),
    "Cộng hưởng bậc 2 (cực 0.85∠45°)": ("1", "1, -1.2, 0.72"),
    "Notch (zero trên vòng tròn, cực r=0.9)": ("1, -1.414, 1", "1, -1.273, 0.81"),
    "FIR trung bình trượt 5 điểm": ("1, 1, 1, 1, 1", "1"),
    "Không ổn định (cực = 1.2)": ("1", "1, -1.2"),
    "Hai cực 0.5 và 2 (thử 3 ROC)": ("1", "1, -2.5, 1"),
    "Trễ đơn vị z⁻¹ / (1-0.5z⁻¹)": ("0, 1", "1, -0.5"),
}

# ====================================================================== HERO
st.markdown("""
<div class="hero">
  <div class="hero-title">Phân tích Pole-Zero &amp; Đáp ứng tần số hệ LTI</div>
  <div class="hero-sub">Công cụ học tập trực quan hóa mối liên hệ giữa vị trí cực/zero với tính ổn định,
  miền hội tụ và đáp ứng tần số của hệ thống rời rạc.</div>
  <div class="hero-tags"><span>Biến đổi Z</span><span>ROC</span><span>Đáp ứng xung h(n)</span>
  <span>Biên độ / Pha</span></div>
</div>
""", unsafe_allow_html=True)

# ====================================================================== SIDEBAR
st.sidebar.markdown('<div class="sb-brand">📈 Pole-Zero Analyzer</div>', unsafe_allow_html=True)
box = st.sidebar.container(border=True)
sb_title(box, 1, "Nhập hệ thống")
mode = box.radio("Cách nhập", ["Hệ số H(z)", "Cực/Zero (số phức)", "Tương tác (slider r–θ)"],
                        label_visibility="collapsed")

try:
    if mode == "Hệ số H(z)":
        name = box.selectbox("Ví dụ mẫu", list(PRESETS))
        b_def, a_def = PRESETS[name]
        b_txt = box.text_input("Tử số b  (b₀, b₁, …)", b_def, key=f"b_{name}")
        a_txt = box.text_input("Mẫu số a  (a₀, a₁, …)", a_def, key=f"a_{name}")
        box.caption("H(z) = B(z⁻¹)/A(z⁻¹); nhập hệ số theo lũy thừa z⁻¹ tăng dần.")
        zeros, poles, gain, b, a = zpk_from_ba(parse_complex_list(b_txt), parse_complex_list(a_txt))

    elif mode == "Cực/Zero (số phức)":
        z_txt = box.text_input("Zeros", "1j, -1j")
        p_txt = box.text_input("Poles", "0.6+0.6j, 0.6-0.6j")
        gain = box.number_input("Hệ số khuếch đại G", value=1.0, step=0.1)
        box.caption("Dùng ký hiệu 1j (không phải j). Ví dụ: 0.6+0.6j, 0.6-0.6j")
        zeros, poles = parse_complex_list(z_txt), parse_complex_list(p_txt)
        b, a = ba_from_zpk(zeros, poles, gain)

    else:
        gain = box.slider("Gain G", -5.0, 5.0, 1.0, 0.1)
        zeros_l, poles_l = [], []
        for kind, label, default_n, store in [("z", "Zero", 2, zeros_l), ("p", "Cực", 2, poles_l)]:
            n_el = box.number_input(f"Số {label.lower()}", 0, 6, default_n, key=f"n_{kind}")
            for i in range(int(n_el)):
                with box.expander(f"{label} {i + 1}", expanded=(i == 0)):
                    r = st.slider("Bán kính r", 0.0, 2.5, 1.0 if kind == "z" else 0.8, 0.01, key=f"r_{kind}{i}")
                    th = st.slider("Góc θ (độ)", -180, 180, 45 + 45 * i, 1, key=f"t_{kind}{i}")
                    cj = st.checkbox("Thêm liên hợp phức", True, key=f"c_{kind}{i}")
                    store.extend(expand_pair(r, th, cj))
        zeros, poles = np.array(zeros_l, dtype=complex), np.array(poles_l, dtype=complex)
        b, a = ba_from_zpk(zeros, poles, gain)
except Exception as e:  # lỗi nhập liệu
    st.error(f"Dữ liệu nhập không hợp lệ: {e}")
    st.stop()

box = st.sidebar.container(border=True)
sb_title(box, 2, "Miền hội tụ (ROC)")
opts = roc_options(poles)
idx = box.radio("ROC", range(len(opts)), index=len(opts) - 1,
                       format_func=lambda i: opts[i][2], label_visibility="collapsed")
roc = opts[idx]
stable, causal = is_stable(roc), is_causal(roc)

box = st.sidebar.container(border=True)
sb_title(box, 3, "Hiển thị")
n_half = box.slider("Khoảng n của h(n): −N … N", 10, 100, 30)
db_scale = box.toggle("Biên độ theo dB", True) if hasattr(st.sidebar, "toggle") \
    else box.checkbox("Biên độ theo dB", True)
show_marks = box.checkbox("Vạch tần số đặc trưng (theo góc cực/zero)", True)

# ====================================================================== FORMULA + KPI
nstr, dstr = poly_str(np.atleast_1d(b)), poly_str(np.atleast_1d(a))
with st.container(border=True):
    st.markdown('<div class="section-title">Hàm truyền đạt</div>', unsafe_allow_html=True)
    if nstr and dstr:
        st.latex(rf"H(z)=\frac{{{nstr}}}{{{dstr}}}")
    else:
        st.write("Hệ số phức:", "b =", np.round(b, 4), " a =", np.round(a, 4))

roc_short = roc[2].split("  ")[0]
order = max(len(np.atleast_1d(a)), len(np.atleast_1d(b))) - 1
k1, k2, k3, k4 = st.columns(4)
k1.markdown(kpi("Nhân quả", "Có" if causal else "Không", "ok" if causal else "warn",
                "h(n) = 0 với n < 0" if causal else "Không phải hệ nhân quả"), unsafe_allow_html=True)
k2.markdown(kpi("Ổn định BIBO", "Ổn định" if stable else "Không ổn định", "ok" if stable else "bad",
                "ROC chứa |z| = 1" if stable else "ROC không chứa |z| = 1"), unsafe_allow_html=True)
k3.markdown(kpi("Miền hội tụ", roc_short, "neutral", roc[2].split("  ")[1].strip("()")
                if "  " in roc[2] else ""), unsafe_allow_html=True)
k4.markdown(kpi("Bậc hệ thống", str(order), "neutral",
                f"{len(zeros)} zero · {len(poles)} cực"), unsafe_allow_html=True)

if stable:
    st.markdown('<div class="banner ok">✔ ROC chứa vòng tròn đơn vị → DTFT tồn tại và hệ thống ổn định.</div>',
                unsafe_allow_html=True)
else:
    st.markdown('<div class="banner bad">⚠ ROC KHÔNG chứa vòng tròn đơn vị → hệ không ổn định với ROC này. '
                'Đồ thị H(e<sup>jω</sup>) chỉ là giá trị đại số của công thức, không phải DTFT thật.</div>',
                unsafe_allow_html=True)

st.write("")
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Pole-Zero & ROC", "📉 Đáp ứng xung h(n)",
                                  "🌊 Biên độ / Pha", "📘 Lý thuyết"])

# ====================================================================== TAB 1
with tab1:
    allr = np.abs(np.r_[zeros, poles, 1.0])
    R = max(1.5, 1.25 * allr.max())
    th = np.linspace(0, 2 * np.pi, 400)
    lo, hi = roc[0], roc[1]
    hi_plot = min(hi, 2 * R)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.r_[hi_plot * np.cos(th), lo * np.cos(th[::-1])],
        y=np.r_[hi_plot * np.sin(th), lo * np.sin(th[::-1])],
        fill="toself", mode="lines", fillcolor="rgba(31,95,191,0.14)", line=dict(width=0),
        name="ROC", hoverinfo="skip"))
    for rad in (lo, hi):                                   # biên ROC
        if 1e-6 < rad < 1.6 * R:
            fig.add_trace(go.Scatter(x=rad * np.cos(th), y=rad * np.sin(th), mode="lines",
                                     line=dict(color=C_PRIMARY, width=1.2, dash="dot"),
                                     showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines",
                             line=dict(color="#344054", width=1.8), name="Vòng tròn đơn vị |z|=1",
                             hoverinfo="skip"))
    for pts, sym, col, nm in [(group_points(zeros), "circle-open", C_ZERO, "Zero"),
                              (group_points(poles), "x", C_POLE, "Cực")]:
        if pts:
            vals = np.array([p for p, _ in pts])
            fig.add_trace(go.Scatter(
                x=vals.real, y=vals.imag, mode="markers+text", name=nm,
                marker=dict(symbol=sym, size=13, color=col, line=dict(width=2.8, color=col)),
                text=[f"×{m}" if m > 1 else "" for _, m in pts], textposition="top right",
                textfont=dict(color=col, size=12),
                customdata=np.column_stack([np.abs(vals), np.degrees(np.angle(vals))]),
                hovertemplate=("Re = %{x:.4f}<br>Im = %{y:.4f}<br>r = %{customdata[0]:.4f}, "
                               "θ = %{customdata[1]:.1f}°<extra>" + nm + "</extra>")))
    style(fig, height=600)
    fig.update_xaxes(title="Phần thực Re{z}", range=[-R, R], zeroline=True, zerolinewidth=1.2)
    fig.update_yaxes(title="Phần ảo Im{z}", range=[-R, R], scaleanchor="x", zeroline=True, zerolinewidth=1.2)

    left, right = st.columns([3, 2], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown('<div class="section-title">Mặt phẳng z</div>', unsafe_allow_html=True)
            show(fig)
    with right:
        with st.container(border=True):
            st.markdown('<div class="section-title">Bảng cực / zero</div>', unsafe_allow_html=True)
            rows = [{"Loại": "Zero", "Re": round(z.real, 4), "Im": round(z.imag, 4),
                     "|z|": round(abs(z), 4), "∠ (độ)": round(float(np.degrees(np.angle(z))), 2)} for z in zeros]
            rows += [{"Loại": "Cực", "Re": round(p.real, 4), "Im": round(p.imag, 4),
                      "|z|": round(abs(p), 4), "∠ (độ)": round(float(np.degrees(np.angle(p))), 2)} for p in poles]
            if rows:
                st.dataframe(rows, hide_index=True)
            else:
                st.info("Hệ không có cực/zero hữu hạn.")
            st.markdown('<div class="note">Điểm tại gốc có thể được thêm tự động để cân bằng bậc '
                        'của tử số và mẫu số.</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="section-title">Chú giải</div>', unsafe_allow_html=True)
            st.markdown(
                f'<span style="color:{C_POLE};font-weight:700">✕</span> Cực &nbsp;&nbsp; '
                f'<span style="color:{C_ZERO};font-weight:700">○</span> Zero &nbsp;&nbsp; '
                f'<span style="color:{C_PRIMARY};font-weight:700">▨</span> Miền hội tụ ROC &nbsp;&nbsp; '
                f'<span style="color:#344054;font-weight:700">◯</span> |z| = 1',
                unsafe_allow_html=True)

# ====================================================================== TAB 2
with tab2:
    try:
        n, h = impulse_response(b, a, roc, -n_half, n_half)
        hf = go.Figure()
        for part, col, nm in [(np.real(h), C_PRIMARY, "Re{h(n)}"), (np.imag(h), C_ORANGE, "Im{h(n)}")]:
            if nm.startswith("Im") and np.all(np.abs(part) < 1e-9):
                continue
            xs, ys = [], []
            for ni, yi in zip(n, part):
                xs += [ni, ni, None]
                ys += [0, yi, None]
            hf.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=col, width=1.6),
                                    showlegend=False, hoverinfo="skip"))
            hf.add_trace(go.Scatter(x=n, y=part, mode="markers", name=nm,
                                    marker=dict(color=col, size=7, line=dict(width=1, color="#fff")),
                                    hovertemplate="n = %{x}<br>h(n) = %{y:.5g}<extra></extra>"))
        style(hf, height=420)
        hf.update_xaxes(title="n")
        hf.update_yaxes(title="h(n)")

        hh = np.abs(np.atleast_1d(h))
        s1, s2, s3 = st.columns(3)
        s1.markdown(kpi("h(0)", f"{np.real(h[n_half]):.4g}", "neutral", "Giá trị tại n = 0"),
                    unsafe_allow_html=True)
        s2.markdown(kpi("max |h(n)|", f"{hh.max():.4g}", "neutral", f"Trong khoảng n ∈ [−{n_half}, {n_half}]"),
                    unsafe_allow_html=True)
        s3.markdown(kpi("Σ |h(n)| (cửa sổ)", f"{hh.sum():.4g}", "ok" if stable else "bad",
                        "Hữu hạn khi ổn định" if stable else "Tăng theo cửa sổ khi không ổn định"),
                    unsafe_allow_html=True)
        st.write("")
        with st.container(border=True):
            st.markdown('<div class="section-title">Đáp ứng xung h(n)</div>', unsafe_allow_html=True)
            show(hf)
            if causal:
                st.info("ROC nằm ngoài mọi cực → h(n) = 0 với n < 0 (hệ nhân quả).")
            elif roc[0] == 0:
                st.info("ROC nằm trong mọi cực → h(n) = 0 với n ≥ 0 (hệ phản nhân quả).")
            else:
                st.info("ROC dạng vành khăn → h(n) hai phía (có cả n < 0 và n ≥ 0).")
            if not stable:
                st.warning("Hệ không ổn định: |h(n)| tăng không giới hạn khi |n| → ∞ "
                           "theo hướng có cực nằm ngoài ROC.")
            buf = io.StringIO()
            buf.write("n,Re_h,Im_h\n")
            for ni, hi_v in zip(n, np.atleast_1d(h)):
                buf.write(f"{ni},{np.real(hi_v):.10g},{np.imag(hi_v):.10g}\n")
            st.download_button("⬇ Tải h(n) dạng CSV", buf.getvalue(), "h_n.csv", "text/csv")
    except Exception as e:
        st.error(f"Không tính được h(n): {e}")

# ====================================================================== TAB 3
with tab3:
    w, H = freq_response(b, a)
    mag = np.abs(H)
    mag = np.where(np.isfinite(mag), mag, np.nan)
    mag_plot = 20 * np.log10(np.maximum(mag, 1e-4)) if db_scale else mag
    ph = np.degrees(np.unwrap(np.angle(np.where(np.isfinite(H), H, 1))))
    x = w / np.pi

    ff = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                       subplot_titles=("Đáp ứng biên độ |H(e^{jω})|", "Đáp ứng pha ∠H(e^{jω})"))
    ff.add_trace(go.Scatter(x=x, y=mag_plot, mode="lines", line=dict(color=C_PRIMARY, width=2.4),
                            name="Biên độ", hovertemplate="ω/π = %{x:.3f}<br>%{y:.4g}<extra></extra>"),
                 row=1, col=1)
    ff.add_trace(go.Scatter(x=x, y=ph, mode="lines", line=dict(color=C_POLE, width=2.4),
                            name="Pha", hovertemplate="ω/π = %{x:.3f}<br>%{y:.4g}°<extra></extra>"),
                 row=2, col=1)
    if show_marks:
        angs = set()
        for p in poles:
            if abs(p) > 0.6:
                angs.add(round(float(np.angle(p)) / np.pi, 4))
        for z in zeros:
            if abs(abs(z) - 1) < 1e-3:
                angs.add(round(float(np.angle(z)) / np.pi, 4))
        for ang in angs:
            for rr in (1, 2):
                ff.add_vline(x=ang, row=rr, col=1, line=dict(color=C_TEAL, width=1, dash="dot"))
    style(ff, height=640, legend_top=False)
    ff.update_layout(showlegend=False)
    ff.update_yaxes(title_text="dB" if db_scale else "|H|", row=1, col=1)
    ff.update_yaxes(title_text="độ", row=2, col=1)
    ff.update_xaxes(title_text="ω / π  (rad/mẫu)", row=2, col=1, dtick=0.25, range=[-1, 1])
    ff.update_xaxes(range=[-1, 1], dtick=0.25, row=1, col=1)
    ff.update_annotations(font=dict(size=14, color=C_NAVY, family=FONT), xanchor="left", x=0.0)

    with st.container(border=True):
        show(ff)
        if not stable:
            st.warning("ROC không chứa |z| = 1: đồ thị chỉ là giá trị đại số của công thức.")

    with st.container(border=True):
        st.markdown('<div class="section-title">Nhận xét tự động</div>', unsafe_allow_html=True)
        notes = []
        for p in poles:
            if abs(p) > 0.6:
                notes.append(f"Cực |p| = {abs(p):.3f} ∠ {np.degrees(np.angle(p)):.1f}° → đỉnh cộng hưởng gần "
                             f"ω ≈ {np.angle(p) / np.pi:.3f}π; càng gần vòng tròn đơn vị đỉnh càng nhọn.")
        for z in zeros:
            if abs(abs(z) - 1) < 1e-3:
                notes.append(f"Zero trên vòng tròn đơn vị ∠ {np.degrees(np.angle(z)):.1f}° → |H| = 0 "
                             f"tại ω = {np.angle(z) / np.pi:.3f}π (triệt tần số này).")
        if notes:
            st.markdown("\n".join(f"- {s}" for s in notes))
        else:
            st.write("Không có đặc điểm nổi bật.")

# ====================================================================== TAB 4
with tab4:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown("#### Biến đổi Z và ROC")
            st.markdown(r"""
**Biến đổi Z:** $H(z)=\sum_{n}h(n)z^{-n}$, chỉ xác định trên **ROC**.

1. ROC là hình tròn / vành khăn tâm O, **không chứa cực nào**.
2. Nhân quả ⇔ ROC là miền $|z|>r_{max}$; phản nhân quả ⇔ $|z|<r_{min}$.
3. **Ổn định BIBO ⇔ ROC chứa vòng tròn đơn vị** $|z|=1$.
4. Hệ nhân quả ổn định ⇔ mọi cực nằm **trong** vòng tròn đơn vị.
""")
    with c2:
        with st.container(border=True):
            st.markdown("#### Đáp ứng tần số và đáp ứng xung")
            st.markdown(r"""
**Đáp ứng tần số:** $H(e^{j\omega})=H(z)\big|_{z=e^{j\omega}}$ (chỉ khi ROC chứa $|z|=1$).

Về hình học: $|H(e^{j\omega})|=|G|\dfrac{\prod|e^{j\omega}-z_k|}{\prod|e^{j\omega}-p_k|}$. Điểm
$e^{j\omega}$ lại gần cực thì biên độ tăng, lại gần zero thì biên độ giảm.

**h(n)** từ khai triển phân thức từng phần: số hạng $\frac{r}{1-pz^{-1}}$ cho
$r\,p^n u(n)$ nếu cực trong biên trong của ROC, hoặc $-r\,p^n u(-n-1)$ nếu cực ngoài.
""")

st.markdown('<div class="footer">Đề tài 13 · Xử lý tín hiệu số · Python · Streamlit · Plotly</div>',
            unsafe_allow_html=True)