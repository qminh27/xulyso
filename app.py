import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lti_core import (
    parse_complex_list, ba_from_zpk, zpk_from_ba, roc_options, is_stable,
    is_causal, impulse_response, freq_response, group_points, expand_pair,
)

st.set_page_config(page_title="Pole-Zero LTI Analyzer", page_icon="📈", layout="wide")
st.title("📈 Phân tích Pole-Zero & Đáp ứng tần số hệ LTI")

PRESETS = {
    "Thông thấp bậc 1: 1/(1-0.8z⁻¹)": ("1", "1, -0.8"),
    "Cộng hưởng bậc 2 (cực 0.85∠45°)": ("1", "1, -1.2, 0.72"),
    "Notch (zero trên vòng tròn, cực r=0.9)": ("1, -1.414, 1", "1, -1.273, 0.81"),
    "FIR trung bình trượt 5 điểm": ("1, 1, 1, 1, 1", "1"),
    "Không ổn định (cực = 1.2)": ("1", "1, -1.2"),
    "Hai cực 0.5 và 2 (thử 3 ROC)": ("1", "1, -2.5, 1"),
    "Trễ đơn vị z⁻¹ / (1-0.5z⁻¹)": ("0, 1", "1, -0.5"),
}

# ============================================================ SIDEBAR: nhập hệ
st.sidebar.header("1. Nhập hệ thống")
mode = st.sidebar.radio(
    "Cách nhập",
    ["Hệ số H(z)", "Cực/Zero (số phức)", "Tương tác (slider r–θ)"],
)

try:
    if mode == "Hệ số H(z)":
        name = st.sidebar.selectbox("Ví dụ mẫu", list(PRESETS))
        b_def, a_def = PRESETS[name]
        st.sidebar.caption("H(z) = B(z⁻¹)/A(z⁻¹); nhập hệ số theo lũy thừa z⁻¹ tăng dần.")
        b_txt = st.sidebar.text_input("Tử số b (b0, b1, ...)", b_def, key=f"b_{name}")
        a_txt = st.sidebar.text_input("Mẫu số a (a0, a1, ...)", a_def, key=f"a_{name}")
        b_in = parse_complex_list(b_txt)
        a_in = parse_complex_list(a_txt)
        zeros, poles, gain, b, a = zpk_from_ba(b_in, a_in)

    elif mode == "Cực/Zero (số phức)":
        st.sidebar.caption("Ví dụ: 0.6+0.6j, 0.6-0.6j (dùng 'j', ví dụ 1j chứ không phải j)")
        z_txt = st.sidebar.text_input("Zeros", "1j, -1j")
        p_txt = st.sidebar.text_input("Poles", "0.6+0.6j, 0.6-0.6j")
        gain = st.sidebar.number_input("Hệ số khuếch đại G", value=1.0, step=0.1)
        zeros = parse_complex_list(z_txt)
        poles = parse_complex_list(p_txt)
        b, a = ba_from_zpk(zeros, poles, gain)

    else:
        st.sidebar.caption("Kéo slider bán kính r và góc θ để di chuyển cực/zero.")
        gain = st.sidebar.slider("Gain G", -5.0, 5.0, 1.0, 0.1)
        zeros_l, poles_l = [], []
        for kind, label, default_n, store in [
            ("z", "Zero", 2, zeros_l), ("p", "Cực", 2, poles_l)
        ]:
            n_el = st.sidebar.number_input(f"Số {label}", 0, 6, default_n, key=f"n_{kind}")
            for i in range(int(n_el)):
                with st.sidebar.expander(f"{label} {i + 1}", expanded=(i == 0)):
                    r = st.slider("r", 0.0, 2.5, 1.0 if kind == "z" else 0.8, 0.01, key=f"r_{kind}{i}")
                    th = st.slider("θ (độ)", -180, 180, 45 + 45 * i, 1, key=f"t_{kind}{i}")
                    cj = st.checkbox("Thêm liên hợp phức", True, key=f"c_{kind}{i}")
                    store.extend(expand_pair(r, th, cj))
        zeros, poles = np.array(zeros_l, dtype=complex), np.array(poles_l, dtype=complex)
        b, a = ba_from_zpk(zeros, poles, gain)
except Exception as e:  # lỗi nhập liệu
    st.error(f"Dữ liệu nhập không hợp lệ: {e}")
    st.stop()

# ============================================================ SIDEBAR: ROC
st.sidebar.header("2. Chọn miền hội tụ (ROC)")
opts = roc_options(poles)
idx = st.sidebar.radio("ROC", range(len(opts)), index=len(opts) - 1,
                       format_func=lambda i: opts[i][2])
roc = opts[idx]
stable, causal = is_stable(roc), is_causal(roc)

st.sidebar.header("3. Tuỳ chọn hiển thị")
n_half = st.sidebar.slider("Khoảng n của h(n): −N…N", 10, 100, 30)
db_scale = st.sidebar.checkbox("Biên độ theo dB", True)

# ============================================================ HEADER METRICS
def poly_str(c):
    if np.iscomplexobj(c):
        return None
    terms = []
    for k, v in enumerate(c):
        if abs(v) < 1e-12:
            continue
        zp = "" if k == 0 else (" z^{-1}" if k == 1 else f" z^{{-{k}}}")
        terms.append(f"{v:+.4g}{zp}")
    s = " ".join(terms).lstrip("+")
    return s or "0"

nstr, dstr = poly_str(np.atleast_1d(b)), poly_str(np.atleast_1d(a))
if nstr and dstr:
    st.latex(rf"H(z)=\frac{{{nstr}}}{{{dstr}}}")
else:
    st.write("Hệ số phức:", "b =", np.round(b, 4), " a =", np.round(a, 4))

c1, c2, c3 = st.columns(3)
c1.metric("Nhân quả (causal)", "Có" if causal else "Không")
c2.metric("Ổn định BIBO", "Có" if stable else "Không")
c3.metric("ROC", roc[2].split("  ")[0])
if stable:
    st.success("ROC chứa vòng tròn đơn vị → DTFT tồn tại, hệ ổn định.")
else:
    st.warning("ROC KHÔNG chứa vòng tròn đơn vị → hệ không ổn định với ROC này; "
               "đồ thị H(e^{jω}) chỉ là giá trị đại số của công thức, không phải DTFT thật.")

# ============================================================ TABS
tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Pole-Zero & ROC", "📉 Đáp ứng xung h(n)", "🌊 Biên độ / Pha", "📘 Lý thuyết"])

# ---------------- Tab 1
with tab1:
    allr = np.abs(np.r_[zeros, poles, 1.0])
    R = max(1.5, 1.25 * allr.max())
    th = np.linspace(0, 2 * np.pi, 400)
    hi_plot = min(roc[1], 2 * R)
    fig = go.Figure()
    ring_x = np.r_[hi_plot * np.cos(th), roc[0] * np.cos(th[::-1])]
    ring_y = np.r_[hi_plot * np.sin(th), roc[0] * np.sin(th[::-1])]
    fig.add_trace(go.Scatter(x=ring_x, y=ring_y, fill="toself", mode="lines",
                             fillcolor="rgba(46,134,222,0.20)", line=dict(width=0),
                             name="ROC", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines",
                             line=dict(color="gray", dash="dash"), name="Vòng tròn đơn vị"))
    for pts, sym, col, nm in [(group_points(zeros), "circle-open", "#1e8449", "Zero"),
                              (group_points(poles), "x", "#c0392b", "Cực")]:
        if pts:
            fig.add_trace(go.Scatter(
                x=[p.real for p, _ in pts], y=[p.imag for p, _ in pts], mode="markers+text",
                marker=dict(symbol=sym, size=14, color=col, line=dict(width=3, color=col)),
                text=[f"({m})" if m > 1 else "" for _, m in pts], textposition="top right",
                name=nm,
                hovertemplate="Re=%{x:.4f}<br>Im=%{y:.4f}<extra>" + nm + "</extra>"))
    fig.update_layout(height=620, xaxis=dict(title="Re{z}", range=[-R, R], zeroline=True),
                      yaxis=dict(title="Im{z}", range=[-R, R], scaleanchor="x", zeroline=True),
                      legend=dict(orientation="h", y=1.05))
    left, right = st.columns([3, 2])
    left.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Bảng cực / zero")
        rows = [{"Loại": "Zero", "Re": z.real, "Im": z.imag, "|z|": abs(z),
                 "∠ (độ)": np.degrees(np.angle(z))} for z in zeros]
        rows += [{"Loại": "Cực", "Re": p.real, "Im": p.imag, "|z|": abs(p),
                  "∠ (độ)": np.degrees(np.angle(p))} for p in poles]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("Chú ý: số cực/zero tại gốc có thể được thêm tự động để cân bằng bậc.")

# ---------------- Tab 2
with tab2:
    try:
        n, h = impulse_response(b, a, roc, -n_half, n_half)
        hf = go.Figure()
        for part, col, nm in [(np.real(h), "#2e86de", "Re{h(n)}"),
                              (np.imag(h), "#e67e22", "Im{h(n)}")]:
            if nm.startswith("Im") and np.all(np.abs(part) < 1e-9):
                continue
            xs, ys = [], []
            for ni, yi in zip(n, part):
                xs += [ni, ni, None]
                ys += [0, yi, None]
            hf.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=col),
                                    showlegend=False))
            hf.add_trace(go.Scatter(x=n, y=part, mode="markers", marker=dict(color=col),
                                    name=nm))
        hf.update_layout(height=450, xaxis_title="n", yaxis_title="h(n)")
        st.plotly_chart(hf, use_container_width=True)
        if causal:
            st.info("ROC bên ngoài mọi cực → h(n) = 0 với n < 0 (nhân quả).")
        elif roc[0] == 0:
            st.info("ROC bên trong mọi cực → h(n) = 0 với n ≥ 0 (phản nhân quả).")
        else:
            st.info("ROC dạng vành khăn → h(n) hai phía (có cả n<0 và n≥0).")
        if not stable:
            st.warning("Hệ không ổn định: |h(n)| tăng không giới hạn khi |n| → ∞ theo hướng có cực ngoài ROC.")
    except Exception as e:
        st.error(f"Không tính được h(n): {e}")

# ---------------- Tab 3
with tab3:
    w, H = freq_response(b, a)
    mag = np.abs(H)
    mag_plot = 20 * np.log10(np.maximum(mag, 1e-6)) if db_scale else np.where(np.isfinite(mag), mag, np.nan)
    ph = np.unwrap(np.angle(np.where(np.isfinite(H), H, 1)))
    x = w / np.pi
    f1 = go.Figure(go.Scatter(x=x, y=mag_plot, line=dict(color="#2e86de")))
    f1.update_layout(height=330, title="Đáp ứng biên độ |H(e^{jω})|", xaxis_title="ω/π (rad/mẫu)",
                     yaxis_title="dB" if db_scale else "|H|")
    f2 = go.Figure(go.Scatter(x=x, y=np.degrees(ph), line=dict(color="#c0392b")))
    f2.update_layout(height=330, title="Đáp ứng pha ∠H(e^{jω}) (đã unwrap)",
                     xaxis_title="ω/π (rad/mẫu)", yaxis_title="độ")
    st.plotly_chart(f1, use_container_width=True)
    st.plotly_chart(f2, use_container_width=True)

    st.subheader("Nhận xét tự động")
    notes = []
    for p in poles:
        if abs(p) > 1e-6 and abs(p) > 0.6:
            notes.append(f"Cực |p|={abs(p):.3f} ∠{np.degrees(np.angle(p)):.1f}° → đỉnh cộng hưởng gần "
                         f"ω ≈ {np.angle(p)/np.pi:.3f}π; càng gần vòng tròn đơn vị đỉnh càng nhọn.")
    for z in zeros:
        if abs(abs(z) - 1) < 1e-3:
            notes.append(f"Zero trên vòng tròn đơn vị ∠{np.degrees(np.angle(z)):.1f}° → |H| = 0 "
                         f"tại ω = {np.angle(z)/np.pi:.3f}π (triệt tần số này).")
    st.write("\n\n".join("• " + s for s in notes) if notes else "Không có đặc điểm nổi bật.")

# ---------------- Tab 4
with tab4:
    st.markdown(r"""
**Biến đổi Z:** $H(z)=\sum_{n}h(n)z^{-n}$ — chỉ xác định trên **ROC**.

**Quy tắc ROC**
1. ROC là hình tròn / vành khăn tâm O, **không chứa cực nào**.
2. Nhân quả ⇔ ROC là miền $|z|>r_{max}$.  Phản nhân quả ⇔ $|z|<r_{min}$.
3. **Ổn định BIBO ⇔ ROC chứa vòng tròn đơn vị** $|z|=1$.
4. Hệ nhân quả ổn định ⇔ mọi cực nằm **trong** vòng tròn đơn vị.

**Đáp ứng tần số:** $H(e^{j\omega})=H(z)\big|_{z=e^{j\omega}}$ (chỉ khi ROC chứa $|z|=1$).
Về hình học: $|H(e^{j\omega})|=|G|\dfrac{\prod|e^{j\omega}-z_k|}{\prod|e^{j\omega}-p_k|}$ —
điểm $e^{j\omega}$ lại gần cực thì biên độ tăng, lại gần zero thì biên độ giảm.

**h(n)** lấy từ khai triển phân thức từng phần: số hạng $\frac{r}{1-pz^{-1}}$ →
$r\,p^n u(n)$ nếu cực nằm trong biên trong của ROC, hoặc $-r\,p^n u(-n-1)$ nếu nằm ngoài.
""")
