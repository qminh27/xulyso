"""Lõi tính toán cho hệ LTI rời rạc, dạng H(z) theo z^-1:

    H(z) = (b0 + b1 z^-1 + ... + bM z^-M) / (a0 + a1 z^-1 + ... + aN z^-N)
         = G * prod(1 - z_k z^-1) / prod(1 - p_k z^-1)
"""
import numpy as np
from scipy import signal

EPS = 1e-9


# ---------------------------------------------------------------- parse
def parse_complex_list(text: str) -> np.ndarray:
    """'0.5+0.5j, 0.5-0.5j' -> mảng số phức (chấp nhận cả 'i' thay cho 'j')."""
    items = [t.strip() for t in text.replace(";", ",").split(",") if t.strip()]
    vals = [complex(t.replace(" ", "").replace("i", "j")) for t in items]
    return np.array(vals, dtype=complex)


def clean(x: np.ndarray):
    """Bỏ phần ảo nhỏ do sai số số học."""
    x = np.atleast_1d(x)
    if np.iscomplexobj(x) and np.all(np.abs(x.imag) < 1e-9):
        return x.real
    return x


# ---------------------------------------------------------------- zpk <-> ba
def ba_from_zpk(z, p, k):
    b = clean(k * np.atleast_1d(np.poly(z)))
    a = clean(np.atleast_1d(np.poly(p)))
    return b, a


def zpk_from_ba(b, a):
    b = np.atleast_1d(b).astype(complex)
    a = np.atleast_1d(a).astype(complex)
    if abs(a[0]) < EPS:
        raise ValueError("a[0] phải khác 0.")
    b, a = b / a[0], a / a[0]
    nz = np.nonzero(np.abs(b) > EPS)[0]
    if len(nz) == 0:
        raise ValueError("Tử số bằng 0.")
    d = nz[0]                      # số 0 đầu: b = z^-d * b'  -> d cực tại gốc
    bp = b[d:]
    M, N = len(bp) - 1, len(a) - 1
    z = np.roots(bp)
    p = np.roots(a)
    z = np.r_[z, np.zeros(M - len(z))]
    p = np.r_[p, np.zeros(N - len(p))]
    if N > M:
        z = np.r_[z, np.zeros(N - M)]
    if M > N:
        p = np.r_[p, np.zeros(M - N)]
    p = np.r_[p, np.zeros(d)]
    z = np.r_[z, np.zeros(0)]
    return z, p, bp[0], clean(b), clean(a)


# ---------------------------------------------------------------- ROC
def roc_options(poles):
    """Danh sách các ROC hợp lệ: [(r_in, r_out, nhãn), ...]."""
    radii = np.unique(np.round(np.abs(poles[np.abs(poles) > 1e-6]), 6))
    if len(radii) == 0:
        return [(0.0, np.inf, "Toàn mặt phẳng z (trừ z=0 / z=∞ nếu có)")]
    edges = [0.0, *radii.tolist(), np.inf]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if np.isinf(hi):
            lab = f"|z| > {lo:.4g}  (nhân quả / right-sided)"
        elif lo == 0:
            lab = f"|z| < {hi:.4g}  (phản nhân quả / left-sided)"
        else:
            lab = f"{lo:.4g} < |z| < {hi:.4g}  (hai phía / two-sided)"
        out.append((lo, hi, lab))
    return out


def is_stable(roc):
    return roc[0] < 1.0 < roc[1]


def is_causal(roc):
    return bool(np.isinf(roc[1]))


# ---------------------------------------------------------------- h(n)
def impulse_response(b, a, roc, nmin, nmax):
    """h(n) theo ROC đã chọn, bằng khai triển phân thức từng phần (residuez).
    Cực nằm trong biên trong của ROC -> số hạng bên phải; ngoài -> bên trái."""
    n = np.arange(nmin, nmax + 1)
    r, p, k = signal.residuez(b, a)
    h = np.zeros(len(n), dtype=complex)
    for j, kj in enumerate(k):
        h[n == j] += kj
    lo = roc[0]
    mult = 0
    for i in range(len(p)):
        mult = mult + 1 if (i > 0 and abs(p[i] - p[i - 1]) < 1e-3) else 1
        if abs(p[i]) < 1e-9:                       # cực tại gốc -> xung
            if mult == 1:
                h[n == 0] += r[i]
            continue
        poly = np.ones(len(n))                     # C(n+m-1, m-1)
        for j in range(1, mult):
            poly = poly * (n + j) / j
        pn = np.power(p[i], n.astype(float))
        if abs(p[i]) <= lo + 1e-5:                 # bên phải
            h += r[i] * poly * pn * (n >= 0)
        else:                                      # bên trái
            h += -r[i] * poly * pn * (n <= -1)
    return n, clean(h)


# ---------------------------------------------------------------- H(e^jw)
def freq_response(b, a, n_points=2048):
    w = np.linspace(-np.pi, np.pi, n_points)
    e = np.exp(-1j * w)
    num = np.polyval(np.atleast_1d(b)[::-1], e)
    den = np.polyval(np.atleast_1d(a)[::-1], e)
    with np.errstate(all="ignore"):
        H = num / den
    return w, H


def group_points(vals, ndigits=5):
    """Gộp điểm trùng nhau -> [(giá trị, bội)]."""
    d = {}
    for v in vals:
        key = (round(v.real, ndigits) + 0.0, round(v.imag, ndigits) + 0.0)
        d[key] = d.get(key, 0) + 1
    return [(complex(*k), m) for k, m in d.items()]


def expand_pair(r, theta_deg, conj):
    """Một phần tử (r, θ) -> list; nếu conj và không nằm trên trục thực thì thêm liên hợp."""
    z = r * np.exp(1j * np.deg2rad(theta_deg))
    if conj and abs(z.imag) > 1e-9:
        return [z, np.conj(z)]
    return [z]
