"""
TF-AdamW: numerical kernel study, SOE fitting, and synthetic optimization experiments.
Generates publication-quality figures and a LaTeX-ready epsilon_SOE table.

All randomness is seeded; results are reproducible.
"""
import numpy as np
import math
import json
import os
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9.5,
    "lines.linewidth": 1.8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "serif",
    "mathtext.fontset": "cm",
})

OUT = r"C:\Users\omar\tfadamw_work\figures"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Tempered fractional kernel
# ---------------------------------------------------------------------------
def a_coeffs(alpha, J):
    """Generalized binomial / fractional-integration coefficients a_j^{(alpha)}."""
    a = np.empty(J + 1)
    a[0] = 1.0
    for j in range(1, J + 1):
        a[j] = a[j - 1] * (j - 1 + alpha) / j
    return a

def tempered_kernel(alpha, lam, J):
    j = np.arange(J + 1)
    return a_coeffs(alpha, J) * np.exp(-lam * j)

def d_const(alpha, lam):
    return (1.0 - math.exp(-lam)) ** (-alpha)

def mean_delay(alpha, lam):
    return alpha * math.exp(-lam) / (1.0 - math.exp(-lam))

# ---------------------------------------------------------------------------
# 2. Sum-of-exponentials (SOE) fit of the tempered kernel
#    minimize  sum_j ( kappa_j - sum_m c_m rho_m^j )^2 over c_m, rho_m in (0,1)
# ---------------------------------------------------------------------------
def soe_fit(alpha, lam, M, J=4000, seed=0):
    kappa = tempered_kernel(alpha, lam, J)
    j = np.arange(J + 1)

    # parameterize rho_m = sigmoid(t_m) in (0,1); solve c_m by linear LS inside residual
    rng = np.random.default_rng(seed)

    def unpack(t):
        rho = 1.0 / (1.0 + np.exp(-t))  # in (0,1)
        return rho

    def residual(t):
        rho = unpack(t)
        # basis matrix Phi[j,m] = rho_m^j
        Phi = rho[None, :] ** j[:, None]
        # solve least squares for c
        c, *_ = np.linalg.lstsq(Phi, kappa, rcond=None)
        r = Phi @ c - kappa
        return r

    # initialize rho on a geometric grid spanning fast..slow modes
    base = np.exp(-lam)
    init_rho = np.clip(np.linspace(0.5 * base, 0.5 + 0.5 * base, M), 1e-3, 1 - 1e-4)
    # spread across decades
    init_rho = np.clip(1 - np.geomspace(1 - 0.999, 1 - 0.2, M), 1e-3, 1 - 1e-5)
    t0 = np.log(init_rho / (1 - init_rho))

    best = None
    for trial in range(6):
        if trial > 0:
            t0 = np.log(np.clip(rng.uniform(0.05, 0.999, M), 1e-3, 1 - 1e-4) /
                        (1 - np.clip(rng.uniform(0.05, 0.999, M), 1e-3, 1 - 1e-4)))
        try:
            sol = least_squares(residual, t0, method="lm", max_nfev=20000)
        except Exception:
            continue
        rho = unpack(sol.x)
        Phi = rho[None, :] ** j[:, None]
        c, *_ = np.linalg.lstsq(Phi, kappa, rcond=None)
        approx = Phi @ c
        l1 = np.sum(np.abs(approx - kappa))
        if best is None or l1 < best[0]:
            best = (l1, rho.copy(), c.copy())
    l1, rho, c = best
    approx = (rho[None, :] ** j[:, None]) @ c
    linf = np.max(np.abs(approx - kappa))
    rel_l1 = l1 / np.sum(np.abs(kappa))
    return dict(rho=rho, c=c, eps_l1=l1, eps_linf=linf, rel_l1=rel_l1,
                kappa=kappa, approx=approx, j=j)

# ---------------------------------------------------------------------------
# 3. Optimizers (numpy). Each takes a closure grad(theta, rng) -> stochastic grad.
# ---------------------------------------------------------------------------
def run_optimizer(name, grad_fn, loss_fn, theta0, n_iter, hyper, seed=0):
    rng = np.random.default_rng(seed)
    theta = theta0.copy()
    p = theta.size
    eta = hyper["eta"]
    b2 = hyper.get("beta2", 0.999)
    eps = hyper.get("eps", 1e-8)
    wd = hyper.get("wd", 0.0)
    amsgrad = hyper.get("amsgrad", False)
    v = np.zeros(p); vhat_max = np.zeros(p)
    losses = []

    if name in ("adamw", "amsgrad"):
        b1 = hyper.get("beta1", 0.9)
        m = np.zeros(p)
        for k in range(1, n_iter + 1):
            g = grad_fn(theta, rng)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mhat = m / (1 - b1 ** k)
            vh = v / (1 - b2 ** k)
            if name == "amsgrad":
                vhat_max = np.maximum(vhat_max, vh); denom = np.sqrt(vhat_max) + eps
            else:
                denom = np.sqrt(vh) + eps
            theta = theta - eta * mhat / denom - eta * wd * theta
            losses.append(loss_fn(theta))

    elif name == "rmsprop":
        for k in range(1, n_iter + 1):
            g = grad_fn(theta, rng)
            v = b2 * v + (1 - b2) * g * g
            theta = theta - eta * g / (np.sqrt(v) + eps) - eta * wd * theta
            losses.append(loss_fn(theta))

    elif name == "sgdm":
        b1 = hyper.get("beta1", 0.9); m = np.zeros(p)
        for k in range(1, n_iter + 1):
            g = grad_fn(theta, rng)
            m = b1 * m + g
            theta = theta - eta * m - eta * wd * theta
            losses.append(loss_fn(theta))

    elif name in ("tf-adamw", "soe-tf-adamw"):
        alpha = hyper["alpha"]; lam = hyper["lam"]
        if name == "tf-adamw":
            J = hyper.get("J", 400)
            kappa = tempered_kernel(alpha, lam, J)
            from collections import deque
            buf = deque(maxlen=J + 1)
        else:
            fit = hyper["fit"]; rho = fit["rho"]; c = fit["c"]
            s = np.zeros((rho.size, p))
        for k in range(1, n_iter + 1):
            g = grad_fn(theta, rng)
            if name == "tf-adamw":
                buf.appendleft(g.copy())
                L = len(buf)
                kk = kappa[:L]
                Dk = kk.sum()
                m = np.tensordot(kk, np.array(buf), axes=(0, 0))
                mbar = m / Dk
            else:
                s = rho[:, None] * s + g[None, :]
                m = (c[:, None] * s).sum(axis=0)
                # running normalizer D_k of SOE kernel
                Dk = np.sum(c * (1 - rho ** k) / (1 - rho))
                mbar = m / Dk
            v = b2 * v + (1 - b2) * g * g
            vh = v / (1 - b2 ** k)
            if amsgrad:
                vhat_max = np.maximum(vhat_max, vh); denom = np.sqrt(vhat_max) + eps
            else:
                denom = np.sqrt(vh) + eps
            theta = theta - eta * mbar / denom - eta * wd * theta
            losses.append(loss_fn(theta))
    else:
        raise ValueError(name)
    return np.array(losses), theta

# ---------------------------------------------------------------------------
# 4. Problems
# ---------------------------------------------------------------------------
def make_quadratic(p=100, cond=1e4, seed=1):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    eig = np.geomspace(1.0, cond, p)
    A = (Q * eig) @ Q.T
    A = 0.5 * (A + A.T)
    xstar = rng.standard_normal(p)
    b = A @ xstar
    fstar = 0.5 * xstar @ A @ xstar - b @ xstar
    def loss(x): return 0.5 * x @ A @ x - b @ x - fstar
    def grad(x, rng_, noise=0.0):
        g = A @ x - b
        if noise > 0:
            g = g + noise * rng_.standard_normal(x.size) * (np.linalg.norm(g) / math.sqrt(x.size) + 1e-8)
        return g
    return loss, grad, xstar, A

def make_logistic(n=2000, p=80, corr=0.95, seed=2):
    rng = np.random.default_rng(seed)
    # correlated design via AR(1)-like covariance
    cov = corr ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    Lc = np.linalg.cholesky(cov)
    X = rng.standard_normal((n, p)) @ Lc.T
    w_true = rng.standard_normal(p)
    logits = X @ w_true
    probs = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=n) < probs).astype(float)
    reg = 1e-3
    def loss(w):
        z = X @ w
        ll = np.mean(np.logaddexp(0, z) - y * z)
        return ll + 0.5 * reg * w @ w
    def grad(w, rng_, batch=256):
        idx = rng_.integers(0, n, size=batch)
        Xb = X[idx]; yb = y[idx]
        z = Xb @ w
        pr = 1 / (1 + np.exp(-z))
        g = Xb.T @ (pr - yb) / batch + reg * w
        return g
    return loss, grad, w_true

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def multi_seed(name, grad_fn, loss_fn, theta0, n_iter, hyper, seeds):
    runs = []
    for s in seeds:
        L, _ = run_optimizer(name, grad_fn, loss_fn, theta0, n_iter, hyper, seed=s)
        runs.append(L)
    R = np.array(runs)
    return R.mean(0), R.std(0)

print("module loaded")
