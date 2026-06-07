"""PyTorch optimizers: SOE-TF-AdamW, a no-tempering Fractional-Adam baseline, and Lion.
Baselines Adam/AdamW/AMSGrad/RMSprop come from torch.optim.
"""
import math
import torch
from torch.optim import Optimizer
import numpy as np
from scipy.optimize import nnls


# --- tempered fractional kernel + nonnegative SOE fit (positivity guaranteed) ---
def tempered_kernel(alpha, lam, J):
    a = np.empty(J + 1); a[0] = 1.0
    for j in range(1, J + 1):
        a[j] = a[j - 1] * (j - 1 + alpha) / j
    return a * np.exp(-lam * np.arange(J + 1))

def soe_fit_nnls(alpha, lam, M, J=4000, seed=0):
    """Fit kappa_j ~ sum_m c_m rho_m^j with GUARANTEED c_m >= 0 (NNLS) and
    rho_m in (0,1). Continuous optimization of rho via least_squares, with c
    solved by nonnegative least squares inside the residual."""
    from scipy.optimize import least_squares
    kappa = tempered_kernel(alpha, lam, J); j = np.arange(J + 1)
    rng = np.random.default_rng(seed)
    def unpack(t): return 1.0 / (1.0 + np.exp(-t))
    def solve_c(rho):
        Phi = rho[None, :] ** j[:, None]
        c, _ = nnls(Phi, kappa)
        return Phi, c
    def residual(t):
        rho = unpack(t); Phi, c = solve_c(rho)
        return Phi @ c - kappa
    init = np.clip(1 - np.geomspace(1e-4, 0.9, M), 1e-4, 1 - 1e-5)
    best = None
    for trial in range(5):
        t0 = (np.log(init / (1 - init)) if trial == 0
              else np.log(np.clip(rng.uniform(0.05, 0.999, M), 1e-3, 1 - 1e-4) /
                          (1 - np.clip(rng.uniform(0.05, 0.999, M), 1e-3, 1 - 1e-4))))
        try:
            sol = least_squares(residual, t0, method="lm", max_nfev=4000)
        except Exception:
            continue
        rho = unpack(sol.x); Phi, c = solve_c(rho)
        err = np.sum(np.abs(Phi @ c - kappa))
        if best is None or err < best[0]:
            best = (err, rho.copy(), c.copy())
    err, rho, c = best
    keep = c > 1e-14
    return dict(rho=rho[keep], c=c[keep], eps_l1=float(err))


class SOETFAdamW(Optimizer):
    """SOE-TF-AdamW: AdamW with first moment replaced by a tempered fractional kernel
    realized by M exponential states. Normalizer = exact partial sum D_k (precomputed)."""
    def __init__(self, params, lr=1e-3, alpha=0.7, lam=0.05, M=8, betas2=0.999,
                 eps=1e-8, weight_decay=0.0, amsgrad=False, fit=None):
        if fit is None:
            fit = soe_fit_nnls(alpha, lam, M)
        c = np.asarray(fit["c"], dtype=np.float64); rho = np.asarray(fit["rho"], dtype=np.float64)
        defaults = dict(lr=lr, beta2=betas2, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad, c=c, rho=rho, alpha=alpha, lam=lam)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            c = group["c"]; rho = group["rho"]; M = len(c)
            b2 = group["beta2"]; eps = group["eps"]; lr = group["lr"]
            wd = group["weight_decay"]; ams = group["amsgrad"]
            ct = [float(x) for x in c]; rt = [float(x) for x in rho]
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["k"] = 0
                    st["v"] = torch.zeros_like(p)
                    if ams: st["vmax"] = torch.zeros_like(p)
                    st["s"] = [torch.zeros_like(p) for _ in range(M)]
                st["k"] += 1; k = st["k"]
                # SOE states and first moment
                m = torch.zeros_like(p)
                for i in range(M):
                    st["s"][i].mul_(rt[i]).add_(g)
                    m.add_(st["s"][i], alpha=ct[i])
                # exact normalizer D_k = sum_{j=0}^{k-1} kappa_j  (approx by SOE mass partial sum)
                Dk = sum(ct[i] * (1 - rt[i] ** k) / (1 - rt[i]) for i in range(M))
                mbar = m / Dk
                # second moment (raw gradient), bias-corrected, optional AMSGrad
                st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
                vhat = st["v"] / (1 - b2 ** k)
                if ams:
                    torch.maximum(st["vmax"], vhat, out=st["vmax"]); denom = st["vmax"].sqrt().add_(eps)
                else:
                    denom = vhat.sqrt().add_(eps)
                if wd != 0: p.mul_(1 - lr * wd)            # decoupled weight decay
                p.addcdiv_(mbar, denom, value=-lr)
        return loss


class FracAdam(Optimizer):
    """Fractional-Adam baseline (no tempering): first moment is a truncated
    Gruenwald-Letnikov fractional integral of gradients (kernel w_j^{(alpha)},
    lambda=0), normalized, with AdamW preconditioning. This is TF-AdamW WITHOUT
    tempering, realized by a truncated kernel buffer of length J."""
    def __init__(self, params, lr=1e-3, alpha=0.7, J=64, betas2=0.999, eps=1e-8,
                 weight_decay=0.0):
        # power-law (untempered) coefficients, truncated; normalized by partial sum
        a = np.empty(J); a[0] = 1.0
        for j in range(1, J): a[j] = a[j - 1] * (j - 1 + alpha) / j
        defaults = dict(lr=lr, beta2=betas2, eps=eps, weight_decay=weight_decay,
                        w=a.astype(np.float64), J=J)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            w = group["w"]; J = group["J"]; b2 = group["beta2"]; eps = group["eps"]
            lr = group["lr"]; wd = group["weight_decay"]; wt = [float(x) for x in w]
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad; st = self.state[p]
                if len(st) == 0:
                    st["k"] = 0; st["v"] = torch.zeros_like(p)
                    st["buf"] = [torch.zeros_like(p) for _ in range(J)]  # ring buffer of recent grads
                    st["pos"] = 0
                st["k"] += 1; k = st["k"]
                st["buf"][st["pos"]] = g.clone(); st["pos"] = (st["pos"] + 1) % J
                L = min(k, J); m = torch.zeros_like(p); D = 0.0
                for j in range(L):
                    idx = (st["pos"] - 1 - j) % J
                    m.add_(st["buf"][idx], alpha=wt[j]); D += wt[j]
                mbar = m / D
                st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
                vhat = st["v"] / (1 - b2 ** k); denom = vhat.sqrt().add_(eps)
                if wd != 0: p.mul_(1 - lr * wd)
                p.addcdiv_(mbar, denom, value=-lr)
        return loss


class Lion(Optimizer):
    """Lion optimizer (Chen et al., 2023): sign of interpolated momentum."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]; lr = group["lr"]; wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad; st = self.state[p]
                if len(st) == 0: st["m"] = torch.zeros_like(p)
                if wd != 0: p.mul_(1 - lr * wd)
                update = st["m"].mul(b1).add(g, alpha=1 - b1).sign_()
                p.add_(update, alpha=-lr)
                st["m"].mul_(b2).add_(g, alpha=1 - b2)
        return loss
