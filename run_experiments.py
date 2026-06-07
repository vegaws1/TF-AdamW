"""Run all TF-AdamW experiments and produce figures + epsilon_SOE table."""
import numpy as np, math, json, os
import matplotlib.pyplot as plt
from tfadamw import (tempered_kernel, a_coeffs, d_const, mean_delay, soe_fit,
                     run_optimizer, make_quadratic, make_logistic, multi_seed, OUT)

COL = {
    "AdamW": "#1f77b4", "AMSGrad": "#ff7f0e", "RMSprop": "#2ca02c",
    "SGD-m": "#9467bd", "TF-AdamW": "#d62728", "SOE-TF-AdamW": "#8c564b",
}

# ===========================================================================
# Figure 1: tempered fractional kernel
# ===========================================================================
def fig_kernel():
    J = 300
    j = np.arange(J + 1)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    # left: fixed alpha, varying lambda; show power-law (lambda->0) vs tempered
    alpha = 0.7
    for lam, ls in [(0.0, "--"), (0.02, "-"), (0.05, "-"), (0.15, "-")]:
        k = a_coeffs(alpha, J) * np.exp(-lam * j)
        lbl = (r"$\lambda=0$ (pure power-law)" if lam == 0 else rf"$\lambda={lam}$")
        ax[0].loglog(j[1:], k[1:], ls, label=lbl)
    ax[0].set_xlabel(r"lag $j$"); ax[0].set_ylabel(r"$\kappa_j^{(\alpha,\lambda)}$")
    ax[0].set_title(rf"Tempered kernel, $\alpha={alpha}$")
    ax[0].legend(frameon=False)
    # right: fixed lambda, varying alpha
    lam = 0.05
    for alpha in [0.2, 0.4, 0.6, 0.8, 0.95]:
        k = a_coeffs(alpha, J) * np.exp(-lam * j)
        ax[1].loglog(j[1:], k[1:], "-", label=rf"$\alpha={alpha}$")
    ax[1].set_xlabel(r"lag $j$"); ax[1].set_ylabel(r"$\kappa_j^{(\alpha,\lambda)}$")
    ax[1].set_title(rf"Tempered kernel, $\lambda={lam}$")
    ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_kernel.pdf")); plt.close(fig)
    print("saved fig_kernel")

# ===========================================================================
# Figure 2: SOE fit + epsilon_SOE table
# ===========================================================================
def fig_soe_and_table():
    alpha, lam = 0.7, 0.05
    J = 300
    j = np.arange(J + 1)
    kappa = tempered_kernel(alpha, lam, J)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax[0].loglog(j[1:], kappa[1:], "k-", lw=2.2, label="exact $\\kappa_j$")
    table = {}
    for M, c_ in [(2, "#1f77b4"), (4, "#2ca02c"), (8, "#d62728"), (16, "#9467bd")]:
        fit = soe_fit(alpha, lam, M, J=4000, seed=0)
        approx = (fit["rho"][None, :] ** j[:, None]) @ fit["c"]
        ax[0].loglog(j[1:], np.abs(approx[1:]), "--", color=c_, label=rf"SOE $M={M}$")
        table[M] = dict(eps_l1=float(fit["eps_l1"]), rel_l1=float(fit["rel_l1"]),
                        eps_linf=float(fit["eps_linf"]))
    ax[0].set_xlabel(r"lag $j$"); ax[0].set_ylabel("kernel value")
    ax[0].set_title(rf"SOE fit of $\kappa_j^{{({alpha},{lam})}}$")
    ax[0].legend(frameon=False)
    # right: epsilon_SOE vs M (sweep)
    Ms = [1, 2, 3, 4, 6, 8, 12, 16]
    for (a_, l_, mk) in [(0.5, 0.1, "o"), (0.7, 0.05, "s"), (0.9, 0.02, "^")]:
        errs = []
        for M in Ms:
            fit = soe_fit(a_, l_, M, J=4000, seed=0)
            errs.append(fit["eps_l1"])
        ax[1].semilogy(Ms, errs, mk + "-", label=rf"$\alpha={a_},\lambda={l_}$")
    ax[1].set_xlabel(r"number of exponentials $M$")
    ax[1].set_ylabel(r"$\varepsilon_{\mathrm{SOE}}=\|\kappa-\tilde\kappa\|_1$")
    ax[1].set_title("SOE approximation error")
    ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_soe.pdf")); plt.close(fig)
    # full table for (alpha=0.7,lam=0.05)
    full = {}
    for M in Ms:
        fit = soe_fit(alpha, lam, M, J=4000, seed=0)
        full[M] = dict(eps_l1=float(fit["eps_l1"]), rel_l1=float(fit["rel_l1"]),
                       eps_linf=float(fit["eps_linf"]))
    with open(os.path.join(OUT, "soe_table.json"), "w") as f:
        json.dump(dict(alpha=alpha, lam=lam, d=d_const(alpha, lam), table=full), f, indent=2)
    print("saved fig_soe; SOE table:", json.dumps(full, indent=2))
    return full

# ===========================================================================
# Figure 3: ill-conditioned quadratic
# ===========================================================================
def fig_quadratic():
    p, cond = 100, 1e4
    loss, grad_full, xstar, A = make_quadratic(p=p, cond=cond, seed=1)
    theta0 = np.zeros(p)
    n_iter = 3000
    noise = 0.0  # deterministic, ill-conditioning is the challenge
    def gfn(noise):
        return lambda th, rng: grad_full(th, rng, noise=noise)
    fit8 = soe_fit(0.7, 0.05, 8, J=4000, seed=0)
    seeds = [0]
    base = dict(eta=2e-2, beta2=0.999, eps=1e-12, wd=0.0)
    configs = {
        "AdamW": ("adamw", dict(**base, beta1=0.9)),
        "AMSGrad": ("amsgrad", dict(**base, beta1=0.9)),
        "RMSprop": ("rmsprop", dict(**base)),
        "TF-AdamW": ("tf-adamw", dict(**base, alpha=0.7, lam=0.05, J=400, amsgrad=True)),
        "SOE-TF-AdamW": ("soe-tf-adamw", dict(**base, alpha=0.7, lam=0.05, fit=fit8, amsgrad=True)),
    }
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for nm, (key, hyp) in configs.items():
        L, _ = run_optimizer(key, gfn(noise), loss, theta0, n_iter, hyp, seed=0)
        ax.semilogy(np.arange(1, n_iter + 1), np.maximum(L, 1e-16), color=COL[nm], label=nm)
    ax.set_xlabel("iteration $k$"); ax.set_ylabel(r"$f(\theta_k)-f^\star$")
    ax.set_title(rf"Ill-conditioned quadratic ($\kappa(A)={int(cond)}$, $p={p}$)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_quadratic.pdf")); plt.close(fig)
    print("saved fig_quadratic")

# ===========================================================================
# Figure 4: logistic regression (stochastic, correlated features)
# ===========================================================================
def fig_logistic():
    loss, grad, w_true = make_logistic(n=3000, p=80, corr=0.95, seed=2)
    p = 80; theta0 = np.zeros(p); n_iter = 2000
    fit8 = soe_fit(0.6, 0.08, 8, J=4000, seed=0)
    seeds = list(range(5))
    base = dict(eta=5e-2, beta2=0.99, eps=1e-8, wd=0.0)
    configs = {
        "AdamW": ("adamw", dict(**base, beta1=0.9)),
        "AMSGrad": ("amsgrad", dict(**base, beta1=0.9)),
        "SGD-m": ("sgdm", dict(eta=2e-2, beta1=0.9, wd=0.0)),
        "TF-AdamW": ("tf-adamw", dict(**base, alpha=0.6, lam=0.08, J=300, amsgrad=True)),
        "SOE-TF-AdamW": ("soe-tf-adamw", dict(**base, alpha=0.6, lam=0.08, fit=fit8, amsgrad=True)),
    }
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    fstar = None
    results = {}
    for nm, (key, hyp) in configs.items():
        mean, std = multi_seed(key, grad, loss, theta0, n_iter, hyp, seeds)
        results[nm] = (mean, std)
    fmin = min(r[0].min() for r in results.values())
    for nm, (mean, std) in results.items():
        y = mean - fmin + 1e-6
        x = np.arange(1, n_iter + 1)
        ax.semilogy(x, y, color=COL[nm], label=nm)
        ax.fill_between(x, np.maximum(y - std, 1e-8), y + std, color=COL[nm], alpha=0.15)
    ax.set_xlabel("iteration $k$"); ax.set_ylabel(r"training loss $-\,\min$")
    ax.set_title("Logistic regression, correlated features\n(mean$\\pm$std over 5 seeds)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_logistic.pdf")); plt.close(fig)
    print("saved fig_logistic")

# ===========================================================================
# Figure 5: noise robustness (gradient-noise injection on the quadratic)
# ===========================================================================
def fig_noise():
    p, cond = 60, 1e3
    loss, grad_full, xstar, A = make_quadratic(p=p, cond=cond, seed=7)
    theta0 = np.zeros(p); n_iter = 2000
    noises = [0.0, 0.5, 1.0, 2.0, 4.0]
    fit8 = soe_fit(0.8, 0.04, 8, J=4000, seed=0)
    seeds = list(range(5))
    base = dict(eta=1e-2, beta2=0.999, eps=1e-10, wd=0.0)
    methods = {
        "AdamW": ("adamw", dict(**base, beta1=0.9)),
        "AMSGrad": ("amsgrad", dict(**base, beta1=0.9)),
        "TF-AdamW": ("tf-adamw", dict(**base, alpha=0.8, lam=0.04, J=400, amsgrad=True)),
        "SOE-TF-AdamW": ("soe-tf-adamw", dict(**base, alpha=0.8, lam=0.04, fit=fit8, amsgrad=True)),
    }
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    final = {nm: [] for nm in methods}
    finalstd = {nm: [] for nm in methods}
    for noise in noises:
        gfn = lambda th, rng: grad_full(th, rng, noise=noise)
        for nm, (key, hyp) in methods.items():
            vals = []
            for s in seeds:
                L, _ = run_optimizer(key, gfn, loss, theta0, n_iter, hyp, seed=s)
                vals.append(np.mean(L[-200:]))  # steady-state loss floor
            final[nm].append(np.mean(vals)); finalstd[nm].append(np.std(vals))
    for nm in methods:
        ax.errorbar(noises, final[nm], yerr=finalstd[nm], marker="o", capsize=3,
                    color=COL[nm], label=nm)
    ax.set_yscale("log")
    ax.set_xlabel(r"gradient-noise level $\nu$")
    ax.set_ylabel(r"steady-state $f(\theta_k)-f^\star$")
    ax.set_title("Robustness to gradient noise\n(mean$\\pm$std over 5 seeds)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_noise.pdf")); plt.close(fig)
    print("saved fig_noise")

# ===========================================================================
# Figure 6: ablation over alpha and lambda (final loss heat / curves)
# ===========================================================================
def fig_ablation():
    loss, grad, w_true = make_logistic(n=3000, p=80, corr=0.95, seed=2)
    p = 80; theta0 = np.zeros(p); n_iter = 1500
    seeds = list(range(3))
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.7))
    # vary alpha (lam fixed)
    lam = 0.08
    alphas = [0.2, 0.4, 0.6, 0.8, 0.95]
    for a_ in alphas:
        fit = soe_fit(a_, lam, 8, J=4000, seed=0)
        hyp = dict(eta=5e-2, beta2=0.99, eps=1e-8, alpha=a_, lam=lam, fit=fit, amsgrad=True)
        mean, _ = multi_seed("soe-tf-adamw", grad, loss, theta0, n_iter, hyp, seeds)
        ax[0].semilogy(np.arange(1, n_iter + 1), mean - mean.min() + 1e-6, label=rf"$\alpha={a_}$")
    ax[0].set_xlabel("iteration $k$"); ax[0].set_ylabel("training loss $-\\min$")
    ax[0].set_title(rf"Effect of $\alpha$ ($\lambda={lam}$)"); ax[0].legend(frameon=False)
    # vary lambda (alpha fixed)
    a_ = 0.7
    lams = [0.02, 0.05, 0.1, 0.3, 1.0]
    for lam in lams:
        fit = soe_fit(a_, lam, 8, J=4000, seed=0)
        hyp = dict(eta=5e-2, beta2=0.99, eps=1e-8, alpha=a_, lam=lam, fit=fit, amsgrad=True)
        mean, _ = multi_seed("soe-tf-adamw", grad, loss, theta0, n_iter, hyp, seeds)
        mw = mean_delay(a_, lam)
        ax[1].semilogy(np.arange(1, n_iter + 1), mean - mean.min() + 1e-6,
                       label=rf"$\lambda={lam}$ ($\mu\!\approx\!{mw:.1f}$)")
    ax[1].set_xlabel("iteration $k$"); ax[1].set_ylabel("training loss $-\\min$")
    ax[1].set_title(rf"Effect of $\lambda$ ($\alpha={a_}$)"); ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_ablation.pdf")); plt.close(fig)
    print("saved fig_ablation")

if __name__ == "__main__":
    fig_kernel()
    soe_table = fig_soe_and_table()
    fig_quadratic()
    fig_logistic()
    fig_noise()
    fig_ablation()
    print("ALL DONE")
