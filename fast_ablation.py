"""Fast ablation figure: cheaper SOE fit (fewer restarts) since the optimizer is
robust to small kernel-approximation error."""
import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tfadamw import tempered_kernel, mean_delay, run_optimizer, make_logistic, multi_seed, OUT
from scipy.optimize import least_squares

plt.rcParams.update({"font.size":11,"axes.labelsize":12,"legend.fontsize":9.5,
    "lines.linewidth":1.8,"figure.dpi":150,"savefig.bbox":"tight","axes.grid":True,
    "grid.alpha":0.3,"font.family":"serif","mathtext.fontset":"cm"})

def soe_fit_fast(alpha, lam, M, J=1200):
    kappa = tempered_kernel(alpha, lam, J); j = np.arange(J+1)
    def unpack(t): return 1.0/(1.0+np.exp(-t))
    def residual(t):
        rho = unpack(t); Phi = rho[None,:]**j[:,None]
        c,*_ = np.linalg.lstsq(Phi, kappa, rcond=None)
        return Phi@c - kappa
    init_rho = np.clip(1 - np.geomspace(1-0.999, 1-0.2, M), 1e-3, 1-1e-5)
    t0 = np.log(init_rho/(1-init_rho))
    sol = least_squares(residual, t0, method="lm", max_nfev=3000)
    rho = unpack(sol.x); Phi = rho[None,:]**j[:,None]
    c,*_ = np.linalg.lstsq(Phi, kappa, rcond=None)
    l1 = np.sum(np.abs(Phi@c - kappa))
    return dict(rho=rho, c=c, eps_l1=l1)

loss, grad, w_true = make_logistic(n=3000, p=80, corr=0.95, seed=2)
p=80; theta0=np.zeros(p); n_iter=1500; seeds=[0,1,2]
fig, ax = plt.subplots(1,2, figsize=(9.6,3.7))

lam=0.08
for a_ in [0.2,0.4,0.6,0.8,0.95]:
    fit=soe_fit_fast(a_,lam,8)
    hyp=dict(eta=5e-2,beta2=0.99,eps=1e-8,alpha=a_,lam=lam,fit=fit,amsgrad=True)
    mean,_=multi_seed("soe-tf-adamw",grad,loss,theta0,n_iter,hyp,seeds)
    ax[0].semilogy(np.arange(1,n_iter+1), mean-mean.min()+1e-6, label=rf"$\alpha={a_}$")
ax[0].set_xlabel("iteration $k$"); ax[0].set_ylabel("training loss $-\\min$")
ax[0].set_title(rf"Effect of $\alpha$ ($\lambda={lam}$)"); ax[0].legend(frameon=False)

a_=0.7
for lam in [0.02,0.05,0.1,0.3,1.0]:
    fit=soe_fit_fast(a_,lam,8)
    hyp=dict(eta=5e-2,beta2=0.99,eps=1e-8,alpha=a_,lam=lam,fit=fit,amsgrad=True)
    mean,_=multi_seed("soe-tf-adamw",grad,loss,theta0,n_iter,hyp,seeds)
    mw=mean_delay(a_,lam)
    ax[1].semilogy(np.arange(1,n_iter+1), mean-mean.min()+1e-6,
                   label=rf"$\lambda={lam}$ ($\mu\!\approx\!{mw:.1f}$)")
ax[1].set_xlabel("iteration $k$"); ax[1].set_ylabel("training loss $-\\min$")
ax[1].set_title(rf"Effect of $\lambda$ ($\alpha={a_}$)"); ax[1].legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_ablation.pdf")); plt.close(fig)
print("saved fig_ablation")
