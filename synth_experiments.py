"""Honest, fairly-tuned synthetic experiments for TF-AdamW.
Each optimizer gets its OWN best learning rate from a shared grid (selected on the
reported metric over the seeds). Primary methods use the vanilla (non-AMSGrad)
denominator (AMSGrad is reported as an ablation). Produces figures + JSON tables.
"""
import numpy as np, json, os, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tfadamw import (tempered_kernel, a_coeffs, d_const, mean_delay, soe_fit,
                     run_optimizer, make_quadratic, make_logistic, multi_seed, OUT)

plt.rcParams.update({"font.size":11,"axes.labelsize":12,"legend.fontsize":9,
    "lines.linewidth":1.7,"figure.dpi":150,"savefig.bbox":"tight","axes.grid":True,
    "grid.alpha":0.3,"font.family":"serif","mathtext.fontset":"cm"})
COL = {"AdamW":"#1f77b4","AMSGrad":"#ff7f0e","RMSprop":"#2ca02c","SGD-m":"#9467bd",
       "F-Adam":"#17becf","TF-AdamW (exact)":"#9467bd","TF-AdamW":"#d62728",
       "SOE-TF-AdamW":"#d62728","TF-AdamW(AMS)":"#e377c2"}
LRGRID = [3e-3, 1e-2, 3e-2, 1e-1]

def configs(fit):
    """method -> (run_optimizer key, extra hyperparams). Vanilla denominators.
    Includes BOTH the exact truncated-kernel TF-AdamW and the scalable SOE-TF-AdamW
    so their coincidence (Lemma 2) is visible in situ."""
    return {
        "AdamW":            ("adamw",        dict(beta1=0.9, wd=0.0)),
        "AMSGrad":          ("amsgrad",      dict(beta1=0.9, wd=0.0)),
        "RMSprop":          ("rmsprop",      dict()),
        "F-Adam":           ("tf-adamw",     dict(alpha=0.7, lam=0.0, J=64, amsgrad=False)),  # no tempering
        "TF-AdamW (exact)": ("tf-adamw",     dict(alpha=0.7, lam=0.05, J=200, amsgrad=False)),
        "SOE-TF-AdamW":     ("soe-tf-adamw", dict(alpha=0.7, lam=0.05, fit=fit, amsgrad=False)),
    }

def select_lr(method_key, extra, gfn, loss, th0, nit, seeds, grid=LRGRID, base=None):
    base = base or dict(beta2=0.999, eps=1e-8)
    best = (None, np.inf, 0.0, None)
    for lr in grid:
        h = dict(base, eta=lr, **extra)
        runs = [run_optimizer(method_key, gfn, loss, th0, nit, h, seed=s)[0] for s in seeds]
        R = np.array(runs); fin = np.mean(R[:, -200:], axis=1)
        if np.mean(fin) < best[1]:
            best = (lr, float(np.mean(fin)), float(np.std(fin)), R)
    return best  # lr, mean_final, std_final, curves(seeds x iters)

# ---------------- Figure: kernel (regenerate) ----------------
def fig_kernel():
    J=300; j=np.arange(J+1)
    fig,ax=plt.subplots(1,2,figsize=(9.6,3.5))
    alpha=0.7
    for lam,ls in [(0.0,"--"),(0.02,"-"),(0.05,"-"),(0.15,"-")]:
        k=a_coeffs(alpha,J)*np.exp(-lam*j)
        lbl=(r"$\lambda=0$ (power law)" if lam==0 else rf"$\lambda={lam}$")
        ax[0].loglog(j[1:],k[1:],ls,label=lbl)
    ax[0].set_xlabel(r"lag $j$"); ax[0].set_ylabel(r"$\kappa_j^{(\alpha,\lambda)}$")
    ax[0].set_title(rf"Tempered kernel, $\alpha={alpha}$"); ax[0].legend(frameon=False)
    lam=0.05
    for alpha in [0.2,0.4,0.6,0.8,0.95]:
        k=a_coeffs(alpha,J)*np.exp(-lam*j); ax[1].loglog(j[1:],k[1:],"-",label=rf"$\alpha={alpha}$")
    ax[1].set_xlabel(r"lag $j$"); ax[1].set_ylabel(r"$\kappa_j^{(\alpha,\lambda)}$")
    ax[1].set_title(rf"Tempered kernel, $\lambda={lam}$"); ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_kernel.pdf")); plt.close(fig); print("fig_kernel")

# ---------------- Figure: SOE + table + coeffs (NNLS positive) ----------------
def fig_soe_and_tables():
    from torch_opts import soe_fit_nnls
    alpha,lam=0.7,0.05; J=300; j=np.arange(J+1); kappa=tempered_kernel(alpha,lam,J)
    fig,ax=plt.subplots(1,2,figsize=(9.6,3.5))
    ax[0].loglog(j[1:],kappa[1:],"k-",lw=2.2,label="exact $\\kappa_j$")
    for M,c_ in [(2,"#1f77b4"),(4,"#2ca02c"),(8,"#d62728"),(16,"#9467bd")]:
        f=soe_fit_nnls(alpha,lam,M); approx=(f["rho"][None,:]**j[:,None])@f["c"]
        ax[0].loglog(j[1:],np.abs(approx[1:]),"--",color=c_,label=rf"SOE $M={M}$")
    ax[0].set_xlabel(r"lag $j$"); ax[0].set_ylabel("kernel value")
    ax[0].set_title(rf"NNLS SOE fit of $\kappa_j^{{({alpha},{lam})}}$"); ax[0].legend(frameon=False)
    Ms=[1,2,3,4,6,8,12]
    for (a_,l_,mk) in [(0.5,0.1,"o"),(0.7,0.05,"s"),(0.9,0.02,"^")]:
        errs=[soe_fit_nnls(a_,l_,M)["eps_l1"] for M in Ms]
        ax[1].semilogy(Ms,errs,mk+"-",label=rf"$\alpha={a_},\lambda={l_}$")
    ax[1].set_xlabel(r"number of exponentials $M$"); ax[1].set_ylabel(r"$\varepsilon_{\mathrm{SOE}}$")
    ax[1].set_title("SOE error (nonnegative fit)"); ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_soe.pdf")); plt.close(fig)
    table={M:soe_fit_nnls(alpha,lam,M)["eps_l1"] for M in Ms}
    # coefficient tables for M=4,8
    coef={}
    for M in [4,8]:
        f=soe_fit_nnls(alpha,lam,M); order=np.argsort(f["rho"])
        coef[M]=dict(rho=[float(x) for x in f["rho"][order]],
                     c=[float(x) for x in f["c"][order]], eps=f["eps_l1"])
    json.dump({"d":d_const(alpha,lam),"table":table,"coef":coef},
              open(os.path.join(OUT,"soe_tables.json"),"w"),indent=2)
    print("fig_soe + tables"); return table, coef

# ---------------- Figure: noisy convergence (fair per-method LR) ----------------
def fig_synth_conv():
    fit=soe_fit(0.7,0.05,8,J=4000,seed=0)
    out={}
    fig,ax=plt.subplots(1,2,figsize=(9.6,3.8))
    # Panel A: noisy ill-conditioned quadratic
    loss,gfull,xs,A=make_quadratic(p=100,cond=1e3,seed=1); th0=np.zeros(100)
    g=lambda th,rng: gfull(th,rng,noise=2.0); nit=1500; seeds=list(range(5))
    def style(nm):
        if nm=="SOE-TF-AdamW": return dict(ls="--", lw=2.4)
        if nm=="TF-AdamW (exact)": return dict(ls="-", lw=1.7)
        return dict(ls="-", lw=1.5)
    res={}
    for nm,(key,extra) in configs(fit).items():
        lr,mfin,sfin,curves=select_lr(key,extra,g,loss,th0,nit,seeds)
        res[nm]=dict(lr=lr,final=mfin,std=sfin)
        mean=curves.mean(0); std=curves.std(0); x=np.arange(1,nit+1)
        ax[0].semilogy(x,np.maximum(mean,1e-12),color=COL[nm],label=f"{nm}",**style(nm))
        ax[0].fill_between(x,np.maximum(mean-std,1e-12),mean+std,color=COL[nm],alpha=0.10)
    ax[0].set_xlabel("iteration $k$"); ax[0].set_ylabel(r"$f(\theta_k)-f^\star$")
    ax[0].set_title("Noisy ill-conditioned quadratic\n($p=100$, $\\kappa(A)=10^3$, $\\nu=2$, best LR/method)")
    ax[0].legend(frameon=False, fontsize=8); out["quadratic_noise2"]=res
    # Panel B: logistic w/ correlated features (stochastic minibatch)
    loss2,grad2,wt=make_logistic(n=3000,p=80,corr=0.95,seed=2); th0b=np.zeros(80); nit2=2000
    res2={}
    fmin=np.inf; curvestore={}
    for nm,(key,extra) in configs(fit).items():
        lr,mfin,sfin,curves=select_lr(key,extra,grad2,loss2,th0b,nit2,seeds,
                                      grid=[1e-2,3e-2,1e-1,3e-1])
        res2[nm]=dict(lr=lr,final=mfin,std=sfin); curvestore[nm]=curves; fmin=min(fmin,curves.min())
    for nm in configs(fit):
        curves=curvestore[nm]; mean=curves.mean(0)-fmin+1e-6; std=curves.std(0); x=np.arange(1,nit2+1)
        ax[1].semilogy(x,mean,color=COL[nm],label=nm,**style(nm)); ax[1].fill_between(x,np.maximum(mean-std,1e-8),mean+std,color=COL[nm],alpha=0.10)
    ax[1].set_xlabel("iteration $k$"); ax[1].set_ylabel(r"training loss $-\min$")
    ax[1].set_title("Logistic, correlated features\n(stochastic, best LR/method, 5 seeds)")
    ax[1].legend(frameon=False); out["logistic"]=res2
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_synth_conv.pdf")); plt.close(fig)
    print("fig_synth_conv"); return out

# ---------------- Figure: robustness sweep + AMSGrad ablation ----------------
def fig_robust():
    fit=soe_fit(0.8,0.04,8,J=4000,seed=0)
    loss,gfull,xs,A=make_quadratic(p=60,cond=1e3,seed=7); th0=np.zeros(60); nit=1500; seeds=list(range(5))
    noises=[0.0,1.0,2.0,4.0,8.0]
    methods={"AdamW":("adamw",dict(beta1=0.9)),
             "RMSprop":("rmsprop",dict()),
             "F-Adam":("tf-adamw",dict(alpha=0.8,lam=0.0,J=64,amsgrad=False)),
             "SOE-TF-AdamW":("soe-tf-adamw",dict(alpha=0.8,lam=0.04,fit=fit,amsgrad=False))}
    fig,ax=plt.subplots(1,2,figsize=(9.6,3.8))
    sweep={}
    for nm,(key,extra) in methods.items():
        ys=[]; es=[]
        for noise in noises:
            g=lambda th,rng: gfull(th,rng,noise=noise)
            lr,mfin,sfin,_=select_lr(key,extra,g,loss,th0,nit,seeds)
            ys.append(mfin); es.append(sfin)
        ax[0].errorbar(noises,np.maximum(ys,1e-12),yerr=es,marker="o",capsize=3,color=COL[nm],label=nm)
        sweep[nm]=dict(noises=noises,final=ys,std=es)
    ax[0].set_yscale("log"); ax[0].set_xlabel(r"gradient-noise level $\nu$")
    ax[0].set_ylabel(r"steady-state $f(\theta_k)-f^\star$")
    ax[0].set_title("Robustness to gradient noise\n(best LR/method, 5 seeds)"); ax[0].legend(frameon=False)
    # AMSGrad ablation: vanilla vs AMSGrad TF-AdamW on noisy quadratic
    g=lambda th,rng: gfull(th,rng,noise=2.0)
    abl={}
    for nm,ams in [("SOE-TF-AdamW",False),("TF-AdamW(AMS)",True)]:
        lr,mfin,sfin,curves=select_lr("soe-tf-adamw",dict(alpha=0.8,lam=0.04,fit=fit,amsgrad=ams),
                                      g,loss,th0,nit,seeds)
        mean=curves.mean(0); x=np.arange(1,nit+1)
        ax[1].semilogy(x,np.maximum(mean,1e-12),color=COL[nm],label=f"{nm}")
        abl[nm]=dict(lr=lr,final=mfin,std=sfin)
    ax[1].set_xlabel("iteration $k$"); ax[1].set_ylabel(r"$f(\theta_k)-f^\star$")
    ax[1].set_title("AMSGrad ablation ($\\nu=2$)\nvanilla vs AMSGrad-stabilized"); ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_robust.pdf")); plt.close(fig)
    print("fig_robust"); return sweep, abl

# ---------------- Figure: ablation alpha/lambda ----------------
def fig_ablation():
    loss,grad,wt=make_logistic(n=3000,p=80,corr=0.95,seed=2); th0=np.zeros(80); nit=1500; seeds=[0,1,2]
    fig,ax=plt.subplots(1,2,figsize=(9.6,3.6))
    lam=0.08
    for a_ in [0.2,0.4,0.6,0.8,0.95]:
        fit=soe_fit(a_,lam,8,J=4000,seed=0)
        h=dict(eta=5e-2,beta2=0.99,eps=1e-8,alpha=a_,lam=lam,fit=fit,amsgrad=False)
        mean,_=multi_seed("soe-tf-adamw",grad,loss,th0,nit,h,seeds)
        ax[0].semilogy(np.arange(1,nit+1),mean-mean.min()+1e-6,label=rf"$\alpha={a_}$")
    ax[0].set_xlabel("iteration $k$"); ax[0].set_ylabel("training loss $-\\min$")
    ax[0].set_title(rf"Effect of $\alpha$ ($\lambda={lam}$)"); ax[0].legend(frameon=False)
    a_=0.7
    for lam in [0.02,0.05,0.1,0.3,1.0]:
        fit=soe_fit(a_,lam,8,J=4000,seed=0)
        h=dict(eta=5e-2,beta2=0.99,eps=1e-8,alpha=a_,lam=lam,fit=fit,amsgrad=False)
        mean,_=multi_seed("soe-tf-adamw",grad,loss,th0,nit,h,seeds)
        ax[1].semilogy(np.arange(1,nit+1),mean-mean.min()+1e-6,
                       label=rf"$\lambda={lam}$ ($\mu\!\approx\!{mean_delay(a_,lam):.1f}$)")
    ax[1].set_xlabel("iteration $k$"); ax[1].set_ylabel("training loss $-\\min$")
    ax[1].set_title(rf"Effect of $\lambda$ ($\alpha={a_}$)"); ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_ablation.pdf")); plt.close(fig); print("fig_ablation")

if __name__=="__main__":
    t0=time.time()
    fig_kernel()
    fig_soe_and_tables()
    conv=fig_synth_conv()
    sweep,abl=fig_robust()
    fig_ablation()
    json.dump({"conv":conv,"sweep":sweep,"amsgrad_ablation":abl},
              open(os.path.join(OUT,"synth_results.json"),"w"),indent=2)
    print("ALL synthetic done in",round(time.time()-t0),"s")
