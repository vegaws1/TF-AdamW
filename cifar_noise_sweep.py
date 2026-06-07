"""CIFAR-10 label-noise robustness sweep: test accuracy vs symmetric label-noise
rate for AdamW, F-Adam (no tempering), and SOE-TF-AdamW. This directly probes the
regime the theory targets (Prop. 5.4): ties at low noise, divergence as noise
grows. Fair per-(method,noise) LR selection on a short probe, then 3 seeds.
"""
import os, json, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from cifar_experiments import load_cifar, train, OUT
from torch_opts import soe_fit_nnls

plt.rcParams.update({"font.size":11,"axes.labelsize":12,"legend.fontsize":10,
    "lines.linewidth":1.9,"figure.dpi":150,"savefig.bbox":"tight","axes.grid":True,
    "grid.alpha":0.3,"font.family":"serif","mathtext.fontset":"cm"})

METHODS = ["AdamW", "F-Adam", "SOE-TF-AdamW"]
GRID = {"AdamW": [5e-4, 1e-3], "F-Adam": [5e-4, 1e-3], "SOE-TF-AdamW": [5e-4, 1e-3]}
NOISES = [0.0, 0.2, 0.4, 0.6, 0.8]
COL = {"AdamW": "#1f77b4", "F-Adam": "#17becf", "SOE-TF-AdamW": "#d62728"}
LS = {"AdamW": "-", "F-Adam": "--", "SOE-TF-AdamW": "-"}

def run():
    t0 = time.time(); data = load_cifar(); fit8 = soe_fit_nnls(0.7, 0.05, 8)
    print("SOE eps", fit8["eps_l1"], flush=True)
    res = {m: {"noise": [], "mean": [], "std": [], "lr": []} for m in METHODS}
    for noise in NOISES:
        for m in METHODS:
            # LR selection: short 5-epoch probe on seed 0
            best_lr, best_val = None, -1
            for lr in GRID[m]:
                r = train(m, lr, data, noise, 0, 5, fit8)
                if r["val"] > best_val: best_val, best_lr = r["val"], lr
            # final: 3 seeds at best LR, full 12 epochs
            tests = [train(m, best_lr, data, noise, s, 12, fit8)["test"] for s in range(3)]
            res[m]["noise"].append(noise); res[m]["mean"].append(float(np.mean(tests)))
            res[m]["std"].append(float(np.std(tests))); res[m]["lr"].append(best_lr)
            print(f"noise={noise} {m:14s} lr={best_lr:.0e} acc={np.mean(tests)*100:.2f}+-{np.std(tests)*100:.2f} [{int(time.time()-t0)}s]", flush=True)
    json.dump(res, open(os.path.join(OUT, "cifar_sweep.json"), "w"), indent=2)
    # figure: accuracy vs noise
    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    for m in METHODS:
        x = [n * 100 for n in res[m]["noise"]]; y = [v * 100 for v in res[m]["mean"]]
        e = [v * 100 for v in res[m]["std"]]
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, color=COL[m], ls=LS[m], label=m)
    ax.set_xlabel(r"symmetric label-noise rate (\%)")
    ax.set_ylabel(r"test accuracy (\%)")
    ax.set_title("CIFAR-10: robustness to label noise\n(best LR/method, 3 seeds)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_cifar_noise.pdf")); plt.close(fig)
    print("CIFAR SWEEP DONE", int(time.time() - t0), "s", flush=True)

if __name__ == "__main__":
    run()
