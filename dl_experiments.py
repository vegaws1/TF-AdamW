"""Real deep-learning experiments on Fashion-MNIST (CPU torch):
   (E1) clean classification, (E2) symmetric label-noise robustness.
Compares SOE-TF-AdamW against Adam, AdamW, AMSGrad, RMSprop, Lion, and a
no-tempering Fractional-Adam baseline, with per-optimizer LR selection on a
validation split. Reports test accuracy (mean +/- std over seeds) and curves.
"""
import time, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_opts import SOETFAdamW, FracAdam, Lion, soe_fit_nnls

torch.set_num_threads(max(1, os.cpu_count() - 1))
OUT = r"C:\Users\omar\tfadamw_work\figures"
WORK = r"C:\Users\omar\tfadamw_work"

# ----------------------------- data -----------------------------
def load_fmnist(n_train=12000, n_val=3000, n_test=5000, seed=0):
    d = np.load(os.path.join(WORK, "fmnist.npz"))
    X = d["X"].astype("float32") / 255.0
    y = d["y"].astype("int64")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    mu, sd = X[:n_train].mean(), X[:n_train].std() + 1e-6
    X = (X - mu) / sd
    sl = lambda a, b: (X[a:b], y[a:b])
    return sl(0, n_train), sl(n_train, n_train + n_val), \
           sl(n_train + n_val, n_train + n_val + n_test)

def add_label_noise(y, rate, num_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    y = y.copy(); n = len(y)
    flip = rng.random(n) < rate
    y[flip] = rng.integers(0, num_classes, flip.sum())
    return y

# ----------------------------- model -----------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.drop = nn.Dropout(0.25)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

# ----------------------------- training -----------------------------
def make_opt(name, params, lr, fit8=None):
    wd = 5e-4
    if name == "Adam":     return torch.optim.Adam(params, lr=lr, weight_decay=0.0)
    if name == "AdamW":    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "AMSGrad":  return torch.optim.AdamW(params, lr=lr, weight_decay=wd, amsgrad=True)
    if name == "RMSprop":  return torch.optim.RMSprop(params, lr=lr, weight_decay=0.0)
    if name == "Lion":     return Lion(params, lr=lr, weight_decay=wd)
    if name == "F-Adam":   return FracAdam(params, lr=lr, alpha=0.7, J=48, weight_decay=wd)
    if name == "SOE-TF-AdamW":
        return SOETFAdamW(params, lr=lr, alpha=0.7, lam=0.05, M=8, weight_decay=wd, fit=fit8)
    raise ValueError(name)

def train(name, lr, data, noise_rate, seed, epochs, batch=128, fit8=None):
    torch.manual_seed(seed); np.random.seed(seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    ytr = add_label_noise(ytr, noise_rate, seed=seed) if noise_rate > 0 else ytr
    Xtr_t = torch.tensor(Xtr).view(-1, 1, 28, 28)
    ytr_t = torch.tensor(ytr)
    Xva_t = torch.tensor(Xva).view(-1, 1, 28, 28); yva_t = torch.tensor(yva)
    Xte_t = torch.tensor(Xte).view(-1, 1, 28, 28); yte_t = torch.tensor(yte)
    model = SmallCNN()
    opt = make_opt(name, model.parameters(), lr, fit8)
    n = len(Xtr_t); idx = np.arange(n)
    losses = []
    def evaluate(Xt, yt):
        model.eval()
        with torch.no_grad():
            pred = []
            for i in range(0, len(Xt), 1024):
                pred.append(model(Xt[i:i+1024]).argmax(1))
            return (torch.cat(pred) == yt).float().mean().item()
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train(); rng.shuffle(idx); ep_loss = 0.0; nb = 0
        for b in range(0, n, batch):
            bi = idx[b:b+batch]
            xb = Xtr_t[bi]; yb = ytr_t[bi]
            opt.zero_grad()
            out = model(xb); loss = F.cross_entropy(out, yb)
            loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        losses.append(ep_loss / nb)
    return dict(val=evaluate(Xva_t, yva_t), test=evaluate(Xte_t, yte_t), losses=losses)

def main():
    t0 = time.time()
    quick = "--quick" in sys.argv
    data = load_fmnist()
    fit8 = soe_fit_nnls(0.7, 0.05, 8)
    print("SOE fit kept", len(fit8["c"]), "terms, eps=", fit8["eps_l1"])
    methods = ["Adam", "AdamW", "AMSGrad", "RMSprop", "Lion", "F-Adam", "SOE-TF-AdamW"]
    lr_grid = {
        "Adam": [5e-4, 1e-3, 2e-3], "AdamW": [5e-4, 1e-3, 2e-3],
        "AMSGrad": [5e-4, 1e-3, 2e-3], "RMSprop": [3e-4, 1e-3, 2e-3],
        "Lion": [1e-4, 3e-4, 1e-3], "F-Adam": [5e-4, 1e-3, 2e-3],
        "SOE-TF-AdamW": [5e-4, 1e-3, 2e-3],
    }
    epochs = 3 if quick else 12
    seeds = [0] if quick else [0, 1, 2]
    noise_levels = [0.0] if quick else [0.0, 0.4]
    results = {}
    for noise in noise_levels:
        results[noise] = {}
        for m in methods:
            # LR selection on validation using seed 0
            best_lr, best_val = None, -1
            for lr in lr_grid[m]:
                r = train(m, lr, data, noise, 0, epochs, fit8=fit8)
                if r["val"] > best_val: best_val, best_lr = r["val"], lr
            # run all seeds at best lr
            tests, curves = [], []
            for s in seeds:
                r = train(m, best_lr, data, noise, s, epochs, fit8=fit8)
                tests.append(r["test"]); curves.append(r["losses"])
            results[noise][m] = dict(lr=best_lr, test_mean=float(np.mean(tests)),
                                     test_std=float(np.std(tests)), tests=tests,
                                     loss_curve=list(np.mean(curves, 0)))
            print(f"noise={noise} {m:14s} lr={best_lr:.0e} "
                  f"test={np.mean(tests)*100:.2f}+-{np.std(tests)*100:.2f}  "
                  f"[{round(time.time()-t0)}s]")
    with open(os.path.join(OUT, "dl_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("TOTAL", round(time.time()-t0), "s")

if __name__ == "__main__":
    main()
