"""CIFAR-10 deep-learning experiment (CPU torch): clean + 40% label-noise
robustness. Focused, fairly-tuned comparison; a real color-image benchmark
stronger than Fashion-MNIST. Reduced protocol for CPU feasibility."""
import os, sys, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_opts import SOETFAdamW, FracAdam, Lion, soe_fit_nnls
torch.set_num_threads(max(1, os.cpu_count() - 1))
WORK = r"C:\Users\omar\tfadamw_work"; OUT = os.path.join(WORK, "figures")

def load_cifar(n_train=15000, n_val=3000, n_test=5000, seed=0):
    d = np.load(os.path.join(WORK, "cifar10.npz"))
    X = d["Xtr"].reshape(-1, 3, 32, 32).astype("float32") / 255.0
    y = d["ytr"].astype("int64")
    rng = np.random.default_rng(seed); perm = rng.permutation(len(X)); X, y = X[perm], y[perm]
    mean = X[:n_train].mean((0, 2, 3), keepdims=True); std = X[:n_train].std((0, 2, 3), keepdims=True) + 1e-6
    X = ((X - mean) / std).astype("float32")
    sl = lambda a, b: (X[a:b], y[a:b])
    return sl(0, n_train), sl(n_train, n_train + n_val), sl(n_train + n_val, n_train + n_val + n_test)

def add_label_noise(y, rate, seed=0):
    rng = np.random.default_rng(seed); y = y.copy(); flip = rng.random(len(y)) < rate
    y[flip] = rng.integers(0, 10, flip.sum()); return y

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1); self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 128); self.fc2 = nn.Linear(128, 10); self.drop = nn.Dropout(0.25)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2); x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.max_pool2d(F.relu(self.c3(x)), 2); x = x.flatten(1)
        return self.fc2(self.drop(F.relu(self.fc1(x))))

def make_opt(name, params, lr, fit8):
    wd = 5e-4
    if name == "Adam":   return torch.optim.Adam(params, lr=lr)
    if name == "AdamW":  return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "Lion":   return Lion(params, lr=lr, weight_decay=wd)
    if name == "F-Adam": return FracAdam(params, lr=lr, alpha=0.7, J=48, weight_decay=wd)
    if name == "SOE-TF-AdamW": return SOETFAdamW(params, lr=lr, alpha=0.7, lam=0.05, M=8, weight_decay=wd, fit=fit8)
    raise ValueError(name)

def train(name, lr, data, noise, seed, epochs, fit8, batch=128):
    torch.manual_seed(seed); np.random.seed(seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    ytr = add_label_noise(ytr, noise, seed) if noise > 0 else ytr
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
    Xva_t = torch.tensor(Xva); yva_t = torch.tensor(yva); Xte_t = torch.tensor(Xte); yte_t = torch.tensor(yte)
    model = CNN(); opt = make_opt(name, model.parameters(), lr, fit8)
    n = len(Xtr_t); idx = np.arange(n); rng = np.random.default_rng(seed)
    def ev(Xt, yt):
        model.eval()
        with torch.no_grad():
            pr = torch.cat([model(Xt[i:i+1024]).argmax(1) for i in range(0, len(Xt), 1024)])
        return (pr == yt).float().mean().item()
    for ep in range(epochs):
        model.train(); rng.shuffle(idx)
        for b in range(0, n, batch):
            bi = idx[b:b+batch]; opt.zero_grad()
            loss = F.cross_entropy(model(Xtr_t[bi]), ytr_t[bi]); loss.backward(); opt.step()
    return dict(val=ev(Xva_t, yva_t), test=ev(Xte_t, yte_t))

def main():
    t0 = time.time(); data = load_cifar()
    fit8 = soe_fit_nnls(0.7, 0.05, 8); print("SOE eps", fit8["eps_l1"], flush=True)
    methods = ["Adam", "AdamW", "Lion", "F-Adam", "SOE-TF-AdamW"]
    grid = {"Adam": [5e-4, 1e-3], "AdamW": [5e-4, 1e-3], "Lion": [1e-4, 3e-4],
            "F-Adam": [5e-4, 1e-3], "SOE-TF-AdamW": [5e-4, 1e-3]}
    epochs = 12; seeds = [0, 1]; noises = [0.0, 0.4]
    res = {}
    for nz in noises:
        res[str(nz)] = {}
        for m in methods:
            blr, bval = None, -1
            for lr in grid[m]:
                r = train(m, lr, data, nz, 0, epochs, fit8)
                if r["val"] > bval: bval, blr = r["val"], lr
            tests = [train(m, blr, data, nz, s, epochs, fit8)["test"] for s in seeds]
            res[str(nz)][m] = dict(lr=blr, test_mean=float(np.mean(tests)),
                                   test_std=float(np.std(tests)), tests=tests)
            print(f"noise={nz} {m:14s} lr={blr:.0e} test={np.mean(tests)*100:.2f}+-{np.std(tests)*100:.2f} [{int(time.time()-t0)}s]", flush=True)
    json.dump(res, open(os.path.join(OUT, "cifar_results.json"), "w"), indent=2)
    print("CIFAR DONE", int(time.time()-t0), "s", flush=True)

if __name__ == "__main__":
    main()
