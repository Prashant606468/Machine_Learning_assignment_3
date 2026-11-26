"""Shared utilities: config, seeding, CIFAR10 loading, the saved stratified split,
spatial/frequency transforms, a generic classifier trainer with caching, evaluation
and calibration metrics. Every question script imports from here."""
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
SMOKE = os.environ.get("SMOKE", "0") == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


# ----------------------------------------------------------------------------- config
def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def load_cfg():
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    if SMOKE:
        _merge(cfg, cfg["smoke"])
        cfg["results_dir"] = "results_smoke"
    res = ROOT / cfg["results_dir"]
    for sub in ["logs", "figs", "tables", "ckpt"]:
        (res / sub).mkdir(parents=True, exist_ok=True)
    cfg["_res"] = res
    return cfg


def rpath(cfg, sub, name):
    return cfg["_res"] / sub / name


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=conv)


def set_seed(s):
    random.seed(s)
    np.random.seed(s % 2**32)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


# ------------------------------------------------------------------------------- data
def _fake_cifar(n, seed):
    """Synthetic class-conditional 32x32 images for SMOKE tests only."""
    g = np.random.default_rng(seed)
    tg = np.random.default_rng(0)
    templates = tg.uniform(0, 255, size=(10, 3, 4, 4))
    templates = templates.repeat(8, axis=2).repeat(8, axis=3)
    y = g.integers(0, 10, n)
    x = templates[y] + g.normal(0, 160, size=(n, 3, 32, 32))
    x = np.clip(x, 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)
    return x, y.astype(np.int64)


_CACHE = {}


def load_cifar(cfg):
    """Returns uint8 arrays (N,32,32,3) and int64 labels for the official train/test sets."""
    if "raw" in _CACHE:
        return _CACHE["raw"]
    if SMOKE:
        xtr, ytr = _fake_cifar(3000, 1)
        xte, yte = _fake_cifar(1000, 2)
    else:
        import torchvision
        root = ROOT / cfg["data_root"]
        tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
        te = torchvision.datasets.CIFAR10(root, train=False, download=True)
        xtr, ytr = tr.data, np.array(tr.targets, dtype=np.int64)
        xte, yte = te.data, np.array(te.targets, dtype=np.int64)
    _CACHE["raw"] = (xtr, ytr, xte, yte)
    return _CACHE["raw"]


def get_split(cfg):
    """One stratified 45k/5k split, seeded with the roll-number digits, saved to disk
    and re-used by every question. If the file exists it is loaded and verified."""
    from sklearn.model_selection import train_test_split
    _, ytr, _, _ = load_cifar(cfg)
    name = f"split_seed{cfg['seed']}{'_smoke' if SMOKE else ''}.npz"
    path = ROOT / "splits" / name
    val_size = cfg["val_size"] if not SMOKE else len(ytr) // 10
    if path.exists():
        s = np.load(path)
        assert int(s["seed"]) == cfg["seed"]
        assert len(s["val_idx"]) == val_size
        return s["train_idx"], s["val_idx"]
    idx = np.arange(len(ytr))
    tr_idx, va_idx = train_test_split(idx, test_size=val_size, stratify=ytr,
                                      random_state=cfg["seed"], shuffle=True)
    tr_idx, va_idx = np.sort(tr_idx), np.sort(va_idx)
    path.parent.mkdir(exist_ok=True)
    np.savez(path, train_idx=tr_idx, val_idx=va_idx, seed=cfg["seed"],
             train_counts=np.bincount(ytr[tr_idx]), val_counts=np.bincount(ytr[va_idx]))
    return tr_idx, va_idx


def raw_splits(cfg):
    """uint8 NHWC arrays for train/val/test (for showing images)."""
    xtr, ytr, xte, yte = load_cifar(cfg)
    tr, va = get_split(cfg)
    return {"train": (xtr[tr], ytr[tr]), "val": (xtr[va], ytr[va]), "test": (xte, yte)}


# ------------------------------------------------------------------------- transforms
def _gauss_kernel(sigma):
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-t**2 / (2 * sigma**2))
    k = k / k.sum()
    return k, r


def gaussian_blur(x, sigma):
    k, r = _gauss_kernel(sigma)
    k = k.to(x.device, x.dtype)
    c = x.shape[1]
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return x


def make_transform(name, cfg):
    """Deterministic transform applied identically to train, val and test images.
    Input/Output: float tensor (N,3,32,32) in [0,1] (high-pass may leave [0,1])."""
    g = torch.Generator().manual_seed(cfg["seed"] + 1)
    if name in (None, "none", "clean"):
        return lambda x: x
    if name == "pixel_perm":
        perm = torch.randperm(32 * 32, generator=g)  # one fixed permutation of pixel positions
        return lambda x: x.reshape(x.shape[0], 3, 1024)[:, :, perm.to(x.device)].reshape(-1, 3, 32, 32)
    if name == "block_shuffle":
        b = cfg["q2"]["block_size"]
        nb = 32 // b
        perm = torch.randperm(nb * nb, generator=g)

        def f(x):
            n = x.shape[0]
            blocks = x.reshape(n, 3, nb, b, nb, b).permute(0, 2, 4, 1, 3, 5).reshape(n, nb * nb, 3, b, b)
            blocks = blocks[:, perm.to(x.device)]
            return blocks.reshape(n, nb, nb, 3, b, b).permute(0, 3, 1, 4, 2, 5).reshape(n, 3, 32, 32)
        return f
    if name == "lowpass":
        s = cfg["q2"]["lowpass_sigma"]
        return lambda x: gaussian_blur(x, s)
    if name == "highpass":
        s = cfg["q2"]["lowpass_sigma"]
        return lambda x: x - gaussian_blur(x, s) + 0.5
    if name == "grayscale":
        return lambda x: (0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:]).repeat(1, 3, 1, 1)
    raise ValueError(name)


def _to_float(x_uint8_nhwc):
    return torch.from_numpy(x_uint8_nhwc).permute(0, 3, 1, 2).float().div_(255.0)


def apply_chunked(f, x, bs=2000):
    return torch.cat([f(x[i:i + bs]) for i in range(0, len(x), bs)])


def make_data(cfg, transform="none", normalize=True):
    """Tensors on DEVICE for train/val/test after a transform. Per-channel normalisation
    statistics are computed on the (transformed) training split only."""
    key = (transform, normalize)
    if key in _CACHE:
        return _CACHE[key]
    raw = raw_splits(cfg)
    f = make_transform(transform, cfg)
    out = {}
    for s in ["train", "val", "test"]:
        x = apply_chunked(f, _to_float(raw[s][0]))
        out[s] = [x, torch.from_numpy(raw[s][1])]
    if normalize:
        mean = out["train"][0].mean(dim=(0, 2, 3), keepdim=True)
        std = out["train"][0].std(dim=(0, 2, 3), keepdim=True)
        for s in out:
            out[s][0] = (out[s][0] - mean) / std
        out["norm"] = (mean, std)
    store = torch.float16 if DEVICE.type == "cuda" else torch.float32
    for s in ["train", "val", "test"]:
        out[s] = (out[s][0].to(DEVICE, store), out[s][1].to(DEVICE))
    out["transform"] = transform
    _CACHE[key] = out
    return out


# ----------------------------------------------------------------------------- models
def count_params(m):
    return sum(p.numel() for p in m.parameters())


def count_macs(model, shape=(1, 3, 32, 32)):
    """Multiply-accumulates of Conv2d/Linear layers for one image."""
    macs = [0]

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            macs[0] += out.numel() // out.shape[0] * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
        elif isinstance(mod, nn.Linear):
            macs[0] += out.numel() // out.shape[0] * mod.in_features
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    was = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(shape, device=next(model.parameters()).device))
    model.train(was)
    for h in hs:
        h.remove()
    return macs[0]


def autocast():
    if DEVICE.type == "cuda":
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dt)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def predict(model, x, bs=1000):
    model.eval()
    outs = []
    for i in range(0, len(x), bs):
        with autocast():
            outs.append(model(x[i:i + bs].float()).float())
    return torch.cat(outs)


def acc_of(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


# ----------------------------------------------------------------------------- training
def train_classifier(model, D, epochs, lr, wd, bs, seed, y_train=None, epoch_cb=None,
                     eval_subset=5000, keep_best=True):
    """AdamW + one-cycle schedule, no augmentation. Each epoch logs eval-mode accuracy on
    a fixed train subset and on the full validation set. Returns (history, best_state,
    final_state) where best_state maximises validation accuracy (early stopping)."""
    set_seed(seed)
    model = model.to(DEVICE)
    Xtr, ytr_true = D["train"]
    ytr = ytr_true if y_train is None else y_train.to(DEVICE)
    Xva, yva = D["val"]
    n = len(Xtr)
    steps_per = math.ceil(n / bs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per, pct_start=0.15)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    sub = torch.randperm(n, generator=gen)[:eval_subset].to(DEVICE)
    hist, best, best_acc = [], None, -1
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        tl, tc = 0.0, 0
        for i in range(steps_per):
            idx = perm[i * bs:(i + 1) * bs]
            x, y = Xtr[idx].float(), ytr[idx]
            with autocast():
                out = model(x)
                loss = F.cross_entropy(out.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(idx)
            tc += (out.argmax(1) == y).sum().item()
        lt = predict(model, Xtr[sub])
        lv = predict(model, Xva)
        rec = {"epoch": ep, "time": time.time() - t0,
               "train_running_loss": tl / n, "train_running_acc": tc / n,
               "train_acc": acc_of(lt, ytr[sub]), "train_loss": F.cross_entropy(lt, ytr[sub]).item(),
               "val_acc": acc_of(lv, yva), "val_loss": F.cross_entropy(lv, yva).item()}
        if epoch_cb is not None:
            rec.update(epoch_cb(model))
        hist.append(rec)
        if keep_best and rec["val_acc"] > best_acc:
            best_acc = rec["val_acc"]
            best = copy.deepcopy(model.state_dict())
        print(f"  ep {ep:3d} train {rec['train_acc']:.3f} val {rec['val_acc']:.3f} ({rec['time']:.1f}s)", flush=True)
    return hist, best, copy.deepcopy(model.state_dict())


def run_classifier(cfg, name, builder, transform="none", lrs=None, epochs=None, wd=None):
    """Train `builder()` on data under `transform` for every lr in the grid, select the lr and
    epoch by validation accuracy, then evaluate the selected checkpoint ONCE on the test set.
    Results are cached by name, so later questions reuse earlier runs."""
    log_p = rpath(cfg, "logs", f"{name}.json")
    ck_p = rpath(cfg, "ckpt", f"{name}.pt")
    if log_p.exists() and ck_p.exists():
        print(f"[reuse] {name}")
        return json.load(open(log_p))
    c = cfg["classifier"]
    lrs = lrs or c["lr_grid"]
    epochs = epochs or c["epochs"]
    wd = c["weight_decay"] if wd is None else wd
    D = make_data(cfg, transform)
    runs = {}
    for lr in lrs:
        print(f"[train] {name} lr={lr} transform={transform}")
        m = builder()
        hist, best, _ = train_classifier(m, D, epochs, lr, wd, c["batch_size"], cfg["seed"],
                                         eval_subset=c["train_eval_subset"])
        runs[lr] = (hist, best)
    sel = max(runs, key=lambda lr: max(h["val_acc"] for h in runs[lr][0]))
    hist, best = runs[sel]
    m = builder().to(DEVICE)
    m.load_state_dict(best)
    torch.save(m.state_dict(), ck_p)
    lt = predict(m, D["test"][0])
    yte = D["test"][1]
    np.savez_compressed(rpath(cfg, "logs", f"{name}_test_logits.npz"), logits=lt.cpu().numpy(), y=yte.cpu().numpy())
    best_ep = max(hist, key=lambda h: h["val_acc"])
    info = {"name": name, "transform": transform, "params": count_params(m), "macs": count_macs(m),
            "lr_grid": lrs, "selected_lr": sel, "epochs": epochs, "weight_decay": wd,
            "best_epoch": best_ep["epoch"], "best_val_acc": best_ep["val_acc"],
            "train_acc_at_best": best_ep["train_acc"],
            "val_acc_per_lr": {str(lr): max(h["val_acc"] for h in runs[lr][0]) for lr in runs},
            "test_acc": acc_of(lt, yte), "test_loss": F.cross_entropy(lt, yte).item(),
            "history": hist, "history_per_lr": {str(lr): runs[lr][0] for lr in runs}}
    save_json(info, log_p)
    print(f"[done] {name}: val {info['best_val_acc']:.4f} test {info['test_acc']:.4f}")
    return info


def load_model(cfg, name, builder):
    m = builder().to(DEVICE)
    m.load_state_dict(torch.load(rpath(cfg, "ckpt", f"{name}.pt"), map_location=DEVICE))
    m.eval()
    return m


# ----------------------------------------------------------------------- calibration
def calib_stats(logits, y, n_bins=15):
    logits = torch.as_tensor(logits).float()
    y = torch.as_tensor(y)
    p = logits.softmax(1)
    conf, pred = p.max(1)
    corr = (pred == y).float()
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    rel = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        msk = (conf > lo) & (conf <= hi)
        if msk.any():
            ece += msk.float().mean().item() * abs(corr[msk].mean().item() - conf[msk].mean().item())
            rel.append((conf[msk].mean().item(), corr[msk].mean().item(), int(msk.sum())))
    return {"acc": corr.mean().item(), "mean_conf": conf.mean().item(),
            "conf_correct": conf[corr == 1].mean().item() if corr.sum() > 0 else float("nan"),
            "conf_wrong": conf[corr == 0].mean().item() if (corr == 0).sum() > 0 else float("nan"),
            "ece": ece, "nll": F.cross_entropy(logits, y).item(), "reliability": rel}


def confusion(pred, y, k=10):
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (np.asarray(y), np.asarray(pred)), 1)
    return cm


def fmt_pct(x):
    return f"{100 * x:.2f}"


def write_tex(cfg, name, s):
    with open(rpath(cfg, "tables", name), "w") as f:
        f.write(s)
