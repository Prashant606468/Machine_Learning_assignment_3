"""Q3. Convolutional autoencoders.
(a) Three AEs that differ only in bottleneck width d (config q3.bottlenecks). Identical
    conv encoder/decoder, optimiser, epochs, batch size, loss (pixel MSE on [0,1] images).
    Selection by validation MSE only; test MSE/PSNR reported for the selected model.
(b) Corruption: additive Gaussian noise (sigma in config), clipped to [0,1]. A DAE with the
    selected d is trained on fresh noisy inputs -> clean targets (no labels). The FIXED
    classifier is the clean CNN from Q2(a) (no re-training). Test conditions: clean, noisy,
    Gaussian-blur(noisy) baseline, AE(noisy), DAE(noisy), and AE/DAE of clean images.
(c) Frozen-encoder class information: linear probe (logistic regression; weight decay chosen
    on val) and kNN (k chosen on val) on bottleneck codes of every AE, the DAE and an
    untrained (random) encoder; compared with test reconstruction error.
"""
import copy
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (load_cfg, make_data, predict, calib_stats, save_json, rpath, write_tex, fmt_pct,
                    gaussian_blur, load_model, count_params, autocast, set_seed, DEVICE)
from models import ConvAE, CNN


def add_noise(x, sigma, gen=None):
    n = torch.randn(x.shape, generator=gen, device="cpu").to(x.device) if gen is not None else torch.randn_like(x)
    return (x + sigma * n).clamp(0, 1)


@torch.no_grad()
def recon(model, x, bs=1000):
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        with autocast():
            out.append(model(x[i:i + bs].float()).float())
    return torch.cat(out)


@torch.no_grad()
def encode(model, x, bs=1000):
    model.eval()
    return torch.cat([model.encode(x[i:i + bs].float()).float() for i in range(0, len(x), bs)])


def mse_psnr(xh, x):
    m = ((xh.float() - x.float()) ** 2).flatten(1).mean(1)
    return m.mean().item(), (10 * torch.log10(1.0 / m.clamp_min(1e-10))).mean().item()


def train_ae(cfg, name, d, sigma=None):
    q = cfg["q3"]
    log_p, ck_p = rpath(cfg, "logs", f"{name}.json"), rpath(cfg, "ckpt", f"{name}.pt")
    if log_p.exists() and ck_p.exists():
        print(f"[reuse] {name}")
        m = ConvAE(d).to(DEVICE); m.load_state_dict(torch.load(ck_p, map_location=DEVICE))
        return m, json.load(open(log_p))
    D = make_data(cfg, "none", normalize=False)
    Xtr, Xva = D["train"][0], D["val"][0]
    set_seed(cfg["seed"])
    m = ConvAE(d).to(DEVICE)
    vgen = torch.Generator().manual_seed(cfg["seed"] + 11)
    Xva_in = add_noise(Xva.float(), sigma, vgen) if sigma else Xva
    opt = torch.optim.Adam(m.parameters(), lr=q["lr"])
    steps = math.ceil(len(Xtr) / q["batch_size"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, q["epochs"] * steps)
    g = torch.Generator().manual_seed(cfg["seed"])
    hist, best, best_v = [], None, 1e9
    for ep in range(1, q["epochs"] + 1):
        m.train(); perm = torch.randperm(len(Xtr), generator=g).to(DEVICE); tl = 0
        for i in range(steps):
            x = Xtr[perm[i * q["batch_size"]:(i + 1) * q["batch_size"]]].float()
            xin = add_noise(x, sigma) if sigma else x       # fresh noise every batch
            with autocast():
                loss = F.mse_loss(m(xin).float(), x)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(x)
        vm, vp = mse_psnr(recon(m, Xva_in), Xva)
        hist.append({"epoch": ep, "train_mse": tl / len(Xtr), "val_mse": vm, "val_psnr": vp})
        if vm < best_v:
            best_v, best = vm, copy.deepcopy(m.state_dict())
        print(f"  {name} ep {ep:3d} train {tl/len(Xtr):.5f} val {vm:.5f}", flush=True)
    m.load_state_dict(best); torch.save(best, ck_p)
    info = {"name": name, "d": d, "params": count_params(m), "noise_sigma": sigma, "history": hist,
            "best_val_mse": best_v, "best_epoch": int(np.argmin([h["val_mse"] for h in hist]) + 1)}
    save_json(info, log_p)
    return m, info


def linear_probe(cfg, ftr, ytr, fva, yva, fte_list, yte):
    """Standardised features -> multinomial logistic regression, wd chosen on val."""
    q = cfg["q3"]
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    z = lambda f: (f - mu) / sd
    Ztr, Zva = z(ftr), z(fva)
    best = None
    for wd in q["probe_wd_grid"]:
        set_seed(cfg["seed"])
        lin = nn.Linear(Ztr.shape[1], 10).to(DEVICE)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(cfg["seed"])
        for ep in range(q["probe_epochs"]):
            perm = torch.randperm(len(Ztr), generator=g).to(DEVICE)
            for i in range(0, len(Ztr), 1024):
                idx = perm[i:i + 1024]
                loss = F.cross_entropy(lin(Ztr[idx]), ytr[idx]) + wd * (lin.weight ** 2).sum()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            va = (lin(Zva).argmax(1) == yva).float().mean().item()
        if best is None or va > best[0]:
            best = (va, wd, copy.deepcopy(lin))
    va, wd, lin = best
    with torch.no_grad():
        tes = [(lin(z(f)).argmax(1) == yte).float().mean().item() for f in fte_list]
    return {"val_acc": va, "wd": wd, "test_acc": tes}


def knn(cfg, ftr, ytr, fva, yva, fte, yte):
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    n = lambda f: F.normalize((f - mu) / sd, dim=1)
    Ntr = n(ftr)

    def pred(fq, k):
        out = []
        for i in range(0, len(fq), 1000):
            s = n(fq[i:i + 1000]) @ Ntr.T
            idx = s.topk(k, dim=1).indices
            votes = F.one_hot(ytr[idx], 10).sum(1)
            out.append(votes.argmax(1))
        return torch.cat(out)
    ks = [k for k in cfg["q3"]["knn_k_grid"] if k <= len(ftr)]
    vals = {k: (pred(fva, k) == yva).float().mean().item() for k in ks}
    k = max(vals, key=vals.get)
    return {"k": k, "val_acc": vals[k], "test_acc": (pred(fte, k) == yte).float().mean().item()}


def main(cfg):
    q = cfg["q3"]
    D = make_data(cfg, "none", normalize=False)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = D["train"], D["val"], D["test"]

    # ---------------- (a)
    aes, infos = {}, {}
    for d in q["bottlenecks"]:
        aes[d], infos[d] = train_ae(cfg, f"ae_{d}", d)
    d_sel = min(infos, key=lambda d: infos[d]["best_val_mse"])
    te_mse, te_psnr = mse_psnr(recon(aes[d_sel], Xte), Xte)
    rows = [f"AE-{d} & {d} & {3072/d:.0f}$\\times$ & {infos[d]['params']:,} & {infos[d]['best_epoch']} & "
            f"{infos[d]['best_val_mse']:.5f} & {infos[d]['history'][infos[d]['best_epoch']-1]['val_psnr']:.2f} & "
            + (f"{te_mse:.5f} / {te_psnr:.2f}" if d == d_sel else "--") + " \\\\" for d in q["bottlenecks"]]
    write_tex(cfg, "q3a.tex",
              "\\begin{tabular}{lrrrrrrr}\n\\toprule\nModel & $d$ & Compr. & Params & ep$^*$ & Val MSE & Val PSNR & Test MSE / PSNR \\\\\n\\midrule\n"
              + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------- (b)
    sigma = q["noise_sigma"]
    dae, dinfo = train_ae(cfg, f"dae_{d_sel}_s{int(sigma*100)}", d_sel, sigma=sigma)
    tgen = torch.Generator().manual_seed(cfg["seed"] + 13)     # one fixed noisy test set
    Xte_n = add_noise(Xte.float(), sigma, tgen)
    clf = load_model(cfg, "cnn", lambda: CNN(cfg["q2"]["cnn_widths"]))   # fixed classifier from Q2(a)
    mean, std = (t.to(DEVICE) for t in make_data(cfg, "none")["norm"])
    classify = lambda x01: predict(clf, ((x01.float() - mean) / std))
    p_clean = classify(Xte).argmax(1)
    conds = {
        "clean images": Xte.float(),
        f"noisy ($\\sigma$={sigma})": Xte_n,
        f"blur(noisy), $\\sigma_b$={q['blur_baseline_sigma']}": gaussian_blur(Xte_n, q["blur_baseline_sigma"]).clamp(0, 1),
        f"AE-{d_sel}(clean)": recon(aes[d_sel], Xte),
        f"AE-{d_sel}(noisy)": recon(aes[d_sel], Xte_n),
        f"DAE-{d_sel}(noisy)": recon(dae, Xte_n),
        f"DAE-{d_sel}(clean)": recon(dae, Xte),
    }
    resb, rows = {}, []
    for k, x in conds.items():
        lg = classify(x)
        cs = calib_stats(lg.cpu(), yte.cpu())
        mse, psnr = mse_psnr(x, Xte)
        agree = (lg.argmax(1) == p_clean).float().mean().item()
        resb[k] = {"acc": cs["acc"], "ece": cs["ece"], "mean_conf": cs["mean_conf"], "mse": mse, "psnr": psnr, "agree_clean_pred": agree}
        psnr_s = "$\\infty$" if psnr > 99 else f"{psnr:.2f}"
        rows.append(f"{k} & {fmt_pct(cs['acc'])} & {fmt_pct(agree)} & {cs['mean_conf']:.2f} & {cs['ece']:.3f} & {mse:.5f} & {psnr_s} \\\\")
    write_tex(cfg, "q3b.tex",
              "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lrrrrrr}\n\\toprule\nCNN input (test) & Acc\\% & Agree\\% & Conf & ECE & MSE & PSNR \\\\\n\\midrule\n"
              + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n")
    fig, axes = plt.subplots(5, 8, figsize=(8, 5.4))
    show = ["clean images", f"noisy ($\\sigma$={sigma})", f"AE-{d_sel}(noisy)", f"DAE-{d_sel}(noisy)", f"AE-{d_sel}(clean)"]
    for i, k in enumerate(show):
        for j in range(8):
            axes[i, j].imshow(conds[k][j].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()); axes[i, j].axis("off")
        axes[i, 0].text(-4, 16, k.replace("$\\sigma$", "s"), ha="right", va="center", fontsize=7)
    fig.tight_layout(); fig.savefig(rpath(cfg, "figs", "q3_recons.pdf"), bbox_inches="tight"); plt.close(fig)

    # ---------------- (c)
    set_seed(cfg["seed"] + 99)
    rand_enc = ConvAE(d_sel).to(DEVICE)   # untrained baseline: architecture prior only
    encs = {**{f"AE-{d}": aes[d] for d in q["bottlenecks"]}, f"DAE-{d_sel}": dae, f"random-{d_sel}": rand_enc}
    resc, rows = {}, []
    for k, m in encs.items():
        ftr, fva, fte = encode(m, Xtr), encode(m, Xva), encode(m, Xte)
        fte_n = encode(m, Xte_n)
        lp = linear_probe(cfg, ftr, ytr, fva, yva, [fte, fte_n], yte)
        kn = knn(cfg, ftr, ytr, fva, yva, fte, yte)
        mse = mse_psnr(recon(m, Xte), Xte)[0] if not k.startswith("random") else float("nan")
        resc[k] = {"probe": lp, "knn": kn, "test_mse": mse, "dim": ftr.shape[1]}
        rows.append(f"{k} & {ftr.shape[1]} & {'--' if mse != mse else f'{mse:.5f}'} & {fmt_pct(lp['test_acc'][0])} & "
                    f"{fmt_pct(lp['test_acc'][1])} & {fmt_pct(kn['test_acc'])} (k={kn['k']}) \\\\")
    write_tex(cfg, "q3c.tex",
              "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lrrrrr}\n\\toprule\nFrozen encoder & dim & Test MSE & Probe\\% (clean) & Probe\\% (noisy) & kNN\\% \\\\\n\\midrule\n"
              + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n")
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for k, r in resc.items():
        if r["test_mse"] == r["test_mse"]:
            ax.scatter(r["test_mse"], 100 * r["probe"]["test_acc"][0]); ax.annotate(k, (r["test_mse"], 100 * r["probe"]["test_acc"][0]), fontsize=7)
    rr = resc[f"random-{d_sel}"]["probe"]["test_acc"][0]
    ax.axhline(100 * rr, ls=":", c="grey"); ax.text(ax.get_xlim()[0], 100 * rr, " random encoder", fontsize=7, va="bottom")
    ax.set_xlabel("test reconstruction MSE (lower = better)"); ax.set_ylabel("linear-probe test acc (%)"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(rpath(cfg, "figs", "q3c_mse_vs_probe.pdf")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for d in q["bottlenecks"]:
        ax.plot([h["val_mse"] for h in infos[d]["history"]], label=f"AE-{d}")
    ax.plot([h["val_mse"] for h in dinfo["history"]], "--", label=f"DAE-{d_sel} (noisy val in)")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("val MSE"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(rpath(cfg, "figs", "q3a_curves.pdf")); plt.close(fig)

    save_json({"selected_d": d_sel, "test_mse_selected": te_mse, "test_psnr_selected": te_psnr,
               "val_mse": {d: infos[d]["best_val_mse"] for d in infos}, "params": {d: infos[d]["params"] for d in infos},
               "dae": {"name": dinfo["name"], "sigma": sigma, "best_val_mse": dinfo["best_val_mse"]},
               "q3b": resb, "q3c": resc}, rpath(cfg, "logs", "q3_summary.json"))


if __name__ == "__main__":
    cfg = load_cfg()
    set_seed(cfg["seed"])
    main(cfg)
