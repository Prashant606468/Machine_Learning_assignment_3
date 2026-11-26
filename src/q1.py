"""Q1. Why does an MLP do poorly on CIFAR10?
(a) Hypothesis H1 (chosen): no spatial inductive bias (locality + weight sharing).
    Rival H2: insufficient capacity / underfitting.
    - MLP-base vs MLP-wide (4x params): capacity probe. H2 predicts a large val gain and
      MLP-base train acc far from 100%.
    - MLP-base vs LocalMLP (same params +0.13%): adds only a shared 4x4 patch layer.
      H1 predicts LocalMLP > MLP-base.
    - Both again on pixel-permuted images (one fixed permutation, same for train/val/test):
      H1 predicts LocalMLP's advantage disappears there, while MLP-base is unchanged.
(b) Memorisation: MLP-wide, no weight decay, long training, with a fraction of training
    labels replaced by random wrong labels (100%: labels drawn uniformly, independent of
    the image). Track per epoch the fit to clean vs noisy labels.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (load_cfg, make_data, run_classifier, train_classifier, predict, acc_of,
                    save_json, rpath, write_tex, fmt_pct, count_params, count_macs, DEVICE, set_seed)
from models import MLP, LocalMLP


def builders(cfg):
    q = cfg["q1"]
    return {
        "mlp_base": lambda: MLP(q["mlp_base_hidden"]),
        "mlp_wide": lambda: MLP(q["mlp_wide_hidden"]),
        "local_mlp": lambda: LocalMLP(q["local_patch"], q["local_token_dim"], q["local_hidden"]),
    }


PRETTY = {"mlp_base": "MLP-base", "mlp_wide": "MLP-wide", "local_mlp": "LocalMLP"}


def part_a(cfg):
    B = builders(cfg)
    res = {}
    for k in ["mlp_base", "mlp_wide", "local_mlp"]:
        res[k] = run_classifier(cfg, k, B[k], "none")
    # permutation control: same lr as selected on clean data (no extra tuning)
    for k in ["mlp_base", "local_mlp"]:
        res[k + "__pixel_perm"] = run_classifier(cfg, k + "__pixel_perm", B[k], "pixel_perm",
                                                 lrs=[res[k]["selected_lr"]])
    summary = {k: {kk: v[kk] for kk in ["params", "macs", "selected_lr", "best_epoch", "best_val_acc",
                                        "train_acc_at_best", "test_acc", "val_acc_per_lr"]} for k, v in res.items()}
    summary["final_train_acc"] = {k: v["history"][-1]["train_acc"] for k, v in res.items()}
    save_json(summary, rpath(cfg, "logs", "q1a_summary.json"))

    # table
    rows = []
    for k in ["mlp_base", "mlp_wide", "local_mlp"]:
        for cond, suffix in [("clean", ""), ("pixel-perm", "__pixel_perm")]:
            if k + suffix not in res:
                continue
            r = res[k + suffix]
            rows.append(f"{PRETTY[k]} & {cond} & {r['params']/1e6:.3f} & {r['macs']/1e6:.2f} & "
                        f"{r['selected_lr']:.0e} & {fmt_pct(r['history'][-1]['train_acc'])} & "
                        f"{fmt_pct(r['best_val_acc'])} & {fmt_pct(r['test_acc'])} \\\\")
    write_tex(cfg, "q1a.tex",
              "\\begin{tabular}{llrrrrrr}\n\\toprule\nModel & Data & Params (M) & MMACs & lr & Train\\% (final) & Val\\% & Test\\% \\\\\n\\midrule\n"
              + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    cols = {"mlp_base": "C0", "mlp_wide": "C1", "local_mlp": "C2"}
    for k, r in res.items():
        base = k.split("__")[0]
        ls = "-" if "__" not in k else ":"
        ep = [h["epoch"] for h in r["history"]]
        lab = PRETTY[base] + (" (perm)" if "__" in k else "")
        axes[0].plot(ep, [h["train_acc"] for h in r["history"]], ls, color=cols[base], label=lab)
        axes[1].plot(ep, [h["val_acc"] for h in r["history"]], ls, color=cols[base], label=lab)
    axes[0].set_title("Train accuracy (eval mode, fixed 5k subset)")
    axes[1].set_title("Validation accuracy")
    for a in axes:
        a.set_xlabel("epoch"); a.grid(alpha=.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(rpath(cfg, "figs", "q1a_curves.pdf"))
    plt.close(fig)
    return res


def part_b(cfg):
    q = cfg["q1"]["memorisation"]
    D = make_data(cfg, "none")
    ytrue = D["train"][1]
    n = len(ytrue)
    g = torch.Generator().manual_seed(cfg["seed"] + 7)
    out = {}
    for rate in q["noise_rates"]:
        tag = f"mem_noise{int(rate*100)}"
        p = rpath(cfg, "logs", f"q1b_{tag}.json")
        if p.exists():
            import json
            out[rate] = json.load(open(p)); print(f"[reuse] {tag}"); continue
        # corrupt a fixed random subset: new label drawn uniformly from the 9 wrong classes
        noisy = torch.zeros(n, dtype=torch.bool)
        noisy[torch.randperm(n, generator=g)[:int(round(rate * n))]] = True
        shift = torch.randint(1, 10, (n,), generator=g)
        ynoisy = ytrue.cpu().clone()
        ynoisy[noisy] = (ynoisy[noisy] + shift[noisy]) % 10
        if rate >= 1.0:
            # fully random labels, independent of the image (Zhang et al. style control);
            # "always wrong" labels would leak information (model learns "not class y")
            ynoisy = torch.randint(0, 10, (n,), generator=g)
        noisy, ynoisy = noisy.to(DEVICE), ynoisy.to(DEVICE)
        sub = torch.randperm(n, generator=g)[:q["eval_subset"]].to(DEVICE)
        cl_idx = sub[~noisy[sub]]
        nz_idx = sub[noisy[sub]]
        Xtr = D["train"][0]

        def cb(model):
            r = {}
            if len(cl_idx):
                r["clean_subset_acc"] = acc_of(predict(model, Xtr[cl_idx]), ytrue[cl_idx])
            if len(nz_idx):
                lz = predict(model, Xtr[nz_idx])
                r["noisy_fit_noisy_label"] = acc_of(lz, ynoisy[nz_idx])   # memorisation
                r["noisy_pred_true_label"] = acc_of(lz, ytrue[nz_idx])    # learned structure
            return r
        print(f"[train] memorisation noise={rate}")
        m = MLP(q["hidden"])
        hist, best, final = train_classifier(m, D, q["epochs"], q["lr"], q["weight_decay"],
                                             cfg["classifier"]["batch_size"], cfg["seed"],
                                             y_train=ynoisy, epoch_cb=cb, eval_subset=q["eval_subset"])
        # early-stopped (val-selected) vs final checkpoint, both on test
        res = {"rate": rate, "history": hist, "params": count_params(m)}
        for nm, st in [("best_val", best), ("final", final)]:
            m.load_state_dict(st)
            res[f"test_acc_{nm}"] = acc_of(predict(m, D["test"][0]), D["test"][1])
        bv = max(hist, key=lambda h: h["val_acc"])
        res["best_val_epoch"], res["best_val_acc"] = bv["epoch"], bv["val_acc"]
        res["final_val_acc"] = hist[-1]["val_acc"]
        if "noisy_fit_noisy_label" in hist[0]:
            mem = [h["noisy_fit_noisy_label"] for h in hist]
            res["mem_at_best_val"] = bv.get("noisy_fit_noisy_label")
            for th in [0.2, 0.5, 0.9]:
                e = next((h["epoch"] for h in hist if h["noisy_fit_noisy_label"] >= th), None)
                res[f"epoch_mem_{int(th*100)}"] = e
            res["final_mem"] = mem[-1]
        save_json(res, p)
        out[rate] = res

    # figure
    rates = list(out)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    mid = [r for r in rates if 0 < r < 1]
    if mid:
        r = out[mid[0]]
        h = r["history"]; ep = [x["epoch"] for x in h]
        a = axes[0]
        a.plot(ep, [x["val_acc"] for x in h], "k-", label="val acc (true labels)")
        a.plot(ep, [x["clean_subset_acc"] for x in h], "C0-", label="train, clean-label pts")
        a.plot(ep, [x["noisy_fit_noisy_label"] for x in h], "C3-", label="train, noisy pts: fit to WRONG label")
        a.plot(ep, [x["noisy_pred_true_label"] for x in h], "C2--", label="train, noisy pts: predicts TRUE label")
        a.axvline(r["best_val_epoch"], color="grey", ls=":", lw=1)
        a.set_title(f"{int(mid[0]*100)}% label noise, MLP-wide ({r['params']/1e6:.1f}M)")
        a.legend(fontsize=7); a.set_xlabel("epoch"); a.grid(alpha=.3)
    a = axes[1]
    for r in rates:
        h = out[r]["history"]
        a.plot([x["epoch"] for x in h], [x["val_acc"] for x in h], label=f"val acc, noise {int(r*100)}%")
    if 1.0 in out:
        h = out[1.0]["history"]
        a.plot([x["epoch"] for x in h], [x["train_acc"] for x in h], "k:", label="train acc vs random labels (100%)")
    a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend(fontsize=7); a.set_title("Validation accuracy across noise rates")
    fig.tight_layout(); fig.savefig(rpath(cfg, "figs", "q1b_memorisation.pdf")); plt.close(fig)

    rows = []
    for r in rates:
        o = out[r]
        g_ = lambda k: "--" if o.get(k) is None else (fmt_pct(o[k]) if isinstance(o[k], float) else str(o[k]))
        rows.append(f"{int(r*100)}\\% & {o['best_val_epoch']} & {fmt_pct(o['best_val_acc'])} & {g_('mem_at_best_val')} & "
                    f"{g_('epoch_mem_50')} & {g_('final_mem')} & {fmt_pct(o['final_val_acc'])} & "
                    f"{fmt_pct(o['test_acc_best_val'])} & {fmt_pct(o['test_acc_final'])} \\\\")
    write_tex(cfg, "q1b.tex",
              "\\begin{tabular}{lrrrrrrrr}\n\\toprule\nNoise & ep$^*$ & Val$^*$ & Mem@ep$^*$ & ep(Mem$\\geq$50\\%) & Mem final & Val final & Test (ep$^*$) & Test final \\\\\n\\midrule\n"
              + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    return out


if __name__ == "__main__":
    cfg = load_cfg()
    set_seed(cfg["seed"])
    part_a(cfg)
    part_b(cfg)
