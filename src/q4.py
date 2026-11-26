"""Q4 (bonus). SVMs and clustering on ONE fixed representation.
Representation (fixed before any SVM is fit): bottleneck code of the clean AE selected in
Q3(a) -> standardise (train statistics) -> PCA to q4.pca_dim (fit on the 45k train split).
SVMs are fit on a fixed stratified train subset (q4.subset_size, chosen before selection and
used for every SVM); validation = the full 5k val split; test = official test set, once.
gamma for poly/RBF is fixed numerically (sklearn 'scale' formula on the subset) so that the
binary pair SVMs in (b) are exactly the one-vs-one sub-problems of the multi-class model.
"""
import json
import time
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

from common import (load_cfg, make_data, save_json, rpath, write_tex, fmt_pct, raw_splits, confusion,
                    CLASSES, DEVICE, ROOT, SMOKE, set_seed)
from models import ConvAE
from q3 import encode


def representation(cfg):
    p = rpath(cfg, "logs", "q4_representation.npz")
    if p.exists():
        z = np.load(p); return {k: z[k] for k in z.files}
    s3 = json.load(open(rpath(cfg, "logs", "q3_summary.json")))
    d = s3["selected_d"]
    ae = ConvAE(d).to(DEVICE)
    ae.load_state_dict(torch.load(rpath(cfg, "ckpt", f"ae_{d}.pt"), map_location=DEVICE))
    D = make_data(cfg, "none", normalize=False)
    F = {s: encode(ae, D[s][0]).cpu().numpy() for s in ["train", "val", "test"]}
    mu, sd = F["train"].mean(0), F["train"].std(0) + 1e-6
    pca = PCA(cfg["q4"]["pca_dim"], random_state=cfg["seed"]).fit((F["train"] - mu) / sd)
    out = {f"X{s}": pca.transform((F[s] - mu) / sd).astype(np.float32) for s in F}
    out.update({f"y{s}": D[s][1].cpu().numpy() for s in ["train", "val", "test"]})
    out["explained_var"] = pca.explained_variance_ratio_.sum()
    out["ae_d"] = d
    np.savez(p, **out)
    return out


def subset_indices(cfg, ytrain):
    p = ROOT / "splits" / f"q4_subset_{cfg['q4']['subset_size']}_seed{cfg['seed']}{'_smoke' if SMOKE else ''}.npz"
    if p.exists():
        return np.load(p)["idx"]
    idx, _ = train_test_split(np.arange(len(ytrain)), train_size=cfg["q4"]["subset_size"],
                              stratify=ytrain, random_state=cfg["seed"])
    idx = np.sort(idx); np.savez(p, idx=idx); return idx


DAGGER = "$^\\dagger$"


def make_svc(kernel, C, gamma, cfg):
    return SVC(kernel=kernel, C=C, gamma=gamma, degree=cfg["q4"]["poly_degree"], coef0=1.0,
               max_iter=cfg["q4"]["max_iter"], cache_size=1000)


def fit_one(kernel, C, gamma, cfg, Xs, ys, Xv, yv):
    t = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        m = make_svc(kernel, C, gamma, cfg).fit(Xs, ys)
        conv = not any(issubclass(x.category, ConvergenceWarning) for x in w)
    return {"kernel": kernel, "C": C, "train_acc": float((m.predict(Xs) == ys).mean()),
            "val_acc": float((m.predict(Xv) == yv).mean()), "sv_frac": len(m.support_) / len(ys),
            "n_sv": int(len(m.support_)), "fit_time": time.time() - t, "converged": conv}


def show(ax, img, title):
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=5.5)


def main(cfg):
    q = cfg["q4"]
    R = representation(cfg)
    sub = subset_indices(cfg, R["ytrain"])
    Xs, ys = R["Xtrain"][sub], R["ytrain"][sub]
    Xv, yv, Xt, yt = R["Xval"], R["yval"], R["Xtest"], R["ytest"]
    gamma = 1.0 / (Xs.shape[1] * Xs.var())

    # ---------------- (a) sweep
    sw_p = rpath(cfg, "logs", "q4_sweep.json")
    if sw_p.exists():
        sweep = json.load(open(sw_p))
    else:
        jobs = [(k, C) for k in q["kernels"] for C in q["C_grid"]]
        sweep = Parallel(n_jobs=q["n_jobs"], verbose=5)(delayed(fit_one)(k, C, gamma, cfg, Xs, ys, Xv, yv) for k, C in jobs)
        save_json(sweep, sw_p)
    best = max(sweep, key=lambda r: (r["val_acc"], -r["sv_frac"]))
    svm = make_svc(best["kernel"], best["C"], gamma, cfg).fit(Xs, ys)
    pt = svm.predict(Xt)
    test_acc = float((pt == yt).mean())
    cm = confusion(pt, yt)
    rows = [f"{r['kernel']} & {r['C']:g} & {fmt_pct(r['train_acc'])} & {fmt_pct(r['val_acc'])} & {r['sv_frac']:.3f} & "
            f"{r['fit_time']:.0f}{'' if r['converged'] else DAGGER}" + (" \\textbf{(sel.)}" if r is best else "") + " \\\\"
            for r in sweep]
    write_tex(cfg, "q4_sweep.tex",
              "\\begin{tabular}{lrrrrr}\n\\toprule\nKernel & $C$ & Train\\% & Val\\% & SV frac & fit (s) \\\\\n\\midrule\n"
              + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    fig, axes = plt.subplots(1, len(q["kernels"]), figsize=(3.4 * len(q["kernels"]), 2.8), sharey=True)
    for a, k in zip(np.atleast_1d(axes), q["kernels"]):
        rs = [r for r in sweep if r["kernel"] == k]
        Cs = [r["C"] for r in rs]
        a.semilogx(Cs, [100 * r["val_acc"] for r in rs], "o-", label="val acc")
        a.semilogx(Cs, [100 * r["train_acc"] for r in rs], "s--", alpha=.6, label="train acc")
        a2 = a.twinx(); a2.semilogx(Cs, [r["sv_frac"] for r in rs], "^-", color="C3", label="SV fraction")
        a2.set_ylim(0, 1.05); a2.set_ylabel("SV fraction", color="C3", fontsize=8)
        a.set_title(k); a.set_xlabel("C"); a.grid(alpha=.3)
    np.atleast_1d(axes)[0].set_ylabel("accuracy (%)"); np.atleast_1d(axes)[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(rpath(cfg, "figs", "q4a_sweep.pdf")); plt.close(fig)

    # ---------------- (b) difficult pairs
    raw = raw_splits(cfg)
    img_tr, img_te = raw["train"][0][sub], raw["test"][0]
    off = cm + cm.T; np.fill_diagonal(off, 0)
    iu = np.triu_indices(10, 1)
    order = np.argsort(-off[iu])[:q["n_pairs"]]
    pairs = [(int(iu[0][o]), int(iu[1][o])) for o in order]
    mus = np.stack([Xs[ys == c].mean(0) for c in range(10)])
    tr_var = np.array([Xs[ys == c].var(0).sum() for c in range(10)])
    fisher = lambda i, j: float(((mus[i] - mus[j]) ** 2).sum() / (tr_var[i] + tr_var[j]))
    fisher_all = float(np.mean([fisher(i, j) for i, j in zip(*iu)]))
    pinfo, rows = [], []
    fig, axes = plt.subplots(2 * len(pairs), 8, figsize=(8, 2.25 * len(pairs)))
    fh, ah = plt.subplots(1, len(pairs), figsize=(3.3 * len(pairs), 2.4))
    for pi, (i, j) in enumerate(pairs):
        mtr, mte = np.isin(ys, [i, j]), np.isin(yt, [i, j])
        yb = np.where(ys[mtr] == i, 1, -1)
        b = make_svc(best["kernel"], best["C"], gamma, cfg).fit(Xs[mtr], yb)
        ftr = b.decision_function(Xs[mtr]); fte = b.decision_function(Xt[mte])
        ybt = np.where(yt[mte] == i, 1, -1)
        alpha = np.abs(b.dual_coef_[0])
        bounded = alpha >= best["C"] * (1 - 1e-6)
        sv_local = b.support_
        marg = ybt * fte
        mc_conf = ((yt[mte] == i) & (pt[mte] == j)) | ((yt[mte] == j) & (pt[mte] == i))   # multi-class i<->j errors
        info = {"pair": [CLASSES[i], CLASSES[j]], "test_confusions_multiclass": int(off[i, j]),
                "n_sv": int(len(sv_local)), "sv_frac": len(sv_local) / mtr.sum(), "frac_bounded": float(bounded.mean()),
                "binary_test_err": float((marg < 0).mean()), "test_frac_in_margin": float((np.abs(fte) < 1).mean()),
                "mc_confusions_in_margin": float((np.abs(fte[mc_conf]) < 1).mean()) if mc_conf.any() else None,
                "mc_confusions_wrong_side": float((marg[mc_conf] < 0).mean()) if mc_conf.any() else None,
                "fisher_ratio": fisher(i, j), "fisher_ratio_mean_all_pairs": fisher_all}
        pinfo.append(info)
        rows.append(f"{CLASSES[i]}--{CLASSES[j]} & {off[i,j]} & {info['sv_frac']:.2f} & {info['frac_bounded']:.2f} & "
                    f"{fmt_pct(info['binary_test_err'])} & {fmt_pct(info['test_frac_in_margin'])} & "
                    f"{'--' if info['mc_confusions_in_margin'] is None else fmt_pct(info['mc_confusions_in_margin'])} & "
                    f"{info['fisher_ratio']:.3f} \\\\")
        # most violating SVs (most negative y f), 4 per class
        ysv = yb[sv_local]; vio = ysv * ftr[sv_local]
        picks = []
        for cls in [1, -1]:
            cand = np.where(ysv == cls)[0]
            picks += list(cand[np.argsort(vio[cand])[:4]])
        gidx_tr = np.where(mtr)[0]
        for c_, k in enumerate(picks):
            g = gidx_tr[sv_local[k]]
            show(axes[2 * pi, c_], img_tr[g], f"SV {CLASSES[ys[g]][:5]} f={ftr[sv_local[k]]:+.2f}")
        near = np.argsort(np.abs(fte))[:8]
        gidx_te = np.where(mte)[0]
        for c_, k in enumerate(near):
            g = gidx_te[k]
            show(axes[2 * pi + 1, c_], img_te[g], f"T {CLASSES[yt[g]][:5]}->{CLASSES[pt[g]][:5]} f={fte[k]:+.2f}")
        axes[2 * pi, 0].set_ylabel(f"{CLASSES[i][:5]}/{CLASSES[j][:5]}\nSVs", fontsize=7)
        axes[2 * pi + 1, 0].set_ylabel("near-bdry\ntest", fontsize=7)
        a = np.atleast_1d(ah)[pi]
        a.hist(fte[ybt == 1], 40, alpha=.6, label=CLASSES[i]); a.hist(fte[ybt == -1], 40, alpha=.6, label=CLASSES[j])
        for v in [-1, 0, 1]:
            a.axvline(v, color="k", ls="-" if v == 0 else ":", lw=1)
        a.set_title(f"test decision f: {CLASSES[i]} (+) vs {CLASSES[j]} (-)", fontsize=8); a.legend(fontsize=7)
    for a in axes.flat:
        if not a.images:
            a.axis("off")
    fig.tight_layout(); fig.savefig(rpath(cfg, "figs", "q4b_pairs.pdf"), dpi=200); plt.close(fig)
    fh.tight_layout(); fh.savefig(rpath(cfg, "figs", "q4b_margins.pdf")); plt.close(fh)
    write_tex(cfg, "q4_pairs.tex",
              "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lrrrrrrr}\n\\toprule\nPair & MC conf. & SV frac & bounded & bin. err\\% & in margin\\% & MC err in margin\\% & Fisher \\\\\n\\midrule\n"
              + "\n".join(rows) + f"\n\\midrule\nmean over all 45 pairs & & & & & & & {fisher_all:.3f} \\\\\n\\bottomrule\n\\end{{tabular}}}}\n")

    # ---------------- (c) clustering
    Xtr_full = R["Xtrain"]
    rng = np.random.default_rng(cfg["seed"])
    vs = rng.choice(len(Xv), min(q["silhouette_sample"], len(Xv)), replace=False)
    cands, rows = [], []
    for meth in q["cluster_methods"]:
        for k in q["cluster_k"]:
            if meth == "kmeans":
                m = KMeans(k, n_init=10, random_state=cfg["seed"]).fit(Xtr_full)
            else:
                m = GaussianMixture(k, covariance_type="diag", random_state=cfg["seed"], max_iter=300).fit(Xtr_full)
            lv = m.predict(Xv)
            sil = float(silhouette_score(Xv[vs], lv[vs])) if len(np.unique(lv[vs])) > 1 else -1
            extra = -m.score(Xv) * len(Xv) if meth == "kmeans" else m.bic(Xv)
            cands.append({"method": meth, "k": k, "val_silhouette": sil, "val_inertia_or_bic": float(extra), "model": m})
            rows.append(f"{meth} & {k} & {sil:.4f} & {extra:.4g} \\\\")
    sel = max(cands, key=lambda c: c["val_silhouette"])
    lt = sel["model"].predict(Xt)
    ari, nmi = adjusted_rand_score(yt, lt), normalized_mutual_info_score(yt, lt)
    K = sel["k"]
    ct = np.zeros((K, 10), dtype=int); np.add.at(ct, (lt, yt), 1)
    write_tex(cfg, "q4_cluster_sel.tex",
              "\\begin{tabular}{lrrr}\n\\toprule\nMethod & $k$ & Val silhouette & Val inertia / BIC \\\\\n\\midrule\n"
              + "\n".join(rows) + f"\n\\midrule\n\\multicolumn{{4}}{{l}}{{Selected: {sel['method']}, $k$={K}. Test ARI {ari:.4f}, test NMI {nmi:.4f}}} \\\\\n\\bottomrule\n\\end{{tabular}}\n")
    write_tex(cfg, "q4_contingency.tex",
              "\\resizebox{\\linewidth}{!}{\\begin{tabular}{l" + "r" * 10 + "r}\n\\toprule\nCluster & " + " & ".join(c[:4] for c in CLASSES)
              + " & size \\\\\n\\midrule\n" + "\n".join(f"{c} & " + " & ".join(str(v) for v in ct[c]) + f" & {ct[c].sum()} \\\\" for c in range(K))
              + "\n\\bottomrule\n\\end{tabular}}\n")
    fig, ax = plt.subplots(figsize=(5, 0.3 * K + 1.2))
    ax.imshow(ct / ct.sum(0, keepdims=True), cmap="viridis", aspect="auto")
    ax.set_xticks(range(10)); ax.set_xticklabels([c[:4] for c in CLASSES], rotation=90, fontsize=7)
    ax.set_yticks(range(K)); ax.set_yticklabels(range(K), fontsize=6); ax.set_ylabel("cluster")
    ax.set_title(f"P(cluster | class), test. ARI {ari:.3f}, NMI {nmi:.3f}", fontsize=8)
    fig.tight_layout(); fig.savefig(rpath(cfg, "figs", "q4c_contingency.pdf")); plt.close(fig)

    # classification vs clustering disagreement
    P = ct / ct.sum(0, keepdims=True)                       # P(cluster | class)
    recall = np.diag(cm) / cm.sum(1)
    purity_class = P.max(0)                                  # share of class in its main cluster
    ent = -(P * np.log(np.clip(P, 1e-12, 1))).sum(0) / np.log(K)
    pair_rows = []
    for i, j in zip(*iu):
        overlap = float(np.minimum(P[:, i], P[:, j]).sum())  # 1 = identical cluster profiles
        svm_conf = float((cm[i, j] + cm[j, i]) / (cm[i].sum() + cm[j].sum()))
        pair_rows.append({"pair": [CLASSES[i], CLASSES[j]], "cluster_overlap": overlap, "svm_pair_confusion": svm_conf})
    ov = np.array([r["cluster_overlap"] for r in pair_rows]); sc = np.array([r["svm_pair_confusion"] for r in pair_rows])
    rk = lambda a: a.argsort().argsort() / (len(a) - 1)
    dis = rk(ov) - rk(sc)                                    # >0: clustering merges what SVM separates
    top = list(np.argsort(-dis)[:2]) + list(np.argsort(dis)[:1])
    rows = [f"{pair_rows[t]['pair'][0]}--{pair_rows[t]['pair'][1]} & {pair_rows[t]['cluster_overlap']:.3f} & "
            f"{fmt_pct(pair_rows[t]['svm_pair_confusion'])} & {dis[t]:+.2f} \\\\" for t in top]
    cls_rows = [f"{CLASSES[c]} & {fmt_pct(recall[c])} & {fmt_pct(purity_class[c])} & {ent[c]:.2f} \\\\" for c in range(10)]
    write_tex(cfg, "q4_disagree.tex",
              "\\begin{tabular}{lrrr}\n\\toprule\nPair & cluster overlap & SVM pair conf.\\% & rank gap \\\\\n\\midrule\n" + "\n".join(rows)
              + "\n\\bottomrule\n\\end{tabular}\n")
    write_tex(cfg, "q4_class.tex",
              "\\begin{tabular}{lrrr}\n\\toprule\nClass & SVM recall\\% & main-cluster share\\% & norm. entropy \\\\\n\\midrule\n" + "\n".join(cls_rows)
              + "\n\\bottomrule\n\\end{tabular}\n")
    save_json({"representation": {"ae_d": int(R["ae_d"]), "pca_dim": q["pca_dim"], "pca_explained_var": float(R["explained_var"]),
                                  "subset_size": len(sub), "gamma": gamma, "poly_coef0": 1.0},
               "selected_svm": best, "svm_test_acc": test_acc, "svm_test_confusion": cm, "pairs": pinfo,
               "cluster_candidates": [{k: v for k, v in c.items() if k != "model"} for c in cands],
               "cluster_selected": {"method": sel["method"], "k": K}, "test_ARI": ari, "test_NMI": nmi,
               "contingency": ct, "pair_disagreement": [pair_rows[t] | {"rank_gap": float(dis[t])} for t in top],
               "class_stats": {CLASSES[c]: {"svm_recall": recall[c], "main_cluster_share": purity_class[c], "entropy": ent[c]} for c in range(10)}},
              rpath(cfg, "logs", "q4_summary.json"))
    print(f"SVM {best['kernel']} C={best['C']} val {best['val_acc']:.4f} TEST {test_acc:.4f} | "
          f"cluster {sel['method']} k={K} ARI {ari:.4f} NMI {nmi:.4f}")


if __name__ == "__main__":
    cfg = load_cfg()
    set_seed(cfg["seed"])
    main(cfg)
