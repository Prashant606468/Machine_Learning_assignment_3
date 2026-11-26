"""Q2. CNN vs MLP.
(a) Fairness criterion fixed BEFORE results: parameter count within +-5%, identical data,
    no augmentation, same optimiser/schedule/epochs/batch size, same lr grid and selection
    rule (validation accuracy). Compute (MACs) is NOT matched and is reported.
    Proposed reason: the CNN's locality + weight sharing exploit local pixel correlations.
(b) Spatial disturbance with all pixel values preserved: block_shuffle (16 blocks of 8x8,
    one fixed permutation: local structure kept, global layout destroyed) and pixel_perm
    (one fixed permutation of 1024 positions: all spatial structure destroyed). Same
    transform for train/val/test. Prediction: MLP ~unchanged under both; CNN drops a bit
    under block_shuffle and collapses to ~MLP level (or below) under pixel_perm.
(c) Two explanations of what the CNN uses: E1 low-frequency global shape / colour layout vs
    E2 high-frequency local texture/edges. CNN and MLP are retrained+tested on low-pass and
    high-pass versions; the clean CNN is additionally evaluated on shifted test images.
    Errors (per-class, confusions) and confidence (ECE, confidence on right/wrong) analysed.
"""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (load_cfg, make_data, run_classifier, predict, calib_stats, confusion, save_json,
                    rpath, write_tex, fmt_pct, raw_splits, make_transform, _to_float, apply_chunked,
                    load_model, DEVICE, CLASSES, set_seed)
from models import MLP, CNN

NICE = {"none": "clean", "block_shuffle": "block-shuffle 8$\\times$8", "pixel_perm": "pixel-perm",
        "lowpass": "low-pass", "highpass": "high-pass"}


def _f(v):
    return "--" if v != v else f"{v:.2f}"


def builders(cfg):
    return {"mlp_base": lambda: MLP(cfg["q1"]["mlp_base_hidden"]),
            "cnn": lambda: CNN(cfg["q2"]["cnn_widths"])}


def run_name(model, cond):
    return model if cond == "none" else f"{model}__{cond}"


def shifted_test(cfg, cond):
    """Test images under `cond`, normalised with CLEAN training statistics (for evaluating a
    clean-trained model under shift)."""
    Dc = make_data(cfg, "none")
    mean, std = Dc["norm"]
    x = apply_chunked(make_transform(cond, cfg), _to_float(raw_splits(cfg)["test"][0]))
    return ((x - mean) / std).to(DEVICE), Dc["test"][1]


def main(cfg):
    B = builders(cfg)
    conds = ["none"] + cfg["q2"]["conditions"]
    res = {}
    for m in ["mlp_base", "cnn"]:
        res[(m, "none")] = run_classifier(cfg, m, B[m], "none")
        lr = res[(m, "none")]["selected_lr"]
        for c in conds[1:]:
            res[(m, c)] = run_classifier(cfg, run_name(m, c), B[m], c, lrs=[lr])
    p_m, p_c = res[("mlp_base", "none")]["params"], res[("cnn", "none")]["params"]
    assert abs(p_c - p_m) / p_m <= 0.05, f"fairness criterion violated: {p_c} vs {p_m}"

    # --- calibration / error analysis from the saved test logits
    stats = {}
    for (m, c), r in res.items():
        z = np.load(rpath(cfg, "logs", f"{run_name(m, c)}_test_logits.npz"))
        s = calib_stats(z["logits"], z["y"])
        cm = confusion(z["logits"].argmax(1), z["y"])
        s["per_class_acc"] = (np.diag(cm) / cm.sum(1)).tolist()
        s["confusion"] = cm.tolist()
        off = cm + cm.T; np.fill_diagonal(off, 0)
        i, j = np.unravel_index(off.argmax(), off.shape)
        s["top_confused_pair"] = [CLASSES[i], CLASSES[j], int(off[i, j])]
        stats[f"{m}|{c}"] = s
    # clean-trained models evaluated on shifted test data (extra, clearly labelled)
    cross = {}
    for m in ["mlp_base", "cnn"]:
        model = load_model(cfg, m, B[m])
        for c in conds[1:]:
            x, y = shifted_test(cfg, c)
            lg = predict(model, x).cpu()
            cross[f"{m}|{c}"] = calib_stats(lg, y.cpu())
    save_json({"runs": {f"{m}|{c}": {k: r[k] for k in ["params", "macs", "selected_lr", "best_val_acc", "test_acc"]}
                        for (m, c), r in res.items()},
               "calibration": stats, "cross_shift_clean_models": cross}, rpath(cfg, "logs", "q2_summary.json"))

    # --- tables
    rm, rc = res[("mlp_base", "none")], res[("cnn", "none")]
    write_tex(cfg, "q2_params.tex",
              "\\begin{tabular}{lrrrrr}\n\\toprule\nModel & Params & MMACs & lr$^*$ & Val\\% & Test\\% \\\\\n\\midrule\n"
              f"MLP-base (Q1a) & {rm['params']:,} & {rm['macs']/1e6:.1f} & {rm['selected_lr']:.0e} & {fmt_pct(rm['best_val_acc'])} & {fmt_pct(rm['test_acc'])} \\\\\n"
              f"CNN & {rc['params']:,} & {rc['macs']/1e6:.1f} & {rc['selected_lr']:.0e} & {fmt_pct(rc['best_val_acc'])} & {fmt_pct(rc['test_acc'])} \\\\\n"
              "\\bottomrule\n\\end{tabular}\n")
    rows = []
    for c in conds:
        a, b = stats[f"mlp_base|{c}"], stats[f"cnn|{c}"]
        rows.append(f"{NICE[c]} & {fmt_pct(a['acc'])} & {a['ece']:.3f} & {fmt_pct(b['acc'])} & {b['ece']:.3f} & "
                    f"{_f(b['conf_correct'])}/{_f(b['conf_wrong'])} & {100*(b['acc']-a['acc']):+.2f} \\\\")
    write_tex(cfg, "q2_main.tex",
              "\\begin{tabular}{lrrrrrr}\n\\toprule\n & \\multicolumn{2}{c}{MLP-base} & \\multicolumn{3}{c}{CNN} & \\\\\n"
              "Train=test condition & Test\\% & ECE & Test\\% & ECE & conf R/W & CNN$-$MLP \\\\\n\\midrule\n"
              + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    rows = []
    for c in conds[1:]:
        a, b = cross[f"mlp_base|{c}"], cross[f"cnn|{c}"]
        rows.append(f"{NICE[c]} & {fmt_pct(a['acc'])} & {fmt_pct(b['acc'])} & {b['mean_conf']:.2f} & {b['ece']:.3f} \\\\")
    write_tex(cfg, "q2_cross.tex",
              "\\begin{tabular}{lrrrr}\n\\toprule\nClean-trained, test under & MLP Test\\% & CNN Test\\% & CNN mean conf & CNN ECE \\\\\n\\midrule\n"
              + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    hdr = " & ".join(c[:4] for c in CLASSES)
    rows = []
    for c in ["none", "lowpass", "highpass", "block_shuffle", "pixel_perm"]:
        pc = stats[f"cnn|{c}"]["per_class_acc"]
        rows.append(f"CNN {NICE[c]} & " + " & ".join(f"{100*v:.0f}" for v in pc) + " \\\\")
    pc = stats["mlp_base|none"]["per_class_acc"]
    rows.append("MLP clean & " + " & ".join(f"{100*v:.0f}" for v in pc) + " \\\\")
    write_tex(cfg, "q2_perclass.tex",
              "\\resizebox{\\linewidth}{!}{\\begin{tabular}{l" + "r" * 10 + "}\n\\toprule\nTest acc.\\ per class & " + hdr
              + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n")

    # --- figures: transform examples, accuracy bars, reliability diagrams
    raw = raw_splits(cfg)["test"][0][:4]
    fig, axes = plt.subplots(4, len(conds), figsize=(1.6 * len(conds), 6.4))
    for j, c in enumerate(conds):
        x = make_transform(c, cfg)(_to_float(raw)).clamp(0, 1)
        for i in range(4):
            axes[i, j].imshow(x[i].permute(1, 2, 0).numpy()); axes[i, j].axis("off")
        axes[0, j].set_title(NICE[c].replace("$\\times$", "x"), fontsize=8)
    fig.tight_layout(); fig.savefig(rpath(cfg, "figs", "q2_transforms.pdf")); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    xs = np.arange(len(conds))
    axes[0].bar(xs - .2, [100 * stats[f"mlp_base|{c}"]["acc"] for c in conds], .4, label="MLP-base")
    axes[0].bar(xs + .2, [100 * stats[f"cnn|{c}"]["acc"] for c in conds], .4, label="CNN")
    axes[0].set_xticks(xs); axes[0].set_xticklabels([NICE[c].replace("$\\times$", "x") for c in conds], fontsize=8)
    axes[0].set_ylabel("test accuracy (%)"); axes[0].legend(); axes[0].grid(axis="y", alpha=.3)
    axes[0].set_title("Trained and tested under the same condition")
    for c, col in [("none", "k"), ("lowpass", "C0"), ("highpass", "C3"), ("pixel_perm", "C2")]:
        rel = stats[f"cnn|{c}"]["reliability"]
        axes[1].plot([r[0] for r in rel], [r[1] for r in rel], "o-", color=col, ms=3,
                     label=f"CNN {NICE[c]} (ECE {stats[f'cnn|{c}']['ece']:.3f})")
    axes[1].plot([0, 1], [0, 1], "k:", lw=1)
    axes[1].set_xlabel("confidence"); axes[1].set_ylabel("accuracy"); axes[1].legend(fontsize=7)
    axes[1].set_title("Reliability (test)"); axes[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(rpath(cfg, "figs", "q2_results.pdf")); plt.close(fig)

    # confusion matrices of the CNN: clean vs low-pass vs high-pass
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for a, c in zip(axes, ["none", "lowpass", "highpass"]):
        cm = np.array(stats[f"cnn|{c}"]["confusion"]); cmn = cm / cm.sum(1, keepdims=True)
        a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        a.set_xticks(range(10)); a.set_yticks(range(10))
        a.set_xticklabels([k[:4] for k in CLASSES], rotation=90, fontsize=7); a.set_yticklabels([k[:4] for k in CLASSES], fontsize=7)
        a.set_title(f"CNN {NICE[c]}: {100*stats[f'cnn|{c}']['acc']:.1f}%", fontsize=9)
    fig.tight_layout(); fig.savefig(rpath(cfg, "figs", "q2_confusions.pdf")); plt.close(fig)


if __name__ == "__main__":
    cfg = load_cfg()
    set_seed(cfg["seed"])
    main(cfg)
