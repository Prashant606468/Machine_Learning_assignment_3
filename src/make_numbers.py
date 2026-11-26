"""Collect key numbers from every summary JSON into LaTeX macros (results/tables/numbers.tex)
so the report text never contains hand-copied numbers."""
import json
from common import load_cfg, rpath, CLASSES

cfg = load_cfg()
L = lambda n: json.load(open(rpath(cfg, "logs", n)))
m = {}
pct = lambda x: f"{100*x:.2f}"

m["SplitSeed"] = str(cfg["seed"])
a = L("q1a_summary.json")
for k, tag in [("mlp_base", "Base"), ("mlp_wide", "Wide"), ("local_mlp", "Local"),
               ("mlp_base__pixel_perm", "BasePerm"), ("local_mlp__pixel_perm", "LocalPerm")]:
    m[f"Qa{tag}Test"] = pct(a[k]["test_acc"]); m[f"Qa{tag}Val"] = pct(a[k]["best_val_acc"])
    m[f"Qa{tag}TrainFinal"] = pct(a["final_train_acc"][k]); m[f"Qa{tag}Params"] = f"{a[k]['params']:,}"
for r in cfg["q1"]["memorisation"]["noise_rates"]:
    b = L(f"q1b_mem_noise{int(r*100)}.json")
    t = "N" + {0: "Zero", 20: "Twenty", 40: "Forty", 60: "Sixty", 80: "Eighty", 100: "Full"}.get(
        int(r * 100), "".join("ABCDEFGHIJ"[int(ch)] for ch in str(int(r * 100))))
    m[f"Qb{t}BestEp"] = str(b["best_val_epoch"]); m[f"Qb{t}BestVal"] = pct(b["best_val_acc"])
    m[f"Qb{t}FinalVal"] = pct(b["final_val_acc"]); m[f"Qb{t}TestBest"] = pct(b["test_acc_best_val"])
    m[f"Qb{t}TestFinal"] = pct(b["test_acc_final"])
    if b.get("mem_at_best_val") is not None:
        m[f"Qb{t}MemAtBest"] = pct(b["mem_at_best_val"]); m[f"Qb{t}MemFinal"] = pct(b["final_mem"])
        m[f"Qb{t}EpMemFifty"] = str(b.get("epoch_mem_50") or "--")
s2 = L("q2_summary.json")
for key, v in s2["calibration"].items():
    mod, c = key.split("|")
    t = ("Mlp" if mod == "mlp_base" else "Cnn") + {"none": "Clean", "block_shuffle": "Block", "pixel_perm": "Perm",
                                                    "lowpass": "Low", "highpass": "High"}[c]
    m[f"Qc{t}Test"] = pct(v["acc"]); m[f"Qc{t}ECE"] = f"{v['ece']:.3f}"
s3 = L("q3_summary.json")
m["QdSelD"] = str(s3["selected_d"]); m["QdTestMSE"] = f"{s3['test_mse_selected']:.5f}"; m["QdTestPSNR"] = f"{s3['test_psnr_selected']:.2f}"
m["QdSigma"] = str(cfg["q3"]["noise_sigma"])
try:
    s4 = L("q4_summary.json")
    m["QeKernel"] = s4["selected_svm"]["kernel"]; m["QeC"] = f"{s4['selected_svm']['C']:g}"
    m["QeTest"] = pct(s4["svm_test_acc"]); m["QeARI"] = f"{s4['test_ARI']:.4f}"; m["QeNMI"] = f"{s4['test_NMI']:.4f}"
    m["QeClusterMethod"] = s4["cluster_selected"]["method"]; m["QeK"] = str(s4["cluster_selected"]["k"])
    m["QeSubset"] = f"{s4['representation']['subset_size']:,}"; m["QePCAVar"] = pct(s4["representation"]["pca_explained_var"])
except FileNotFoundError:
    pass

with open(rpath(cfg, "tables", "numbers.tex"), "w") as f:
    for k, v in m.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(m)} macros")
