"""
Aggregate and plot the publishable Stackelberg-vs-CoPO ablation.

Reads every seed's progress.csv under each experiment dir, aligns by
training_iteration, computes mean +/- std across seeds, and produces:

  1) <out_prefix>_curves.png    -- 4-panel curves (reward / success / crash / out)
                                   with mean line and +/-1 std shading per algorithm
  2) <out_prefix>_final_bar.png -- final-iteration mean +/- std bar chart
  3) <out_prefix>_summary.csv   -- one row per (algo, seed) with final metrics

Usage (single dir per algorithm):
    python plot_publishable_ablation.py \
        --stack-dir minimal_stackelberg_1M \
        --copo-dir  copo_baseline_1M

Usage (merge multiple seed dirs per algorithm, e.g. an extra seed run later):
    python plot_publishable_ablation.py \
        --stack-dir minimal_stackelberg_1M,minimal_stackelberg_1M_s200 \
        --copo-dir  copo_baseline_1M,copo_baseline_1M_s200
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = [
    ("episode_reward_mean", "episode_reward_mean", None),
    ("success",             "success rate",        (-0.02, 1.02)),
    ("crash",               "crash rate",          (-0.02, 1.02)),
    ("out",                 "out_of_road rate",    (-0.02, 1.02)),
]


def find_progress_csvs(exp_dir: Path):
    """Return list of (seed_label, df) for every progress.csv under exp_dir."""
    out = []
    for csv in sorted(exp_dir.rglob("progress.csv")):
        # Trial dir name typically ends with .../seed=<N>_<timestamp>/progress.csv
        trial = csv.parent.name
        seed = "?"
        for part in trial.split("_"):
            if part.startswith("seed="):
                seed = part.split("=", 1)[1]
                break
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            print(f"  skipping {csv}: {e}")
            continue
        if "training_iteration" not in df.columns:
            print(f"  skipping (no training_iteration column): {csv}")
            continue
        out.append((seed, df))
    return out


def aggregate(runs):
    """
    runs: list of (seed_label, df).

    Returns a dict keyed by metric -> (x, mean, std, n) where x is the common
    training_iteration grid (intersection of all seeds, so we never plot beyond
    the slowest seed's progress).
    """
    if not runs:
        return None
    common_max_iter = min(int(df["training_iteration"].max()) for _, df in runs)
    x = np.arange(1, common_max_iter + 1)

    out = {}
    for col, _title, _ylim in METRICS:
        vals = []
        for _, df in runs:
            d = df[df["training_iteration"] <= common_max_iter]
            d = d.set_index("training_iteration").reindex(x)[col].values.astype(float)
            vals.append(d)
        arr = np.stack(vals, axis=0)  # (n_seeds, T)
        out[col] = (x, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0), arr.shape[0])
    return out, common_max_iter


def plot_curves(agg_by_algo, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    colors = {"Minimal Stackelberg (no CoPO)": "tab:blue",
              "CoPO baseline (original)":      "tab:gray"}

    for (col, title, ylim), ax in zip(METRICS, axes.ravel()):
        for label, agg in agg_by_algo.items():
            if agg is None:
                continue
            (x, mean, std, n) = agg[col]
            c = colors.get(label, None)
            ax.plot(x, mean, color=c, linewidth=1.7, label=f"{label} (n={n})")
            ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("training_iteration")
        ax.grid(True)
        if ylim:
            ax.set_ylim(*ylim)

    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(
        "Publishable ablation: Minimal Stackelberg (no CoPO) vs CoPO baseline\n"
        "MetaDrive Intersection, 1M env steps per seed (mean +/- std across seeds)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140)
    print("wrote", out_path)


def plot_final_bar(agg_by_algo, out_path):
    metrics_for_bar = ["episode_reward_mean", "success", "crash", "out"]
    labels = list(agg_by_algo.keys())
    means = {m: [] for m in metrics_for_bar}
    stds  = {m: [] for m in metrics_for_bar}
    for label in labels:
        agg = agg_by_algo[label]
        if agg is None:
            for m in metrics_for_bar:
                means[m].append(np.nan); stds[m].append(np.nan)
            continue
        for m in metrics_for_bar:
            x, mean, std, _ = agg[m]
            means[m].append(mean[-1]); stds[m].append(std[-1])

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    colors = ["tab:blue", "tab:gray"]
    for ax, m in zip(axes, metrics_for_bar):
        ax.bar(labels, means[m], yerr=stds[m], color=colors[: len(labels)], capsize=5)
        ax.set_title(f"final {m}")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.3)
        if m in ("success", "crash", "out"):
            ax.set_ylim(0, 1)
    fig.suptitle("Final-iteration metrics (mean +/- std across seeds)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=140)
    print("wrote", out_path)


def write_summary_csv(runs_by_algo, out_path):
    rows = []
    for algo, runs in runs_by_algo.items():
        for seed, df in runs:
            last = df.iloc[-1]
            row = {
                "algo": algo,
                "seed": seed,
                "iter": int(last.get("training_iteration", -1)),
                "timesteps_total": int(last.get("timesteps_total", -1)),
                "episode_reward_mean": float(last.get("episode_reward_mean", float("nan"))),
            }
            for m in ("success", "crash", "out", "length"):
                if m in df.columns:
                    row[m] = float(last[m])
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("wrote", out_path)


def find_progress_csvs_multi(dir_arg: str):
    """Accept comma-separated dir list and merge all seed runs across them."""
    runs = []
    for d in [s.strip() for s in dir_arg.split(",") if s.strip()]:
        runs.extend(find_progress_csvs(Path(d)))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stack-dir", required=True,
        help="experiment dir(s) of Minimal Stackelberg (comma-separated to merge multiple seed dirs)",
    )
    ap.add_argument(
        "--copo-dir", required=True,
        help="experiment dir(s) of CoPO baseline (comma-separated to merge multiple seed dirs)",
    )
    ap.add_argument("--out-prefix", default="publishable_ablation")
    args = ap.parse_args()

    stack_runs = find_progress_csvs_multi(args.stack_dir)
    copo_runs  = find_progress_csvs_multi(args.copo_dir)
    print(f"Found seeds: Stackelberg={len(stack_runs)}, CoPO={len(copo_runs)}")

    runs_by_algo = {
        "Minimal Stackelberg (no CoPO)": stack_runs,
        "CoPO baseline (original)":      copo_runs,
    }

    agg_by_algo = {}
    for label, runs in runs_by_algo.items():
        if not runs:
            print(f"WARNING: no progress.csv found for {label}")
            agg_by_algo[label] = None
            continue
        agg, max_iter = aggregate(runs)
        agg_by_algo[label] = agg
        print(f"  {label}: {len(runs)} seed(s), aligned to iter {max_iter}")

    plot_curves(agg_by_algo,    f"{args.out_prefix}_curves.png")
    plot_final_bar(agg_by_algo, f"{args.out_prefix}_final_bar.png")
    write_summary_csv(runs_by_algo, f"{args.out_prefix}_summary.csv")


if __name__ == "__main__":
    main()
