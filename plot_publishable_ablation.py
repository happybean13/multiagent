"""
Aggregate and plot the publishable Stackelberg-vs-CoPO ablation.

Reads every seed's progress.csv under each experiment dir, aligns by
training_iteration, computes mean +/- std across seeds, and produces:

  1) <out_prefix>_curves.png    -- 4-panel curves (reward / success / crash / out)
                                   with mean line and +/-1 std shading per algorithm
  2) <out_prefix>_final_bar.png -- final-iteration mean +/- std bar chart
  3) <out_prefix>_summary.csv   -- one row per (algo, seed) with final metrics

Supports an optional third method (CPA Stackelberg) via --cpa-dir, and
per-method seed filters so seed counts can be aligned across methods when
needed.

Usage (2-method, single dir each):
    python plot_publishable_ablation.py \
        --stack-dir minimal_stackelberg_1M \
        --copo-dir  copo_baseline_1M

Usage (2-method, merge multiple seed dirs per algorithm):
    python plot_publishable_ablation.py \
        --stack-dir minimal_stackelberg_1M,minimal_stackelberg_1M_s200 \
        --copo-dir  copo_baseline_1M,copo_baseline_1M_s200

Usage (3-method with CPA Stackelberg, all 3 seeds each):
    python plot_publishable_ablation.py \
        --stack-dir minimal_stackelberg_1M,minimal_stackelberg_1M_s200 \
        --copo-dir  copo_baseline_1M,copo_baseline_1M_s200 \
        --cpa-dir   minimal_stackelberg_cpa_1M \
        --out-prefix publishable_ablation_3way_n3

Usage (3-method n=2 slice, restrict CPA seeds to {0, 100}):
    python plot_publishable_ablation.py \
        --stack-dir minimal_stackelberg_1M \
        --copo-dir  copo_baseline_1M \
        --cpa-dir   minimal_stackelberg_cpa_1M --cpa-seeds 0,100 \
        --out-prefix publishable_ablation_3way
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

ALGO_COLORS = {
    "Minimal Stackelberg (no CoPO)":  "tab:blue",
    "CoPO baseline (original)":       "tab:gray",
    "Stackelberg + 2D CPA (no CoPO)": "tab:orange",
}


def find_progress_csvs(exp_dir: Path, allow_seeds=None):
    """Return list of (seed_label, df) for every progress.csv under exp_dir.

    If allow_seeds is a set/list of seed-label strings, only those seeds are
    kept. Pass None (default) to keep every seed found.
    """
    out = []
    for csv in sorted(exp_dir.rglob("progress.csv")):
        # Trial dir name typically ends with .../seed=<N>_<timestamp>/progress.csv
        trial = csv.parent.name
        seed = "?"
        for part in trial.split("_"):
            if part.startswith("seed="):
                seed = part.split("=", 1)[1]
                break
        if allow_seeds is not None and seed not in allow_seeds:
            continue
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


def _title_line():
    return "MetaDrive Intersection, 1M env steps per seed (mean +/- std across seeds)"


def _algos_in_title(agg_by_algo):
    return " vs ".join(agg_by_algo.keys())


def plot_curves(agg_by_algo, out_path, title_suffix=""):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    for (col, title, ylim), ax in zip(METRICS, axes.ravel()):
        for label, agg in agg_by_algo.items():
            if agg is None:
                continue
            (x, mean, std, n) = agg[col]
            c = ALGO_COLORS.get(label, None)
            ax.plot(x, mean, color=c, linewidth=1.7, label=f"{label} (n={n})")
            ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("training_iteration")
        ax.grid(True)
        if ylim:
            ax.set_ylim(*ylim)

    axes[0, 0].legend(loc="best", fontsize=8)
    head = "Publishable ablation: " + _algos_in_title(agg_by_algo)
    if title_suffix:
        head += "  [" + title_suffix + "]"
    fig.suptitle(head + "\n" + _title_line(), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140)
    print("wrote", out_path)


def plot_final_bar(agg_by_algo, out_path, title_suffix=""):
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
    bar_colors = [ALGO_COLORS.get(l, "tab:gray") for l in labels]
    for ax, m in zip(axes, metrics_for_bar):
        ax.bar(labels, means[m], yerr=stds[m], color=bar_colors, capsize=5)
        ax.set_title(f"final {m}")
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        if m in ("success", "crash", "out"):
            ax.set_ylim(0, 1)
    head = "Final-iteration metrics (mean +/- std across seeds)"
    if title_suffix:
        head += "  [" + title_suffix + "]"
    fig.suptitle(head, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
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


def find_progress_csvs_multi(dir_arg: str, allow_seeds=None):
    """Accept comma-separated dir list and merge all seed runs across them."""
    runs = []
    for d in [s.strip() for s in dir_arg.split(",") if s.strip()]:
        runs.extend(find_progress_csvs(Path(d), allow_seeds=allow_seeds))
    return runs


def _seed_filter(s):
    if s is None or not str(s).strip():
        return None
    return set(x.strip() for x in str(s).split(",") if x.strip())


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
    ap.add_argument(
        "--cpa-dir", default=None,
        help="optional experiment dir(s) of CPA Stackelberg (comma-separated). "
             "When set, a 3-method comparison is produced.",
    )
    ap.add_argument("--stack-seeds", default=None,
                    help="comma-separated seed labels to keep (default: all). "
                         "Useful for aligning seed counts across methods.")
    ap.add_argument("--copo-seeds",  default=None)
    ap.add_argument("--cpa-seeds",   default=None)
    ap.add_argument("--out-prefix",   default="publishable_ablation")
    ap.add_argument("--title-suffix", default="",
                    help="optional bracketed tag appended to plot titles "
                         "(e.g. 'n=3 seeds').")
    args = ap.parse_args()

    stack_runs = find_progress_csvs_multi(
        args.stack_dir, allow_seeds=_seed_filter(args.stack_seeds)
    )
    copo_runs = find_progress_csvs_multi(
        args.copo_dir, allow_seeds=_seed_filter(args.copo_seeds)
    )
    print(f"Found seeds: Stackelberg={len(stack_runs)}, CoPO={len(copo_runs)}")

    runs_by_algo = {
        "Minimal Stackelberg (no CoPO)": stack_runs,
        "CoPO baseline (original)":      copo_runs,
    }
    if args.cpa_dir:
        cpa_runs = find_progress_csvs_multi(
            args.cpa_dir, allow_seeds=_seed_filter(args.cpa_seeds)
        )
        runs_by_algo["Stackelberg + 2D CPA (no CoPO)"] = cpa_runs
        print(f"                CPA Stackelberg={len(cpa_runs)}")

    agg_by_algo = {}
    for label, runs in runs_by_algo.items():
        if not runs:
            print(f"WARNING: no progress.csv found for {label}")
            agg_by_algo[label] = None
            continue
        agg, max_iter = aggregate(runs)
        agg_by_algo[label] = agg
        print(f"  {label}: {len(runs)} seed(s), aligned to iter {max_iter}")

    plot_curves(agg_by_algo,    f"{args.out_prefix}_curves.png",    title_suffix=args.title_suffix)
    plot_final_bar(agg_by_algo, f"{args.out_prefix}_final_bar.png", title_suffix=args.title_suffix)
    write_summary_csv(runs_by_algo, f"{args.out_prefix}_summary.csv")


if __name__ == "__main__":
    main()
