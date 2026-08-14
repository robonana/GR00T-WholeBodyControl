#!/usr/bin/env python3
"""Compare joint-tracking RMSE between 3-point and SMPL teleop datasets."""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from batch_joint_comparison import BODY_29_NAMES, process_dataset


def _dataset_spec(value):
    """Parse PATH=LABEL:METHOD, where METHOD is 3-point or SMPL."""
    path_label, method_separator, method = value.rpartition(":")
    path_text, label_separator, label = path_label.rpartition("=")
    if (
        not method_separator
        or not label_separator
        or not path_text
        or not label
        or method.lower() not in {"3-point", "smpl"}
    ):
        raise argparse.ArgumentTypeError(
            "dataset must be PATH=LABEL:METHOD (METHOD: 3-point or SMPL)"
        )
    return str(Path(path_text).expanduser()), label, method


def _stats_spec(value):
    path_text, separator, method = value.rpartition(":")
    if (
        not separator
        or not path_text
        or method.lower() not in {"3-point", "smpl"}
    ):
        raise argparse.ArgumentTypeError(
            "stats must be PATH:METHOD (METHOD: 3-point or SMPL)"
        )
    return str(Path(path_text).expanduser()), method


def _load_stats(path, method):
    with open(path, encoding="utf-8") as handle:
        stats = json.load(handle)
    stats["method"] = method
    return stats


def _plot_method_comparison(point3, smpl, output_dir):
    point3_rmse = np.mean(
        [
            [stats["per_joint_mean_rmse"][name] for name in BODY_29_NAMES]
            for stats in point3
        ],
        axis=0,
    )
    smpl_rmse = np.mean(
        [
            [stats["per_joint_mean_rmse"][name] for name in BODY_29_NAMES]
            for stats in smpl
        ],
        axis=0,
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14))
    x = np.arange(29)
    width = 0.35
    ax1.bar(
        x - width / 2,
        point3_rmse,
        width,
        label=f"3-point ({len(point3)} datasets)",
        color="steelblue",
    )
    ax1.bar(
        x + width / 2,
        smpl_rmse,
        width,
        label=f"SMPL ({len(smpl)} datasets)",
        color="coral",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(BODY_29_NAMES, rotation=75, ha="right", fontsize=8)
    ax1.set_ylabel("Mean RMSE (rad)")
    ax1.set_title("3-point vs SMPL Teleop - Per-Joint RMSE")
    ax1.axhline(0.1, color="green", linestyle="--", alpha=0.5)
    ax1.axhline(0.3, color="red", linestyle="--", alpha=0.5)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    difference = smpl_rmse - point3_rmse
    ax2.bar(
        x,
        difference,
        color=["red" if value > 0 else "green" for value in difference],
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(BODY_29_NAMES, rotation=75, ha="right", fontsize=8)
    ax2.set_ylabel("SMPL - 3-point RMSE (rad)")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "_3point_vs_smpl.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        type=_dataset_spec,
        metavar="PATH=LABEL:METHOD",
        help="Raw LeRobot dataset; repeat for each dataset.",
    )
    parser.add_argument(
        "--stats",
        action="append",
        default=[],
        type=_stats_spec,
        metavar="PATH:METHOD",
        help="Existing _stats.json; repeat as needed.",
    )
    parser.add_argument("--output-dir", default="comparison/teleop_methods")
    args = parser.parse_args()
    if not args.dataset and not args.stats:
        parser.error("provide at least one --dataset or --stats")

    output_dir = str(Path(args.output_dir).expanduser())
    os.makedirs(output_dir, exist_ok=True)
    all_stats = []
    for data_path, label, method in args.dataset:
        stats = process_dataset(
            data_path,
            os.path.join(output_dir, label),
            label,
        )
        if stats:
            stats["method"] = method
            all_stats.append(stats)
    for stats_path, method in args.stats:
        all_stats.append(_load_stats(stats_path, method))

    point3 = [s for s in all_stats if s["method"].lower() == "3-point"]
    smpl = [s for s in all_stats if s["method"].lower() == "smpl"]
    print(f"3-point datasets: {len(point3)}, SMPL datasets: {len(smpl)}")
    if point3 and smpl:
        _plot_method_comparison(point3, smpl, output_dir)
    else:
        print("Both groups are required for a cross-method plot.")


if __name__ == "__main__":
    main()
