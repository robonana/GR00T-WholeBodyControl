"""Compare joint-tracking RMSE across multiple calibration datasets."""
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from batch_joint_comparison import (
    BODY_29_NAMES,
    FPS,
    parse_dataset_spec,
    process_dataset,
)


def cross_calibration_analysis(all_stats, save_dir):
    fig = plt.figure(figsize=(30, 30))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25)
    names = [s['dataset'] for s in all_stats]
    n_ds = len(all_stats)
    colors = plt.cm.Set2(np.linspace(0, 1, n_ds))

    # 1. Grouped bar: per-joint RMSE per dataset
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(29)
    w = 0.8 / n_ds
    for i, s in enumerate(all_stats):
        vals = [s['per_joint_mean_rmse'][n] for n in BODY_29_NAMES]
        ax1.bar(x + i * w - 0.4 + w/2, vals, w, label=s['dataset'], color=colors[i],
                alpha=0.8, edgecolor='black', lw=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(BODY_29_NAMES, rotation=75, ha='right', fontsize=7)
    ax1.set_ylabel('Mean RMSE (rad)')
    ax1.set_title(
        f'Per-Joint RMSE Across {n_ds} Calibrations',
        fontweight='bold',
        fontsize=14,
    )
    ax1.legend(fontsize=8, ncol=3)
    ax1.axhline(0.1, color='green', ls='--', alpha=0.5)
    ax1.axhline(0.3, color='red', ls='--', alpha=0.5)
    ax1.grid(axis='y', alpha=0.3)

    # 2. Per-dataset overall RMSE bar chart
    ax2 = fig.add_subplot(gs[1, 0])
    overall = [s['overall_mean_rmse'] for s in all_stats]
    bars = ax2.bar(names, overall, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Overall Mean RMSE (rad)')
    ax2.set_title('Overall RMSE by Calibration', fontweight='bold')
    ax2.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, overall):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # 3. Episode count and total duration
    ax3 = fig.add_subplot(gs[1, 1])
    ep_counts = [s['n_episodes'] for s in all_stats]
    total_dur = [
        s.get('total_duration_s', s['total_frames'] / s.get('fps', FPS))
        for s in all_stats
    ]
    ax3_twin = ax3.twinx()
    b1 = ax3.bar(np.arange(n_ds) - 0.15, ep_counts, 0.3, color='steelblue', alpha=0.7, label='Episodes')
    b2 = ax3_twin.bar(np.arange(n_ds) + 0.15, total_dur, 0.3, color='coral', alpha=0.7, label='Duration (s)')
    ax3.set_xticks(np.arange(n_ds))
    ax3.set_xticklabels(names, rotation=30)
    ax3.set_ylabel('Episode Count', color='steelblue')
    ax3_twin.set_ylabel('Total Duration (s)', color='coral')
    ax3.set_title('Dataset Size Comparison', fontweight='bold')
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # 4. Heatmap: dataset x joint RMSE
    ax4 = fig.add_subplot(gs[2, :])
    rmse_matrix = np.array([[s['per_joint_mean_rmse'][n] for n in BODY_29_NAMES] for s in all_stats])
    im = ax4.imshow(rmse_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.7)
    ax4.set_yticks(range(n_ds))
    ax4.set_yticklabels(names, fontsize=9)
    ax4.set_xticks(range(29))
    ax4.set_xticklabels(BODY_29_NAMES, rotation=75, ha='right', fontsize=7)
    ax4.set_title('RMSE Heatmap (Calibration x Joint)', fontweight='bold', fontsize=14)
    for i in range(n_ds):
        for j in range(29):
            ax4.text(j, i, f'{rmse_matrix[i,j]:.3f}', ha='center', va='center', fontsize=5.5,
                    color='white' if rmse_matrix[i,j] > 0.35 else 'black')
    plt.colorbar(im, ax=ax4, label='RMSE (rad)')

    # 5. Ranking + analysis text
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')

    text = "=== CROSS-CALIBRATION ANALYSIS ===\n\n"
    sorted_ds = sorted(all_stats, key=lambda s: s['overall_mean_rmse'])
    text += "--- Ranking (best to worst) ---\n"
    for i, s in enumerate(sorted_ds):
        duration = s.get(
            'total_duration_s', s['total_frames'] / s.get('fps', FPS)
        )
        text += (
            f"  {i+1}. {s['dataset']:>10}: overall RMSE = "
            f"{s['overall_mean_rmse']:.4f} rad  "
            f"({s['n_episodes']} eps, {duration:.0f}s)\n"
        )

    text += f"\n--- Variation across calibrations ---\n"
    text += f"{'Joint':<25} {'min':>7} {'max':>7} {'range':>7} {'mean':>7} {'std':>7} {'CV%':>7}\n"
    text += "-" * 70 + "\n"

    for j, jname in enumerate(BODY_29_NAMES):
        vals = [s['per_joint_mean_rmse'][jname] for s in all_stats]
        mn, mx = min(vals), max(vals)
        rng = mx - mn
        avg = np.mean(vals)
        sd = np.std(vals)
        cv = (sd / avg * 100) if avg > 0.001 else 0
        marker = " <<<" if rng > 0.05 else ""
        text += f"{jname:<25} {mn:>7.4f} {mx:>7.4f} {rng:>7.4f} {avg:>7.4f} {sd:>7.4f} {cv:>6.1f}%{marker}\n"

    text += f"\n--- Most stable joints (lowest cross-calib std) ---\n"
    joint_vars = []
    for jname in BODY_29_NAMES:
        vals = [s['per_joint_mean_rmse'][jname] for s in all_stats]
        joint_vars.append((jname, np.std(vals), np.mean(vals)))
    joint_vars.sort(key=lambda x: x[1])
    for jname, sd, avg in joint_vars[:5]:
        text += f"  {jname}: std={sd:.4f}, mean={avg:.4f}\n"

    text += f"\n--- Most variable joints (highest cross-calib std) ---\n"
    for jname, sd, avg in joint_vars[-5:]:
        text += f"  {jname}: std={sd:.4f}, mean={avg:.4f}\n"

    ax5.text(0.02, 0.98, text, transform=ax5.transAxes, fontsize=7.5,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(
        f'Bottle Task: {n_ds} Calibrations Cross-Comparison',
        fontsize=16,
        fontweight='bold',
        y=1.01,
    )
    fig.savefig(os.path.join(save_dir, '_cross_calibration.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save text report
    with open(os.path.join(save_dir, '_cross_calibration_report.txt'), 'w') as f:
        f.write(text)
    print(f"\nCross-calibration analysis saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare joint RMSE across multiple task calibrations."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        type=parse_dataset_spec,
        metavar="PATH[=LABEL]",
        help="Calibration dataset; repeat this option for every calibration.",
    )
    parser.add_argument(
        "--output-dir",
        default="comparison/calibrations",
        help="Directory for per-dataset and cross-calibration reports.",
    )
    args = parser.parse_args()

    output_dir = str(Path(args.output_dir).expanduser())
    os.makedirs(output_dir, exist_ok=True)
    all_stats = []

    for data_path, label in args.dataset:
        save_dir = os.path.join(output_dir, label)
        stats = process_dataset(data_path, save_dir, label)
        if stats:
            all_stats.append(stats)

    if all_stats:
        cross_calibration_analysis(all_stats, output_dir)

    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
