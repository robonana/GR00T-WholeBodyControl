"""
Batch joint-level comparison: observation.state (black) vs action.wbc (blue)
for 29 body joints across all episodes in multiple datasets.

Outputs per-episode plots + aggregated RMSE statistics + summary figures.
"""
import argparse
import os
import json
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BODY_29_NAMES = [
    'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee',
    'left_ankle_pitch', 'left_ankle_roll',
    'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_knee',
    'right_ankle_pitch', 'right_ankle_roll',
    'waist_yaw', 'waist_roll', 'waist_pitch',
    'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
    'left_elbow', 'left_wrist_roll', 'left_wrist_pitch', 'left_wrist_yaw',
    'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
    'right_elbow', 'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw',
]

# Body joint indices: 0-21 (legs+waist+arms) + 29-35 (right arm, skip left hand 22-28 and right hand 36-42)
BODY_IDX = list(range(0, 22)) + list(range(29, 36))

FPS = 50


def get_dataset_fps(data_path):
    """Read the recorded FPS, falling back to the historical 50 Hz default."""
    info_path = Path(data_path) / "meta" / "info.json"
    try:
        with info_path.open(encoding="utf-8") as handle:
            fps = float(json.load(handle)["fps"])
        return fps if fps > 0 else FPS
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return FPS


def load_episode(data_path, episode_idx):
    matches = sorted(
        (Path(data_path) / "data").glob(
            f"chunk-*/episode_{episode_idx:06d}.parquet"
        )
    )
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"multiple parquet files found for episode {episode_idx}")
    table = pq.read_table(matches[0])
    obs_state = np.stack([v.as_py() for v in table['observation.state']])  # (T, 43)
    act_wbc = np.stack([v.as_py() for v in table['action.wbc']])  # (T, 43)
    return obs_state, act_wbc, len(table)


def compute_rmse_per_joint(obs_state, act_wbc):
    obs_body = obs_state[:, BODY_IDX]  # (T, 29)
    act_body = act_wbc[:, BODY_IDX]    # (T, 29)
    diff = obs_body - act_body
    rmse = np.sqrt(np.mean(diff ** 2, axis=0))  # (29,)
    return rmse, obs_body, act_body


def plot_episode_comparison(
    obs_body, act_body, episode_idx, dataset_name, save_dir, duration_s, fps
):
    n_joints = 29
    ncols = 5
    nrows = (n_joints + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(28, 18))
    fig.suptitle(f'{dataset_name} - Episode {episode_idx} ({duration_s:.1f}s, {len(obs_body)} frames)\n'
                 f'Black: observation.state | Blue: action.wbc (decoder output)',
                 fontsize=14, fontweight='bold')

    t = np.arange(len(obs_body)) / fps

    for i, ax in enumerate(axes.flat):
        if i >= n_joints:
            ax.set_visible(False)
            continue
        ax.plot(t, obs_body[:, i], color='black', linewidth=0.6, alpha=0.8, label='observation')
        ax.plot(t, act_body[:, i], color='blue', linewidth=0.6, alpha=0.7, label='decoder/wbc')
        rmse_val = np.sqrt(np.mean((obs_body[:, i] - act_body[:, i]) ** 2))
        ax.set_title(f'{BODY_29_NAMES[i]}\nRMSE={rmse_val:.4f}', fontsize=8)
        ax.tick_params(labelsize=6)

        # Color title based on RMSE severity
        if rmse_val > 0.3:
            ax.title.set_color('red')
        elif rmse_val > 0.1:
            ax.title.set_color('orange')
        else:
            ax.title.set_color('green')

    # Add single legend
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'ep{episode_idx:03d}.png')
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_dataset_rmse_summary(all_rmse, dataset_name, save_path):
    """Plot aggregated RMSE across all episodes of a dataset."""
    all_rmse = np.array(all_rmse)  # (n_episodes, 29)
    mean_rmse = np.mean(all_rmse, axis=0)
    std_rmse = np.std(all_rmse, axis=0)
    max_rmse = np.max(all_rmse, axis=0)
    min_rmse = np.min(all_rmse, axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14))

    # Bar chart: mean RMSE per joint
    x = np.arange(29)
    colors = ['green' if m < 0.1 else 'orange' if m < 0.3 else 'red' for m in mean_rmse]
    bars = ax1.bar(x, mean_rmse, yerr=std_rmse, capsize=3, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(BODY_29_NAMES, rotation=75, ha='right', fontsize=8)
    ax1.set_ylabel('RMSE (rad)', fontsize=12)
    ax1.set_title(f'{dataset_name} - Mean RMSE per Joint (across {len(all_rmse)} episodes)\n'
                  f'Green < 0.1 | Orange 0.1-0.3 | Red > 0.3', fontsize=13, fontweight='bold')
    ax1.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='0.1 rad')
    ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='0.3 rad')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, mean_rmse):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    # Box plot: RMSE distribution per joint
    bp = ax2.boxplot(all_rmse, tick_labels=BODY_29_NAMES, patch_artist=True)
    ax2.set_xticklabels(BODY_29_NAMES, rotation=75, ha='right', fontsize=8)
    ax2.set_ylabel('RMSE (rad)', fontsize=12)
    ax2.set_title(f'{dataset_name} - RMSE Distribution per Joint', fontsize=13, fontweight='bold')
    ax2.axhline(y=0.1, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.3, color='red', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)

    for patch, median in zip(bp['boxes'], np.median(all_rmse, axis=0)):
        color = 'green' if median < 0.1 else 'orange' if median < 0.3 else 'red'
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return mean_rmse, std_rmse, max_rmse, min_rmse


def process_dataset(data_path, save_dir, dataset_name):
    """Process all episodes in a dataset."""
    os.makedirs(save_dir, exist_ok=True)

    episodes_path = os.path.join(data_path, 'meta', 'episodes.jsonl')
    with open(episodes_path) as f:
        episodes = [json.loads(l) for l in f]

    all_rmse = []
    episode_stats = []
    fps = get_dataset_fps(data_path)

    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"Episodes: {len(episodes)}, Save dir: {save_dir}")
    print(f"{'='*60}")

    for ep in episodes:
        ep_idx = ep['episode_index']
        ep_len = ep['length']

        result = load_episode(data_path, ep_idx)
        if result is None:
            print(f"  Ep {ep_idx}: SKIPPED (no parquet)")
            continue
        obs_state, act_wbc, n_frames = result

        if n_frames < fps:
            print(f"  Ep {ep_idx}: SKIPPED (too short: {n_frames} frames)")
            continue

        rmse, obs_body, act_body = compute_rmse_per_joint(obs_state, act_wbc)
        all_rmse.append(rmse)
        duration_s = n_frames / fps

        # Save per-episode plot
        plot_episode_comparison(
            obs_body, act_body, ep_idx, dataset_name, save_dir, duration_s, fps
        )

        ep_mean = np.mean(rmse)
        ep_max_joint = BODY_29_NAMES[np.argmax(rmse)]
        ep_max_val = np.max(rmse)
        episode_stats.append({
            'episode': ep_idx,
            'frames': n_frames,
            'duration_s': duration_s,
            'mean_rmse': float(ep_mean),
            'max_rmse_joint': ep_max_joint,
            'max_rmse_value': float(ep_max_val),
            'per_joint_rmse': rmse.tolist(),
        })

        print(f"  Ep {ep_idx:3d}: {n_frames:5d} frames ({duration_s:6.1f}s) | "
              f"mean RMSE={ep_mean:.4f} | max joint={ep_max_joint} ({ep_max_val:.4f})")

    if not all_rmse:
        print(f"  No valid episodes found!")
        return None

    all_rmse = np.array(all_rmse)

    # Save summary plot
    summary_path = os.path.join(save_dir, '_summary_rmse.png')
    mean_rmse, std_rmse, max_rmse, min_rmse = plot_dataset_rmse_summary(
        all_rmse, dataset_name, summary_path
    )

    # Save stats JSON
    stats = {
        'dataset': dataset_name,
        'n_episodes': len(episode_stats),
        'total_frames': sum(s['frames'] for s in episode_stats),
        'fps': fps,
        'total_duration_s': sum(s['duration_s'] for s in episode_stats),
        'per_joint_mean_rmse': {n: float(v) for n, v in zip(BODY_29_NAMES, mean_rmse)},
        'per_joint_std_rmse': {n: float(v) for n, v in zip(BODY_29_NAMES, std_rmse)},
        'per_joint_max_rmse': {n: float(v) for n, v in zip(BODY_29_NAMES, max_rmse)},
        'overall_mean_rmse': float(np.mean(mean_rmse)),
        'episodes': episode_stats,
    }
    stats_path = os.path.join(save_dir, '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n  Summary saved: {summary_path}")
    print(f"  Stats saved: {stats_path}")
    print(f"  Overall mean RMSE: {stats['overall_mean_rmse']:.4f}")

    return stats


def plot_white_vs_bottle(white_stats_list, bottle_stats, save_dir):
    """Compare white vs bottle aggregated RMSE."""
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))

    # Aggregate all white stats
    white_all_rmse = []
    for ws in white_stats_list:
        for ep in ws['episodes']:
            white_all_rmse.append(ep['per_joint_rmse'])
    white_all_rmse = np.array(white_all_rmse)

    bottle_all_rmse = []
    for ep in bottle_stats['episodes']:
        bottle_all_rmse.append(ep['per_joint_rmse'])
    bottle_all_rmse = np.array(bottle_all_rmse)

    white_mean = np.mean(white_all_rmse, axis=0)
    bottle_mean = np.mean(bottle_all_rmse, axis=0)

    x = np.arange(29)
    width = 0.35

    # 1. Grouped bar chart
    ax = axes[0, 0]
    ax.bar(x - width/2, white_mean, width, label=f'White (n={len(white_all_rmse)})',
           color='gray', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, bottle_mean, width, label=f'Bottle (n={len(bottle_all_rmse)})',
           color='blue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(BODY_29_NAMES, rotation=75, ha='right', fontsize=7)
    ax.set_ylabel('Mean RMSE (rad)')
    ax.set_title('White vs Bottle - Mean RMSE per Joint', fontweight='bold')
    ax.legend()
    ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    # 2. Difference (White - Bottle)
    ax = axes[0, 1]
    diff = white_mean - bottle_mean
    colors = ['red' if d > 0 else 'green' for d in diff]
    ax.bar(x, diff, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(BODY_29_NAMES, rotation=75, ha='right', fontsize=7)
    ax.set_ylabel('RMSE Difference (rad)')
    ax.set_title('White - Bottle (Red=White worse, Green=Bottle worse)', fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    # 3. Overall distribution comparison
    ax = axes[1, 0]
    white_ep_means = np.mean(white_all_rmse, axis=1)
    bottle_ep_means = np.mean(bottle_all_rmse, axis=1)
    ax.hist(white_ep_means, bins=30, alpha=0.6, label='White', color='gray', edgecolor='black')
    ax.hist(bottle_ep_means, bins=30, alpha=0.6, label='Bottle', color='blue', edgecolor='black')
    ax.set_xlabel('Per-Episode Mean RMSE (rad)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Per-Episode Mean RMSE', fontweight='bold')
    ax.legend()
    ax.axvline(x=np.mean(white_ep_means), color='gray', linestyle='--', linewidth=2)
    ax.axvline(x=np.mean(bottle_ep_means), color='blue', linestyle='--', linewidth=2)

    # 4. Text summary
    ax = axes[1, 1]
    ax.axis('off')

    # Sort joints by white RMSE (worst to best)
    sorted_idx = np.argsort(white_mean)[::-1]
    text = "=== WHITE vs BOTTLE COMPARISON ===\n\n"
    text += f"White: {len(white_all_rmse)} episodes, overall mean RMSE = {np.mean(white_mean):.4f} rad\n"
    text += f"Bottle: {len(bottle_all_rmse)} episodes, overall mean RMSE = {np.mean(bottle_mean):.4f} rad\n\n"

    text += "--- Joint Ranking (worst to best, by White RMSE) ---\n"
    text += f"{'Joint':<25} {'White':>8} {'Bottle':>8} {'Diff':>8}\n"
    text += "-" * 55 + "\n"
    for idx in sorted_idx:
        d = white_mean[idx] - bottle_mean[idx]
        marker = " <<<" if abs(d) > 0.05 else ""
        text += f"{BODY_29_NAMES[idx]:<25} {white_mean[idx]:>8.4f} {bottle_mean[idx]:>8.4f} {d:>+8.4f}{marker}\n"

    text += "\n--- Top 5 Worst Joints (White) ---\n"
    for i, idx in enumerate(sorted_idx[:5]):
        text += f"  {i+1}. {BODY_29_NAMES[idx]}: {white_mean[idx]:.4f} rad\n"

    text += "\n--- Top 5 Best Joints (White) ---\n"
    for i, idx in enumerate(sorted_idx[-5:]):
        text += f"  {i+1}. {BODY_29_NAMES[idx]}: {white_mean[idx]:.4f} rad\n"

    text += "\n--- Top 5 Worst Joints (Bottle) ---\n"
    bottle_sorted = np.argsort(bottle_mean)[::-1]
    for i, idx in enumerate(bottle_sorted[:5]):
        text += f"  {i+1}. {BODY_29_NAMES[idx]}: {bottle_mean[idx]:.4f} rad\n"

    text += "\n--- Top 5 Best Joints (Bottle) ---\n"
    for i, idx in enumerate(bottle_sorted[-5:]):
        text += f"  {i+1}. {BODY_29_NAMES[idx]}: {bottle_mean[idx]:.4f} rad\n"

    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('White vs Bottle - Comprehensive RMSE Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, '_white_vs_bottle.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWhite vs Bottle comparison saved: {save_path}")

    # Also save the text report
    report_path = os.path.join(save_dir, '_white_vs_bottle_report.txt')
    with open(report_path, 'w') as f:
        f.write(text)
    print(f"Report saved: {report_path}")


def parse_dataset_spec(value):
    """Parse PATH[=LABEL] without confusing the colon in Windows paths."""
    path_text, separator, label = value.rpartition("=")
    if not separator:
        path_text = value
        label = Path(value).expanduser().resolve().name
    if not path_text or not label:
        raise argparse.ArgumentTypeError("dataset must be PATH or PATH=LABEL")
    return str(Path(path_text).expanduser()), label


def main():
    parser = argparse.ArgumentParser(
        description="Plot 29-joint target/measured RMSE for LeRobot datasets."
    )
    parser.add_argument(
        "--white-dataset",
        action="append",
        default=[],
        type=parse_dataset_spec,
        metavar="PATH[=LABEL]",
        help="Reference-group dataset; repeat for multiple datasets.",
    )
    parser.add_argument(
        "--bottle-dataset",
        type=parse_dataset_spec,
        metavar="PATH[=LABEL]",
        help="Candidate dataset used for the cross-group comparison.",
    )
    parser.add_argument(
        "--output-dir",
        default="comparison/joints",
        help="Directory for plots, JSON statistics, and reports.",
    )
    args = parser.parse_args()

    if not args.white_dataset and not args.bottle_dataset:
        parser.error("provide at least one --white-dataset or --bottle-dataset")

    base_dir = str(Path(args.output_dir).expanduser())
    white_stats_list = []
    for data_path, name in args.white_dataset:
        stats = process_dataset(
            data_path,
            os.path.join(base_dir, "white", name),
            name,
        )
        if stats:
            white_stats_list.append(stats)

    bottle_stats = None
    if args.bottle_dataset:
        bottle_data_path, bottle_name = args.bottle_dataset
        bottle_stats = process_dataset(
            bottle_data_path,
            os.path.join(base_dir, "bottle", bottle_name),
            bottle_name,
        )

    if white_stats_list and bottle_stats:
        plot_white_vs_bottle(white_stats_list, bottle_stats, base_dir)

    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
