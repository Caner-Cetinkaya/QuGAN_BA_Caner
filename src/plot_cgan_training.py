"""
Visualization Script for Classical GAN Training Results
Matches the same format as plot_qgan_training.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json


# Colorblind-friendly palette (Deuteranopia)
PALETTE = {
    "blue": "#0173B2",
    "orange": "#DE8F05",
    "red": "#CC78BC",
    "green": "#CA9161",
    "yellow": "#ECE133",
    "black": "#029E73",
}


def smooth(arr, window=10):
    """
    Apply rolling average smoothing to array
    
    Args:
        arr: Input array
        window: Smoothing window size
    
    Returns:
        Smoothed array (same shape as input)
    """
    if len(arr) < window:
        return arr
    
    # Use np.convolve for rolling average
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="same")
    
    # Handle edge effects
    for i in range(min(window // 2, len(arr))):
        smoothed[i] = np.mean(arr[:i + window // 2 + 1])
    for i in range(len(arr) - window // 2, len(arr)):
        smoothed[i] = np.mean(arr[max(0, i - window // 2):])
    
    return smoothed


def _resolve_run_paths(run_path, config_path=None):
    run_path = Path(run_path)
    if run_path.is_dir():
        csv_path = run_path / "metrics.csv"
        resolved_config_path = run_path / "config.json"
    else:
        csv_path = run_path
        resolved_config_path = Path(config_path) if config_path is not None else csv_path.parent / "config.json"
    return csv_path, resolved_config_path


def _load_run(run_path, config_path=None):
    csv_path, config_path = _resolve_run_paths(run_path, config_path=config_path)

    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    score_fake_d_col = "score_fake_d" if "score_fake_d" in df.columns else "score_fake"
    score_fake_g_col = "score_fake_g" if "score_fake_g" in df.columns else score_fake_d_col

    config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    label = csv_path.parent.name
    model_type = config.get("model_type")
    if model_type:
        label = f"{label} ({model_type})"

    return {
        "label": label,
        "csv_path": csv_path,
        "config_path": config_path,
        "steps": df["step"].to_numpy(),
        "loss_d": df["loss_d"].to_numpy(),
        "loss_g": df["loss_g"].to_numpy(),
        "score_real": df["score_real"].to_numpy(),
        "score_fake_d": df[score_fake_d_col].to_numpy(),
        "score_fake_g": df[score_fake_g_col].to_numpy(),
        "grad_norm_d": df["grad_norm_d"].to_numpy(),
        "grad_norm_g": df["grad_norm_g"].to_numpy(),
    }


def _default_run_paths():
    log_dir = Path("logs")
    cgan_logs = sorted(path for path in log_dir.glob("cgan*") if path.is_dir())
    if not cgan_logs:
        print("[ERROR] No cgan* logs found in logs/ directory")
        sys.exit(1)
    return cgan_logs[-2:] if len(cgan_logs) >= 2 else cgan_logs


def _print_run_statistics(run):
    label = run["label"]
    loss_d = run["loss_d"]
    loss_g = run["loss_g"]
    score_real = run["score_real"]
    score_fake_d = run["score_fake_d"]
    score_fake_g = run["score_fake_g"]
    grad_norm_d = run["grad_norm_d"]
    grad_norm_g = run["grad_norm_g"]

    print(f"\n=== TRAINING STATISTICS: {label} ===\n")

    print("DISCRIMINATOR LOSS:")
    print(f"  Range: {loss_d.min():.6f} -> {loss_d.max():.6f}")
    print(f"  Trend: {loss_d[len(loss_d)//4]:.6f} -> {loss_d[-1]:.6f}")
    delta_d = loss_d[-1] - loss_d[len(loss_d)//4]
    print(f"  Delta (last 75%): {delta_d:+.6f}")
    print(f"  Mean: {loss_d.mean():.6f}")

    print("\nGENERATOR LOSS:")
    valid_g = loss_g[~np.isnan(loss_g)]
    if len(valid_g) > 0:
        valid_indices = np.where(~np.isnan(loss_g))[0]
        idx_start = valid_indices[len(valid_g) // 4]
        print(f"  Range: {valid_g.min():.6f} -> {valid_g.max():.6f}")
        print(f"  Trend: {loss_g[idx_start]:.6f} -> {valid_g[-1]:.6f}")
        delta_g = valid_g[-1] - loss_g[idx_start]
        improvement = (loss_g[idx_start] - valid_g[-1]) / loss_g[idx_start] * 100
        print(f"  Delta (last 75%): {delta_g:+.6f}")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"  Mean: {np.nanmean(loss_g):.6f}")
    else:
        print("  No generator updates recorded")

    print("\nDISCRIMINATOR CLASSIFICATION:")
    print(f"  Real Score  (real data):  {score_real[0]:.4f} -> {score_real[-1]:.4f}")
    print(f"  Fake Score D(gen data):   {score_fake_d[0]:.4f} -> {score_fake_d[-1]:.4f}")
    margin = score_real[-1] - score_fake_d[-1]
    print(f"  Margin (Real - Fake):     {margin:.4f}")

    print("\nGENERATOR FOOLING:")
    print(f"  Success Rate: {score_fake_g[0]:.4f} -> {score_fake_g[-1]:.4f} (Goal: 1.0)")
    print(f"  Current Fooling Rate: {score_fake_g[-1]:.1%}")

    print("\nGRADIENT NORMS:")
    non_zero_grad_g = grad_norm_g[grad_norm_g > 0]
    print(f"  D Gradient: Mean {grad_norm_d.mean():.6f}, Max {grad_norm_d.max():.6f}")
    if len(non_zero_grad_g) > 0:
        print(f"  G Gradient: Mean {non_zero_grad_g.mean():.6f}, Max {non_zero_grad_g.max():.6f}")

    print("\nCONVERGENCE ANALYSIS:")
    last_quarter_d = loss_d[int(len(loss_d) * 0.75):]
    last_quarter_g = loss_g[int(len(loss_g) * 0.75):]
    d_std = np.std(last_quarter_d)
    g_std = np.nanstd(last_quarter_g)

    if d_std < 0.01 and g_std < 0.01:
        status = "Good convergence (losses stable)"
    elif d_std < 0.05:
        status = "Equilibrium reached (oscillating)"
    else:
        status = "Training still ongoing (improving)"

    print(f"  D Loss Std (last 25%): {d_std:.6f}")
    print(f"  G Loss Std (last 25%): {g_std:.6f}")
    print(f"  Status: {status}")


def _plot_single_run_like_qgan(fig, axes, run, smooth_window=10):
    steps = run["steps"]
    loss_d = run["loss_d"]
    loss_g = run["loss_g"]
    score_real = run["score_real"]
    score_fake_d = run["score_fake_d"]
    score_fake_g = run["score_fake_g"]
    grad_norm_d = run["grad_norm_d"]
    grad_norm_g = run["grad_norm_g"]

    loss_d_smooth = smooth(loss_d, window=smooth_window)
    loss_g_smooth = smooth(np.nan_to_num(loss_g, nan=0.0), window=smooth_window)
    score_real_smooth = smooth(score_real, window=smooth_window)
    score_fake_d_smooth = smooth(score_fake_d, window=smooth_window)
    score_fake_g_smooth = smooth(score_fake_g, window=smooth_window)
    grad_norm_d_smooth = smooth(grad_norm_d, window=smooth_window)
    grad_norm_g_smooth = smooth(np.nan_to_num(grad_norm_g, nan=0.0), window=smooth_window)
    valid_gen = ~np.isnan(loss_g)
    valid_grad_g = grad_norm_g > 0

    model_name = run["label"].split("(")[-1].rstrip(")") if "(" in run["label"] else "classical_gan"
    if model_name == "classical_gan_comparable_to_qgan":
        title_prefix = "Classical GAN Training"
    else:
        title_prefix = "Classical GAN Training"
    fig.suptitle(f"{title_prefix} (Smoothed over {smooth_window} steps)", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(steps, loss_d, color=PALETTE["black"], alpha=0.3, linewidth=1, label="Raw")
    ax.plot(steps, loss_d_smooth, color=PALETTE["blue"], linewidth=2.5, label="Smoothed")
    ax.fill_between(steps, loss_d_smooth, alpha=0.2, color=PALETTE["blue"])
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Discriminator Loss", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    ax = axes[0, 1]
    if valid_gen.sum() > 0:
        ax.plot(steps[valid_gen], loss_g[valid_gen], color=PALETTE["black"], alpha=0.3, linewidth=1, label="Raw")
        ax.plot(steps[valid_gen], loss_g_smooth[valid_gen], color=PALETTE["orange"], linewidth=2.5, label="Smoothed")
        ax.fill_between(steps[valid_gen], loss_g_smooth[valid_gen], alpha=0.2, color=PALETTE["orange"])
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Generator Loss", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    ax = axes[0, 2]
    ax.plot(steps, loss_d_smooth, color=PALETTE["blue"], linewidth=2.5, label="D Loss")
    if valid_gen.sum() > 0:
        ax.plot(steps[valid_gen], loss_g_smooth[valid_gen], color=PALETTE["orange"], linewidth=2.5, label="G Loss")
        ax.fill_between(steps[valid_gen], loss_g_smooth[valid_gen], alpha=0.15, color=PALETTE["orange"])
    ax.fill_between(steps, loss_d_smooth, alpha=0.15, color=PALETTE["blue"])
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Loss Comparison (D vs G)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    ax = axes[1, 0]
    ax.plot(steps, score_real_smooth, color=PALETTE["blue"], linewidth=2.5, label="Real (should→1.0)")
    ax.plot(steps, score_fake_d_smooth, color=PALETTE["red"], linewidth=2.5, label="Fake-Disc (should→0.0)")
    ax.axhline(0.5, color=PALETTE["black"], linestyle="--", alpha=0.4, linewidth=1.5, label="Chance level")
    ax.fill_between(steps, score_real_smooth, alpha=0.15, color=PALETTE["blue"])
    ax.fill_between(steps, score_fake_d_smooth, alpha=0.15, color=PALETTE["red"])
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Score (D output probability)", fontsize=11)
    ax.set_title("Discriminator Classification Scores", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    ax = axes[1, 1]
    ax.plot(steps, score_fake_g_smooth, color=PALETTE["green"], linewidth=2.5, label="Fake-Gen (should→1.0)")
    ax.axhline(0.5, color=PALETTE["black"], linestyle="--", alpha=0.4, linewidth=1.5, label="Chance level")
    ax.fill_between(steps, score_fake_g_smooth, alpha=0.2, color=PALETTE["green"])
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Score (D output probability)", fontsize=11)
    ax.set_title("Generator Fooling Success", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    ax = axes[1, 2]
    ax.plot(steps, grad_norm_d_smooth, color=PALETTE["blue"], linewidth=2.5, label="D Grad Norm")
    if valid_grad_g.sum() > 0:
        ax.plot(steps[valid_grad_g], grad_norm_g_smooth[valid_grad_g], color=PALETTE["orange"], linewidth=2.5, label="G Grad Norm")
        ax.fill_between(steps[valid_grad_g], grad_norm_g_smooth[valid_grad_g], alpha=0.15, color=PALETTE["orange"])
    ax.fill_between(steps, grad_norm_d_smooth, alpha=0.15, color=PALETTE["blue"])
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Gradient Norm", fontsize=11)
    ax.set_title("Gradient Magnitudes", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)


def plot_cgan_training(csv_path=None, config_path=None):
    """
    Create comprehensive visualization of Classical GAN training
    
    Args:
        csv_path: Path to metrics.csv file
        config_path: Path to config.json file
    """
    
    if csv_path is None:
        run_paths = _default_run_paths()
        print("Using latest logs:", ", ".join(path.name for path in run_paths))
    elif isinstance(csv_path, (list, tuple)):
        run_paths = list(csv_path)
    else:
        run_paths = [csv_path]

    runs = []
    if config_path is not None and len(run_paths) == 1:
        runs.append(_load_run(run_paths[0], config_path=config_path))
    else:
        runs.extend(_load_run(path) for path in run_paths)

    run_colors = [
        PALETTE["blue"],
        PALETTE["orange"],
        PALETTE["black"],
        PALETTE["red"],
        PALETTE["green"],
        PALETTE["yellow"],
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    if len(runs) == 1:
        _plot_single_run_like_qgan(fig, axes, runs[0], smooth_window=10)
    else:
        title = "Classical GAN Training Dynamics - Comparison"
        fig.suptitle(title, fontsize=16, fontweight="bold")

        ax = axes[0, 0]
        for index, run in enumerate(runs):
            color = run_colors[index % len(run_colors)]
            steps = run["steps"]
            loss_d = run["loss_d"]
            loss_d_smooth = smooth(loss_d, window=10)
            ax.plot(steps, loss_d, alpha=0.15, color=color)
            ax.plot(steps, loss_d_smooth, color=color, linewidth=2, label=run["label"])
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title("Discriminator Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        for index, run in enumerate(runs):
            color = run_colors[index % len(run_colors)]
            steps = run["steps"]
            loss_g = run["loss_g"]
            valid_mask = ~np.isnan(loss_g)
            if np.any(valid_mask):
                valid_steps = steps[valid_mask]
                valid_g = loss_g[valid_mask]
                loss_g_smooth = smooth(np.nan_to_num(loss_g, nan=0.0), window=10)
                ax.plot(valid_steps, valid_g, alpha=0.15, color=color)
                ax.plot(steps[valid_mask], loss_g_smooth[valid_mask], color=color, linewidth=2, label=run["label"])
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title("Generator Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        for index, run in enumerate(runs):
            color = run_colors[index % len(run_colors)]
            steps = run["steps"]
            loss_d_smooth = smooth(run["loss_d"], window=10)
            loss_g_smooth = smooth(np.nan_to_num(run["loss_g"], nan=0.0), window=10)
            valid_mask = ~np.isnan(run["loss_g"])
            ax.plot(steps, loss_d_smooth, color=color, linewidth=2, label=f"{run['label']} - D")
            if np.any(valid_mask):
                ax.plot(steps[valid_mask], loss_g_smooth[valid_mask], color=color, linewidth=2, linestyle="--", label=f"{run['label']} - G")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title("D vs G Loss Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        for index, run in enumerate(runs):
            color = run_colors[index % len(run_colors)]
            steps = run["steps"]
            score_real_smooth = smooth(run["score_real"], window=10)
            score_fake_d_smooth = smooth(run["score_fake_d"], window=10)
            ax.plot(steps, score_real_smooth, color=color, linewidth=2, label=f"{run['label']} - real")
            ax.plot(steps, score_fake_d_smooth, color=color, linewidth=2, linestyle="--", label=f"{run['label']} - fake(D)")
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Discriminator Score")
        ax.set_title("Binary Classification Scores (Higher = More Real)")
        ax.set_ylim([-0.1, 1.1])
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        for index, run in enumerate(runs):
            color = run_colors[index % len(run_colors)]
            steps = run["steps"]
            score_fake_g_smooth = smooth(run["score_fake_g"], window=10)
            ax.plot(steps, score_fake_g_smooth, color=color, linewidth=2.5, label=run["label"])
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random guess (50%)")
        ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.7, label="Perfect fool (100%)")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Discriminator Score on Fakes")
        ax.set_title("Generator Fooling Rate (Goal: 1.0)")
        ax.set_ylim([0, 1.05])
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        for index, run in enumerate(runs):
            color = run_colors[index % len(run_colors)]
            steps = run["steps"]
            grad_norm_d_smooth = smooth(run["grad_norm_d"], window=10)
            grad_norm_g_smooth = smooth(np.nan_to_num(run["grad_norm_g"], nan=0.0), window=10)
            valid_mask = run["grad_norm_g"] > 0
            ax.plot(steps, grad_norm_d_smooth, color=color, linewidth=2, label=f"{run['label']} - D")
            if np.any(valid_mask):
                ax.plot(steps[valid_mask], grad_norm_g_smooth[valid_mask], color=color, linewidth=2, linestyle="--", label=f"{run['label']} - G")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Gradient Norm (clipped at 1.0)")
        ax.set_title("Gradient Magnitude Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    if len(runs) == 1:
        plot_path = runs[0]["csv_path"].parent / "plot_cgan_training.png"
    else:
        plot_path = Path("logs") / "plot_cgan_training_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Plot saved: {plot_path}")

    for run in runs:
        _print_run_statistics(run)

    if "agg" in plt.get_backend().lower():
        print("[INFO] Skipping interactive display because Matplotlib backend is non-interactive")
    else:
        plt.show()


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        plot_cgan_training()
    elif len(args) == 1:
        plot_cgan_training(args[0])
    else:
        plot_cgan_training(args)
