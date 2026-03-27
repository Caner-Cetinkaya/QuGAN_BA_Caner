"""
Plot QGAN adversarial training metrics with smoothing and colorblind-friendly colors.

Features:
- Load qgan_TIMESTAMP/metrics.csv
- 10-step rolling average for smooth curves
- Colorblind-friendly palette (Deuteranopia friendly)
- Plots: D-loss, G-loss, scores, grad norms, convergence trends
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import json

from plot_cgan_training import smooth


# Colorblind-friendly palette (Deuteranopia - Green-blind friendly)
# Source: https://jfly.uni-koeln.de/color/
PALETTE = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'red': '#CC78BC',
    'green': '#CA9161',
    'yellow': '#ECE133',
    'black': '#029E73',
}


def plot_qgan_training(csv_path: Path, config_path: Optional[Path] = None, 
                       output_dir: Optional[Path] = None, smooth_window: int = 10):
    """Plot QGAN training metrics with smoothing.
    
    Args:
        csv_path: Path to metrics.csv
        config_path: Path to config.json (auto-detected if None)
        output_dir: Where to save plot (defaults to csv_path parent)
        smooth_window: Rolling average window (default 10)
    """
    
    # Load metrics
    rows = []
    with csv_path.open(newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: float(v) if k != 'step' else int(v) for k, v in row.items()})
    
    if not rows:
        print("ERROR: No data found in CSV")
        return
    
    print(f"Loaded {len(rows)} rows from CSV")
    
    # Extract data
    steps = np.array([r['step'] for r in rows])
    disc_loss = np.array([r['disc_loss'] for r in rows])
    disc_grad_norm = np.array([r['disc_grad_norm'] for r in rows])
    disc_loss_real = np.array([r['disc_loss_real'] for r in rows])
    disc_loss_fake = np.array([r['disc_loss_fake'] for r in rows])
    gen_loss = np.array([r['gen_loss'] for r in rows])
    gen_grad_norm = np.array([r['gen_grad_norm'] for r in rows])
    
    real_score_mean = np.array([r['real_score_mean'] for r in rows])
    fake_score_mean_disc = np.array([r['fake_score_mean_disc'] for r in rows])
    fake_score_mean_gen = np.array([r['fake_score_mean_gen'] for r in rows])
    
    # Rolling average smoothing (10 steps)
    def smooth(arr, window=smooth_window):
        if len(arr) < window:
            return arr
        kernel = np.ones(window) / window
        padded = np.pad(arr, (window//2, window//2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed[:len(arr)]
    
    disc_loss_smooth = smooth(disc_loss)
    gen_loss_smooth = smooth(gen_loss)
    disc_loss_real_smooth = smooth(disc_loss_real)
    disc_loss_fake_smooth = smooth(disc_loss_fake)
    disc_grad_norm_smooth = smooth(disc_grad_norm)
    gen_grad_norm_smooth = smooth(gen_grad_norm)
    real_score_smooth = smooth(real_score_mean)
    fake_score_disc_smooth = smooth(fake_score_mean_disc)
    fake_score_gen_smooth = smooth(fake_score_mean_gen)
    
    # Load config
    config_path = config_path or csv_path.parent / "config.json"
    loss_type = "unknown"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            loss_type = config.get("LOSS_TYPE", "unknown")
            print(f"Loss type: {loss_type}")
    
    # Output directory
    if output_dir is None:
        output_dir = csv_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== Create Figure =====
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'QuGAN Adversarial Training (Smoothed over {smooth_window} steps)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Discriminator Loss
    ax = axes[0, 0]
    ax.plot(steps, disc_loss, color=PALETTE['black'], alpha=0.3, linewidth=1, label='Raw')
    ax.plot(steps, disc_loss_smooth, color=PALETTE['blue'], linewidth=2.5, label='Smoothed')
    ax.fill_between(steps, disc_loss_smooth, alpha=0.2, color=PALETTE['blue'])
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Discriminator Loss', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Plot 2: Generator Loss (handling NaN for warmup)
    ax = axes[0, 1]
    valid_gen = ~np.isnan(gen_loss)
    if valid_gen.sum() > 0:
        ax.plot(steps[valid_gen], gen_loss[valid_gen], color=PALETTE['black'], 
               alpha=0.3, linewidth=1, label='Raw')
        ax.plot(steps[valid_gen], gen_loss_smooth[valid_gen], color=PALETTE['orange'], 
               linewidth=2.5, label='Smoothed')
        ax.fill_between(steps[valid_gen], gen_loss_smooth[valid_gen], alpha=0.2, 
                       color=PALETTE['orange'])
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Generator Loss', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Plot 3: Both Losses on same axis
    # Plot 3: Discriminator Loss Breakdown
    ax = axes[0, 2]
    ax.plot(steps, disc_loss_smooth, color=PALETTE['blue'], linewidth=2.5, label='D Loss Total')
    ax.plot(steps, disc_loss_real_smooth, color=PALETTE['orange'], linewidth=2.5, label='D Loss Real')
    ax.plot(steps, disc_loss_fake_smooth, color=PALETTE['red'], linewidth=2.5, label='D Loss Fake')
    ax.fill_between(steps, disc_loss_smooth, alpha=0.12, color=PALETTE['blue'])
    ax.fill_between(steps, disc_loss_real_smooth, alpha=0.10, color=PALETTE['orange'])
    ax.fill_between(steps, disc_loss_fake_smooth, alpha=0.10, color=PALETTE['red'])
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Discriminator Loss Breakdown', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Plot 4: Discriminator Scores
    ax = axes[1, 0]
    ax.plot(steps, real_score_smooth, color=PALETTE['blue'], linewidth=2.5, label='Real (should→1.0)')
    ax.plot(steps, fake_score_disc_smooth, color=PALETTE['red'], linewidth=2.5, label='Fake-Disc (should→0.0)')
    ax.axhline(0.5, color=PALETTE['black'], linestyle='--', alpha=0.4, linewidth=1.5, label='Chance level')
    ax.fill_between(steps, real_score_smooth, alpha=0.15, color=PALETTE['blue'])
    ax.fill_between(steps, fake_score_disc_smooth, alpha=0.15, color=PALETTE['red'])
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Score (D output probability)', fontsize=11)
    ax.set_title("Discriminator Classification Scores", fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Plot 5: Generator Scores (fooling discriminator)
    ax = axes[1, 1]
    ax.plot(steps, fake_score_gen_smooth, color=PALETTE['green'], linewidth=2.5, 
           label='Fake-Gen (should→1.0)')
    ax.axhline(0.5, color=PALETTE['black'], linestyle='--', alpha=0.4, linewidth=1.5, label='Chance level')
    ax.fill_between(steps, fake_score_gen_smooth, alpha=0.2, color=PALETTE['green'])
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Score (D output probability)', fontsize=11)
    ax.set_title("Generator Fooling Success", fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Plot 6: Gradient Norms
    ax = axes[1, 2]
    ax.plot(steps, disc_grad_norm_smooth, color=PALETTE['blue'], linewidth=2.5, label='D Grad Norm')
    ax.plot(steps[valid_gen], gen_grad_norm_smooth[valid_gen], color=PALETTE['orange'], 
           linewidth=2.5, label='G Grad Norm')
    ax.fill_between(steps, disc_grad_norm_smooth, alpha=0.15, color=PALETTE['blue'])
    if valid_gen.sum() > 0:
        ax.fill_between(steps[valid_gen], gen_grad_norm_smooth[valid_gen], alpha=0.15, 
                       color=PALETTE['orange'])
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Gradient Norm', fontsize=11)
    ax.set_title('Gradient Magnitudes', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / "plot_qgan_training.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Plot saved: {output_path}")
    
    # ===== Statistics =====
    print(f"\n{'='*70}")
    print(f"QGAN TRAINING STATISTICS (Smoothed, {smooth_window}-step window)")
    print(f"{'='*70}")
    
    print(f"\nTotal Steps: {len(steps)}")
    
    print(f"\n--- Discriminator Loss ---")
    print(f"  Min: {disc_loss_smooth.min():.6f}")
    print(f"  Max: {disc_loss_smooth.max():.6f}")
    print(f"  Mean: {disc_loss_smooth.mean():.6f}")
    print(f"  Trend: {disc_loss_smooth[0]:.6f} → {disc_loss_smooth[-1]:.6f} " + 
          f"(Δ={disc_loss_smooth[-1] - disc_loss_smooth[0]:+.6f})")
    
    print(f"\n--- Generator Loss ---")
    if valid_gen.sum() > 0:
        print(f"  Min: {np.nanmin(gen_loss_smooth):.6f}")
        print(f"  Max: {np.nanmax(gen_loss_smooth):.6f}")
        print(f"  Mean: {np.nanmean(gen_loss_smooth):.6f}")
        valid_steps = steps[valid_gen]
        if len(valid_steps) > 1:
            print(f"  Trend: {gen_loss_smooth[valid_gen][0]:.6f} → {gen_loss_smooth[valid_gen][-1]:.6f} " + 
                  f"(Δ={gen_loss_smooth[valid_gen][-1] - gen_loss_smooth[valid_gen][0]:+.6f})")
    else:
        print("  No data (warmup phase)")
    
    print(f"\n--- Discriminator Scores ---")
    print(f"  Real Score: mean={real_score_smooth.mean():.4f}, min={real_score_smooth.min():.4f}, max={real_score_smooth.max():.4f}")
    print(f"    Trend: {real_score_smooth[0]:.4f} → {real_score_smooth[-1]:.4f} (goal: ~1.0)")
    print(f"  Fake Score (D): mean={fake_score_disc_smooth.mean():.4f}, min={fake_score_disc_smooth.min():.4f}, max={fake_score_disc_smooth.max():.4f}")
    print(f"    Trend: {fake_score_disc_smooth[0]:.4f} → {fake_score_disc_smooth[-1]:.4f} (goal: ~0.0)")
    print(f"  Convergence Gap: |Real - Fake| = {abs(real_score_smooth[-1] - fake_score_disc_smooth[-1]):.4f}")
    
    print(f"\n--- Generator Fooling ---")
    print(f"  Fake Score (G): mean={fake_score_gen_smooth.mean():.4f}, min={fake_score_gen_smooth.min():.4f}, max={fake_score_gen_smooth.max():.4f}")
    print(f"    Trend: {fake_score_gen_smooth[0]:.4f} → {fake_score_gen_smooth[-1]:.4f} (goal: →1.0)")
    
    print(f"\n--- Gradient Norms ---")
    print(f"  D Grad: min={disc_grad_norm_smooth.min():.6f}, max={disc_grad_norm_smooth.max():.6f}, mean={disc_grad_norm_smooth.mean():.6f}")
    print(f"  G Grad: min={np.nanmin(gen_grad_norm_smooth):.6f}, max={np.nanmax(gen_grad_norm_smooth):.6f}, mean={np.nanmean(gen_grad_norm_smooth):.6f}")
    
    # Convergence detection
    print(f"\n--- Convergence Analysis ---")
    last_10_disc = disc_loss_smooth[-10:].mean()
    first_10_disc = disc_loss_smooth[:10].mean()
    disc_improvement = (first_10_disc - last_10_disc) / first_10_disc * 100 if first_10_disc > 0 else 0
    print(f"  D Loss improvement (first 10 vs last 10): {disc_improvement:+.1f}%")
    
    if valid_gen.sum() >= 10:
        last_10_gen = np.nanmean(gen_loss_smooth[valid_gen][-10:])
        first_10_gen = np.nanmean(gen_loss_smooth[valid_gen][:10])
        gen_improvement = (first_10_gen - last_10_gen) / first_10_gen * 100 if first_10_gen > 0 else 0
        print(f"  G Loss improvement (first 10 vs last 10): {gen_improvement:+.1f}%")
    
    # Convergence check
    real_final = real_score_smooth[-1]
    fake_final = fake_score_disc_smooth[-1]
    if abs(real_final - 1.0) < 0.1 and abs(fake_final - 0.0) < 0.1:
        print(f"  ✓ CONVERGED: D learned to classify well (Real→1.0, Fake→0.0)")
    elif abs(real_final - 0.5) < 0.05 and abs(fake_final - 0.5) < 0.05:
        print(f"  ⚠ EQUILIBRIUM: Scores near 0.5 (mode collapse or equilibrium)")
    else:
        print(f"  → Training still ongoing")
    
    print(f"\n{'='*70}\n")
    
    plt.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        # Find latest qgan log
        logs_dir = Path("logs")
        qgan_logs = sorted(logs_dir.glob("qgan_*"))
        if not qgan_logs:
            print("ERROR: No qgan logs found in logs/")
            sys.exit(1)
        
        latest = qgan_logs[-1]
        csv_path = latest / "metrics.csv"
        print(f"Using latest log: {latest}")
    
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)
    
    config_path = csv_path.parent / "config.json"
    plot_qgan_training(csv_path, config_path=config_path)
