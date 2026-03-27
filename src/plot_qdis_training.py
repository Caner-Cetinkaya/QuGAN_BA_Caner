"""
Plot discriminator-only training (10k steps).

Produces:
- Loss curve
- Score evolution (mean, std, min, max)
- Gradient norm
- Weight evolution
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_qdis_training(csv_path: Path, output_dir: Optional[Path] = None):
    """Plot QDiscriminator-only training metrics."""
    
    # Load data
    rows = []
    with csv_path.open(newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: float(v) if k != 'step' else int(v) for k, v in row.items()})
    
    steps = np.array([r['step'] for r in rows])
    loss = np.array([r['loss'] for r in rows])
    score_mean = np.array([r['score_mean'] for r in rows])
    score_std = np.array([r['score_std'] for r in rows])
    score_min = np.array([r['score_min'] for r in rows])
    score_max = np.array([r['score_max'] for r in rows])
    grad_norm = np.array([r['grad_norm'] for r in rows])
    
    # Output directory
    if output_dir is None:
        output_dir = csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # ===== Plot 1: Loss Curve =====
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(steps, loss, 'b-', linewidth=2)
    ax1.fill_between(steps, loss, alpha=0.3, color='b')
    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss (10k steps)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    loss_text = f'min={loss.min():.4f}\nmean={loss.mean():.4f}\nmax={loss.max():.4f}'
    ax1.text(0.98, 0.97, loss_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ===== Plot 2: Score Mean with confidence band =====
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(steps, score_mean, 'g-', linewidth=2.5, label='Score Mean')
    ax2.fill_between(steps, score_min, score_max, alpha=0.2, color='g', label='Min/Max Range')
    ax2.fill_between(steps, score_mean - score_std, score_mean + score_std, alpha=0.3, color='g', label='±1 Std')
    ax2.axhline(0.5, color='k', linestyle='--', alpha=0.3, label='Random (0.5)')
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('Discrimination Score Evolution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # ===== Plot 3: Score Components =====
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(steps, score_mean, 'g-', linewidth=2, label='Mean', marker='o', markersize=3, markevery=100)
    ax3.plot(steps, score_min, 'r--', linewidth=1.5, label='Min', alpha=0.7)
    ax3.plot(steps, score_max, 'b--', linewidth=1.5, label='Max', alpha=0.7)
    ax3.fill_between(steps, score_min, score_max, alpha=0.1, color='gray')
    ax3.set_xlabel('Step', fontsize=11)
    ax3.set_ylabel('Score', fontsize=11)
    ax3.set_title('Score Range (Min/Mean/Max)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # ===== Plot 4: Gradient Norm =====
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(steps, grad_norm, 'purple', linewidth=2)
    ax4.fill_between(steps, 0, grad_norm, alpha=0.3, color='purple')
    ax4.set_xlabel('Step', fontsize=11)
    ax4.set_ylabel('Gradient Norm', fontsize=11)
    ax4.set_title('Parameter Gradient Norm', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    grad_text = f'min={grad_norm.min():.6f}\nmean={grad_norm.mean():.6f}\nmax={grad_norm.max():.6f}'
    ax4.text(0.98, 0.97, grad_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # ===== Plot 5: Loss (log scale) =====
    ax5 = plt.subplot(2, 3, 5)
    ax5.semilogy(steps, loss, 'r-', linewidth=2)
    ax5.set_xlabel('Step', fontsize=11)
    ax5.set_ylabel('Loss (log scale)', fontsize=11)
    ax5.set_title('Loss - Log Scale', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, which='both')
    
    # ===== Plot 6: Score std =====
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(steps, score_std, 'orange', linewidth=2)
    ax6.fill_between(steps, 0, score_std, alpha=0.3, color='orange')
    ax6.set_xlabel('Step', fontsize=11)
    ax6.set_ylabel('Standard Deviation', fontsize=11)
    ax6.set_title('Score Variability (Std Dev)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    std_text = f'min={score_std.min():.4f}\nmean={score_std.mean():.4f}\nmax={score_std.max():.4f}'
    ax6.text(0.98, 0.97, std_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # Overall title
    plt.suptitle(
        f'Quantum Discriminator Training: {csv_path.parent.name}\nTotal Steps: {len(steps)} | Loss: {loss.min():.4f} → {loss[-1]:.4f}',
        fontsize=14, fontweight='bold', y=0.995
    )
    plt.tight_layout()
    
    # Save main plot
    output_file = output_dir / 'qdis_training_plots.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Main plot saved: {output_file}")
    plt.close()
    
    # ===== Detailed Loss Plot =====
    fig_loss, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, loss, 'b-', linewidth=2.5, label='Training Loss')
    ax.fill_between(steps, loss, alpha=0.2, color='b')
    
    # Add moving average
    window = 100
    if len(steps) > window:
        loss_ma = np.convolve(loss, np.ones(window)/window, mode='valid')
        steps_ma = steps[window-1:]
        ax.plot(steps_ma, loss_ma, 'r--', linewidth=2, label=f'Moving Avg (w={window})', alpha=0.8)
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training Loss - Detail View\n(Initial: {loss[0]:.4f}, Final: {loss[-1]:.4f}, Improvement: {(loss[0]-loss[-1])/loss[0]*100:.1f}%)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_file = output_dir / 'qdis_loss_detail.png'
    plt.savefig(loss_file, dpi=150, bbox_inches='tight')
    print(f"✅ Loss detail plot saved: {loss_file}")
    plt.close(fig_loss)
    
    # ===== Detailed Score Plot =====
    fig_score, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, score_mean, 'g-', linewidth=2.5, label='Mean Score', marker='o', markersize=4, markevery=200)
    ax.fill_between(steps, score_min, score_max, alpha=0.15, color='g', label='Min/Max Range')
    ax.fill_between(steps, score_mean - score_std, score_mean + score_std, alpha=0.25, color='g', label='±1 Std Dev')
    ax.axhline(0.5, color='k', linestyle='--', linewidth=1.5, alpha=0.4, label='Random (0.5)')
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Score (Probability)', fontsize=12)
    ax.set_title(f'Discrimination Score - Detail View\n(Initial: {score_mean[0]:.4f}, Final: {score_mean[-1]:.4f})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    score_file = output_dir / 'qdis_score_detail.png'
    plt.savefig(score_file, dpi=150, bbox_inches='tight')
    print(f"✅ Score detail plot saved: {score_file}")
    plt.close(fig_score)
    
    # ===== Gradient Norm Detail =====
    fig_grad, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, grad_norm, 'purple', linewidth=2.5, label='Gradient Norm')
    ax.fill_between(steps, 0, grad_norm, alpha=0.2, color='purple')
    
    if len(steps) > 100:
        grad_ma = np.convolve(grad_norm, np.ones(50)/50, mode='valid')
        steps_ma = steps[49:]
        ax.plot(steps_ma, grad_ma, 'orange', linewidth=2, label='Moving Avg (w=50)', alpha=0.8)
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title(f'Gradient Norm Over Training\n(Mean: {grad_norm.mean():.6f}, Max: {grad_norm.max():.6f})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    grad_file = output_dir / 'qdis_grad_detail.png'
    plt.savefig(grad_file, dpi=150, bbox_inches='tight')
    print(f"✅ Gradient detail plot saved: {grad_file}")
    plt.close(fig_grad)
    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"📊 DISCRIMINATOR TRAINING STATISTICS (10,000 steps)")
    print(f"{'='*70}")
    print(f"Loss:")
    print(f"  Initial:     {loss[0]:.6f}")
    print(f"  Final:       {loss[-1]:.6f}")
    print(f"  Improvement: {(loss[0]-loss[-1])/loss[0]*100:.2f}%")
    print(f"  Mean:        {loss.mean():.6f}")
    print(f"  Std:         {loss.std():.6f}")
    print(f"\nScore (Discrimination Performance):")
    print(f"  Initial mean:  {score_mean[0]:.4f}")
    print(f"  Final mean:    {score_mean[-1]:.4f}")
    print(f"  Overall mean:  {score_mean.mean():.4f}")
    print(f"  Min range:     {score_min.min():.4f}")
    print(f"  Max range:     {score_max.max():.4f}")
    print(f"\nGradient Norm:")
    print(f"  Mean:  {grad_norm.mean():.6f}")
    print(f"  Std:   {grad_norm.std():.6f}")
    print(f"  Min:   {grad_norm.min():.6f}")
    print(f"  Max:   {grad_norm.max():.6f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_file = Path(sys.argv[1])
    else:
        # Find latest qdis metrics.csv
        logs_dir = Path("logs")
        qdis_dirs = sorted([d for d in logs_dir.iterdir() if d.name.startswith("qdis_") and d.is_dir()])
        if not qdis_dirs:
            print("No QDiscriminator runs found in logs/")
            sys.exit(1)
        csv_file = qdis_dirs[-1] / "metrics.csv"
    
    if not csv_file.exists():
        print(f"File not found: {csv_file}")
        sys.exit(1)
    
    plot_qdis_training(csv_file)
