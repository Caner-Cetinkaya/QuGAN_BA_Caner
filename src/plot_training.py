"""
Plot training metrics from QuGAN run.

Produces:
- Loss curve (Discriminator & Generator)
- Score evolution (Real vs Fake)
- Separation trend
- Gradient norms
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_training(csv_path: Path, output_dir: Optional[Path] = None):
    """Plot training metrics from metrics.csv."""
    
    # Load data
    rows = []
    with csv_path.open(newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: float(v) if k != 'step' else int(v) for k, v in row.items()})
    
    steps = np.array([r['step'] for r in rows])
    
    # Extract metrics
    disc_loss = np.array([r['disc_loss'] for r in rows])
    gen_loss = np.array([r['gen_loss'] for r in rows])
    
    real_scores = np.array([r['real_score_mean'] for r in rows])
    fake_scores_disc = np.array([r['fake_score_mean_disc'] for r in rows])
    fake_scores_gen = np.array([r['fake_score_mean_gen'] for r in rows])
    
    disc_grad = np.array([r['disc_grad_norm'] for r in rows])
    gen_grad = np.array([r['gen_grad_norm'] for r in rows])
    
    separation = real_scores - fake_scores_disc
    
    # Output directory
    if output_dir is None:
        output_dir = csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # ===== Plot 1: Loss Curves =====
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(steps, disc_loss, 'b-', label='Discriminator Loss', linewidth=2)
    ax1_twin = ax1.twinx()
    
    # Only plot gen_loss where not NaN (after warmup)
    gen_loss_valid = np.where(gen_loss == gen_loss, gen_loss, np.nan)
    ax1_twin.plot(steps, gen_loss_valid, 'r-', label='Generator Loss', linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Discriminator Loss', color='b', fontsize=11)
    ax1_twin.set_ylabel('Generator Loss', color='r', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.set_title('Loss Curves', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # ===== Plot 2: Score Evolution =====
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(steps, real_scores, 'g-', label='Real Score (D)', linewidth=2)
    ax2.plot(steps, fake_scores_disc, 'r-', label='Fake Score (D)', linewidth=2)
    ax2.plot(steps, fake_scores_gen, 'orange', label='Fake Score (G)', linewidth=2, alpha=0.7)
    ax2.axhline(0.5, color='k', linestyle='--', alpha=0.3, label='Random (0.5)')
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Score (Probability)', fontsize=11)
    ax2.set_title('Score Evolution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # ===== Plot 3: Separation Trend =====
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(steps, separation, 'purple', linewidth=2, label='Real - Fake (D)')
    ax3.fill_between(steps, 0, separation, where=(separation >= 0), alpha=0.3, color='green', label='D correctly separates')
    ax3.fill_between(steps, 0, separation, where=(separation < 0), alpha=0.3, color='red', label='D confused')
    ax3.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Step', fontsize=11)
    ax3.set_ylabel('Separation (Real - Fake)', fontsize=11)
    ax3.set_title('Discrimination Separation', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ===== Plot 4: Disc Grad Norm =====
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(steps, disc_grad, 'b-', linewidth=2)
    ax4.fill_between(steps, 0, disc_grad, alpha=0.3, color='b')
    ax4.set_xlabel('Step', fontsize=11)
    ax4.set_ylabel('Gradient Norm', fontsize=11)
    ax4.set_title('Discriminator Gradient Norm', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # ===== Plot 5: Gen Grad Norm =====
    ax5 = plt.subplot(2, 3, 5)
    gen_grad_valid = np.where(gen_grad == gen_grad, gen_grad, np.nan)
    ax5.plot(steps, gen_grad_valid, 'r-', linewidth=2)
    ax5.fill_between(steps, 0, gen_grad_valid, alpha=0.3, color='r')
    ax5.set_xlabel('Step', fontsize=11)
    ax5.set_ylabel('Gradient Norm', fontsize=11)
    ax5.set_title('Generator Gradient Norm', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # ===== Plot 6: Score Statistics (moving average) =====
    ax6 = plt.subplot(2, 3, 6)
    window = 50
    if len(steps) > window:
        real_ma = np.convolve(real_scores, np.ones(window)/window, mode='valid')
        fake_ma = np.convolve(fake_scores_disc, np.ones(window)/window, mode='valid')
        steps_ma = steps[window-1:]
        
        ax6.plot(steps_ma, real_ma, 'g-', linewidth=2.5, label=f'Real (MA-{window})')
        ax6.plot(steps_ma, fake_ma, 'r-', linewidth=2.5, label=f'Fake (MA-{window})')
        ax6.fill_between(steps_ma, fake_ma, real_ma, where=(real_ma >= fake_ma), alpha=0.3, color='green')
    
    ax6.set_xlabel('Step', fontsize=11)
    ax6.set_ylabel('Score (Moving Average)', fontsize=11)
    ax6.set_title(f'Scores Moving Average (window={window})', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1])
    
    # Overall title
    plt.suptitle(
        f'QuGAN Training: {csv_path.parent.name}\nTotal Steps: {len(steps)}',
        fontsize=14, fontweight='bold', y=0.995
    )
    plt.tight_layout()
    
    # Save
    output_file = output_dir / 'training_plots.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_file}")
    
    # Also save individual plots for detail
    # Loss only
    fig_loss, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, disc_loss, 'b-', linewidth=2.5, label='Discriminator Loss')
    ax_twin = ax.twinx()
    ax_twin.plot(steps, gen_loss_valid, 'r-', linewidth=2.5, label='Generator Loss', alpha=0.8)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Discriminator Loss', color='b', fontsize=12)
    ax_twin.set_ylabel('Generator Loss', color='r', fontsize=12)
    ax.set_title('Loss Curves - Detail View', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='y', labelcolor='b')
    ax_twin.tick_params(axis='y', labelcolor='r')
    plt.tight_layout()
    loss_file = output_dir / 'loss_detail.png'
    plt.savefig(loss_file, dpi=150, bbox_inches='tight')
    print(f"✅ Loss detail plot saved to: {loss_file}")
    plt.close(fig_loss)
    
    # Scores only
    fig_scores, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, real_scores, 'g-', linewidth=2.5, label='Real Score')
    ax.plot(steps, fake_scores_disc, 'r-', linewidth=2.5, label='Fake Score (D)')
    ax.plot(steps, fake_scores_gen, 'orange', linewidth=2.5, label='Fake Score (G)', alpha=0.8)
    ax.axhline(0.5, color='k', linestyle='--', alpha=0.3, label='Random (0.5)')
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Score (Probability)', fontsize=12)
    ax.set_title('Score Evolution - Detail View', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    scores_file = output_dir / 'scores_detail.png'
    plt.savefig(scores_file, dpi=150, bbox_inches='tight')
    print(f"✅ Scores detail plot saved to: {scores_file}")
    plt.close(fig_scores)
    
    # Separation only
    fig_sep, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, separation, 'purple', linewidth=2.5, label='Separation (Real - Fake)')
    ax.fill_between(steps, 0, separation, where=(separation >= 0), alpha=0.4, color='green', label='D wins')
    ax.fill_between(steps, 0, separation, where=(separation < 0), alpha=0.4, color='red', label='D confused')
    ax.axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Separation Score', fontsize=12)
    ax.set_title('Real vs Fake Separation - Detail View', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    sep_file = output_dir / 'separation_detail.png'
    plt.savefig(sep_file, dpi=150, bbox_inches='tight')
    print(f"✅ Separation detail plot saved to: {sep_file}")
    plt.close(fig_sep)
    
    plt.show()
    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"📊 TRAINING STATISTICS")
    print(f"{'='*70}")
    print(f"Discriminator Loss:  min={disc_loss.min():.4f}, mean={disc_loss.mean():.4f}, max={disc_loss.max():.4f}")
    print(f"Real Score:          min={real_scores.min():.4f}, mean={real_scores.mean():.4f}, max={real_scores.max():.4f}")
    print(f"Fake Score (D):      min={fake_scores_disc.min():.4f}, mean={fake_scores_disc.mean():.4f}, max={fake_scores_disc.max():.4f}")
    print(f"Separation:          min={separation.min():.4f}, mean={separation.mean():.4f}, max={separation.max():.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_file = Path(sys.argv[1])
    else:
        # Find latest qgan metrics.csv
        logs_dir = Path("logs")
        qgan_dirs = sorted([d for d in logs_dir.iterdir() if d.name.startswith("qgan_") and d.is_dir()])
        if not qgan_dirs:
            print("No QGAN runs found in logs/")
            sys.exit(1)
        csv_file = qgan_dirs[-1] / "metrics.csv"
    
    if not csv_file.exists():
        print(f"File not found: {csv_file}")
        sys.exit(1)
    
    plot_training(csv_file)
