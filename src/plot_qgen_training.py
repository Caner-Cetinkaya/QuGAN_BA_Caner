"""
Plot generator-only training (10k steps).

Produces:
- Loss curve
- Generated edge distribution vs Target
- Gradient norm evolution
- Individual edge evolution (fake_mean_0 to fake_mean_5)
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import json


def plot_qgen_training(csv_path: Path, config_path: Optional[Path] = None, output_dir: Optional[Path] = None):
    """Plot QGenerator-only training metrics."""
    
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
    
    # Handle different column names
    loss_key = 'gen_loss' if 'gen_loss' in rows[0] else 'loss'
    grad_key = 'gen_grad_norm' if 'gen_grad_norm' in rows[0] else None
    
    if loss_key not in rows[0]:
        print(f"ERROR: No loss column found. Available columns: {list(rows[0].keys())}")
        return
    
    steps = np.array([r['step'] for r in rows])
    loss = np.array([r[loss_key] for r in rows])
    grad_norm = np.array([r[grad_key] for r in rows]) if grad_key else np.ones_like(loss) * 0.01
    
    # Extract fake_mean_0 to fake_mean_5 if available
    has_fake_means = all(f'fake_mean_{i}' in rows[0] for i in range(6))
    if has_fake_means:
        fake_means = np.array([[r[f'fake_mean_{i}'] for i in range(6)] for r in rows])  # (N_steps, 6)
        print(f"Loaded fake_means with shape: {fake_means.shape}")
    else:
        print(f"WARNING: No fake_mean columns found - edge plots will be skipped")
        fake_means = None
    
    # Load config for target distribution
    target_mean = None
    target_std = None
    loss_type = "unknown"
    
    if config_path is None:
        config_path = csv_path.parent / "config.json"
    
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            target_mean = np.array(config.get("real_edge_mean", []))
            target_std = np.array(config.get("real_edge_std", []))
            loss_type = config.get("LOSS_TYPE", "unknown")
            print(f"Loaded config: loss_type={loss_type}")
    
    # Output directory
    if output_dir is None:
        output_dir = csv_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots - adapt based on available data
    if fake_means is not None:
        # 10-subplot layout (2x5 for 4 main plots, then 6 edge plots) - use 4x3 grid (12 slots)
        fig = plt.figure(figsize=(18, 14))
        n_cols = 3
        plot_rows = 4
    else:
        # Simpler layout (2x2) when no edge data
        fig = plt.figure(figsize=(12, 9))
        n_cols = 2
        plot_rows = 2
    
    # ===== Plot 1: Loss Curve =====
    ax1 = plt.subplot(plot_rows, n_cols, 1)
    ax1.plot(steps, loss, 'b-', linewidth=2, label='Generator Loss')
    ax1.fill_between(steps, loss, alpha=0.3, color='b')
    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title(f'Training Loss ({loss_type})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # ===== Plot 2: Gradient Norm =====
    ax2 = plt.subplot(plot_rows, n_cols, 2)
    ax2.plot(steps, grad_norm, 'g-', linewidth=2, label='Gradient Norm')
    ax2.fill_between(steps, grad_norm, alpha=0.3, color='g')
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Gradient Norm', fontsize=11)
    ax2.set_title('Gradient Evolution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # ===== Plot 3: Loss Distribution (histogram) =====
    ax3 = plt.subplot(plot_rows, n_cols, 3)
    ax3.hist(loss, bins=30, color='b', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Loss Value', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Loss Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ===== Plot 4 & 5-10: Edge Distribution and Evolution (only if data available) =====
    if fake_means is not None:
        ax4 = plt.subplot(plot_rows, n_cols, 4)
        if target_mean is not None and len(target_mean) == 6:
            x_pos = np.arange(6)
            width = 0.35
            final_means = fake_means[-1]  # Last step
            
            ax4.bar(x_pos - width/2, target_mean, width, label='Target (Real)', 
                   alpha=0.8, color='orange')
            ax4.bar(x_pos + width/2, final_means, width, label='Generated (Final)', 
                   alpha=0.8, color='blue')
            
            ax4.set_xlabel('Edge Index', fontsize=11)
            ax4.set_ylabel('Mean Value [0,1]', fontsize=11)
            ax4.set_title('Edge Distribution: Target vs Final', fontsize=12, fontweight='bold')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([f'e{i}' for i in range(6)])
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            ax4.text(0.5, 0.5, 'No target data', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Edge Distribution', fontsize=12, fontweight='bold')
        
        # ===== Plot 5-10: Individual edge evolution =====
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
        for i in range(6):
            subplot_idx = 7 + i  # 7, 8, 9, 10, 11, 12 - in a 4x3 grid
            ax = plt.subplot(plot_rows, n_cols, subplot_idx)
            ax.plot(steps, fake_means[:, i], color=colors[i], linewidth=2, label=f'Generated e{i}')
            
            if target_mean is not None and len(target_mean) > i:
                ax.axhline(target_mean[i], color=colors[i], linestyle='--', linewidth=2, 
                           alpha=0.7, label=f'Target e{i}')
                if target_std is not None and len(target_std) > i:
                    ax.fill_between(steps, target_mean[i] - target_std[i], 
                                   target_mean[i] + target_std[i], 
                                   alpha=0.2, color=colors[i])
            
            ax.set_xlabel('Step', fontsize=10)
            ax.set_ylabel(f'Edge {i} Value', fontsize=10)
            ax.set_title(f'Edge {i} Evolution', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / "plot_qgen_training.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {output_path}")
    
    # Statistics
    print(f"\n=== Generator Training Statistics ===")
    print(f"Total steps: {len(steps)}")
    print(f"Loss: min={loss.min():.6f}, max={loss.max():.6f}, mean={loss.mean():.6f}")
    print(f"Loss trend: start={loss[0]:.6f}, end={loss[-1]:.6f}, delta={loss[-1]-loss[0]:.6f}")
    if grad_key:
        print(f"Grad norm: min={grad_norm.min():.6f}, max={grad_norm.max():.6f}, mean={grad_norm.mean():.6f}")
    
    if target_mean is not None and fake_means is not None and len(target_mean) == 6:
        final_means = fake_means[-1]
        mse_final = np.mean((final_means - target_mean)**2)
        print(f"\nFinal Edge Distribution:")
        for i in range(6):
            print(f"  e{i}: target={target_mean[i]:.4f}, generated={final_means[i]:.4f}")
        print(f"  MSE: {mse_final:.6f}")
    
    plt.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        # Find the latest qgen log
        logs_dir = Path("logs")
        qgen_logs = sorted(logs_dir.glob("qgen_*"))
        if not qgen_logs:
            print("ERROR: No qgen logs found in logs/")
            sys.exit(1)
        
        latest = qgen_logs[-1]
        csv_path = latest / "metrics.csv"
        print(f"Using latest log: {latest}")
    
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)
    
    config_path = csv_path.parent / "config.json"
    plot_qgen_training(csv_path, config_path=config_path)
