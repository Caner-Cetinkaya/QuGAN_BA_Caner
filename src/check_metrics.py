"""
Quick metrics checker for QuGAN training runs.

3 Key Checks:
1. Disc learning? (real_score > fake_score, both not ~0.5)
2. Gen improving? (fake_score_gen should rise)
3. Grad norm explosions? (should stay stable)
"""

import csv
import statistics as stats
from pathlib import Path
from typing import Optional
import sys


def check_latest_qgan_run():
    """Find and analyze the latest QGAN run."""
    logs_dir = Path("logs")
    qgan_dirs = sorted([d for d in logs_dir.iterdir() if d.name.startswith("qgan_") and d.is_dir()])
    
    if not qgan_dirs:
        print("No QGAN runs found in logs/")
        return
    
    latest_dir = qgan_dirs[-1]
    metrics_file = latest_dir / "metrics.csv"
    
    if not metrics_file.exists():
        print(f"No metrics.csv in {latest_dir.name}")
        return
    
    analyze_metrics(metrics_file, latest_dir.name)


def analyze_metrics(csv_path: Path, run_name: str = ""):
    """Analyze metrics CSV for the 3 key checks."""
    rows = []
    with csv_path.open(newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: float(v) if k != 'step' else int(v) for k, v in row.items()})
    
    if not rows:
        print("No data in metrics CSV")
        return
    
    n = len(rows)
    print(f"\n{'='*70}")
    print(f"QGAN Run Analysis: {run_name}")
    print(f"Total steps: {n} / {rows[-1]['step']}")
    print(f"{'='*70}")
    
    # ===== CHECK 1: Discriminator Learning =====
    print(f"\n1️⃣  DISCRIMINATOR LEARNING?")
    real_scores = [r['real_score_mean'] for r in rows]
    fake_scores_disc = [r['fake_score_mean_disc'] for r in rows]
    sep = [r - f for r, f in zip(real_scores, fake_scores_disc)]
    
    print(f"   Real score:  mean={stats.fmean(real_scores):.4f}, std={stats.pstdev(real_scores) if n>1 else 0:.4f}")
    print(f"   Fake score:  mean={stats.fmean(fake_scores_disc):.4f}, std={stats.pstdev(fake_scores_disc) if n>1 else 0:.4f}")
    print(f"   Separation:  mean={stats.fmean(sep):.4f}, std={stats.pstdev(sep) if n>1 else 0:.4f}")
    print(f"   → Real > Fake? {stats.fmean(real_scores) > stats.fmean(fake_scores_disc)}")
    if stats.fmean(sep) > 0.05:
        print(f"   ✓ GOOD: Clear separation (sep={stats.fmean(sep):.4f})")
    elif stats.fmean(sep) > 0.01:
        print(f"   ⚠️  WEAK: Small separation (sep={stats.fmean(sep):.4f})")
    else:
        print(f"   ✗ BAD: No separation (sep={stats.fmean(sep):.4f})")
    
    # ===== CHECK 2: Generator Improving =====
    print(f"\n2️⃣  GENERATOR IMPROVING?")
    fake_scores_gen = [r['fake_score_mean_gen'] for r in rows]
    disc_loss = [r['disc_loss'] for r in rows]
    gen_loss = [r['gen_loss'] for r in rows if not (isinstance(r['gen_loss'], float) and r['gen_loss'] != r['gen_loss'])]  # skip NaN
    
    print(f"   Fake(Gen) score: mean={stats.fmean(fake_scores_gen):.4f}, std={stats.pstdev(fake_scores_gen) if n>1 else 0:.4f}")
    if gen_loss:
        gen_loss_valid = [x for x in gen_loss if x == x]  # remove NaN
        if gen_loss_valid:
            print(f"   Gen loss:        mean={stats.fmean(gen_loss_valid):.4f}")
    
    # Trend: early vs late fake_gen scores
    early_fake_gen = stats.fmean(fake_scores_gen[:min(10, n//2)])
    late_fake_gen = stats.fmean(fake_scores_gen[-min(10, n//2):])
    trend = late_fake_gen - early_fake_gen
    print(f"   Early fake(gen): {early_fake_gen:.4f}, Late: {late_fake_gen:.4f}, Trend: {trend:+.4f}")
    
    if trend > 0.02:
        print(f"   ✓ GOOD: Gen improving (pushing fake scores up)")
    elif trend > -0.02:
        print(f"   ~ OK: Gen stable")
    else:
        print(f"   ⚠️  Possible issue: Gen not improving or degrading")
    
    # ===== CHECK 3: Gradient Norms =====
    print(f"\n3️⃣  GRADIENT STABILITY?")
    disc_grad = [r['disc_grad_norm'] for r in rows]
    gen_grad = [r['gen_grad_norm'] for r in rows]
    
    # Filter out NaN from gen_grad (warmup phase)
    gen_grad_valid = [x for x in gen_grad if x == x]
    
    print(f"   Disc grad norm:  mean={stats.fmean(disc_grad):.6f}, min={min(disc_grad):.6f}, max={max(disc_grad):.6f}")
    if gen_grad_valid:
        print(f"   Gen grad norm:   mean={stats.fmean(gen_grad_valid):.6f}, min={min(gen_grad_valid):.6f}, max={max(gen_grad_valid):.6f}")
    
    # Check for explosions
    disc_grad_max = max(disc_grad)
    disc_grad_mean = stats.fmean(disc_grad)
    if disc_grad_max > disc_grad_mean * 5:
        print(f"   ⚠️  Disc grad spikes: max={disc_grad_max:.6f} >> mean={disc_grad_mean:.6f}")
    else:
        print(f"   ✓ Disc grad stable")
    
    if gen_grad_valid:
        gen_grad_max = max(gen_grad_valid)
        gen_grad_mean = stats.fmean(gen_grad_valid)
        if gen_grad_max > gen_grad_mean * 5:
            print(f"   ⚠️  Gen grad spikes: max={gen_grad_max:.6f} >> mean={gen_grad_mean:.6f}")
        else:
            print(f"   ✓ Gen grad stable")
    
    # ===== SUMMARY =====
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY (Last 5 steps)")
    print(f"{'='*70}")
    for row in rows[-5:]:
        step = int(row['step'])
        r = row['real_score_mean']
        f = row['fake_score_mean_disc']
        s = r - f
        dloss = row['disc_loss']
        gloss = row['gen_loss'] if row['gen_loss'] == row['gen_loss'] else float('nan')
        gloss_str = f"{gloss:.4f}" if gloss == gloss else "nan"
        print(f"Step {step:5d}: real={r:.4f} fake={f:.4f} sep={s:+.4f}  dloss={dloss:.4f}  gloss={gloss_str}")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = Path(sys.argv[1])
        if csv_file.exists():
            analyze_metrics(csv_file, csv_file.parent.name)
        else:
            print(f"File not found: {csv_file}")
    else:
        check_latest_qgan_run()
