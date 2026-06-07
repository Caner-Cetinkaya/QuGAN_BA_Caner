"""
Korrelations-Analyse: Hat der Generator die Korrelationsstruktur
zwischen den Kanten gelernt, oder nur die Marginalen?

Laedt fake_samples_final.npy und real_samples_final.npy aus einem Run und:
1. Berechnet 6x6 Korrelationsmatrizen fuer Real und Fake
2. Plottet beide als Heatmap + Differenz
3. Berechnet Frobenius-Distanz als skalare Zusammenfassung
4. Berechnet Wasserstein-Distanz pro Kante (zusaetzlich)
5. Berechnet Anteil gueltiger Dreiecksungleichungen (falls Daten im Distanz-Bereich)

Aufruf:
  python eval_correlations.py --run-dir logs/exp2_probs_qgen_cdisc_20260524_185934
  python eval_correlations.py --run-dir logs/exp1_mbd_qgen_cdisc_20260524_184503

Optional fuer mehrere Runs gleichzeitig:
  python eval_correlations.py --run-dir runA --run-dir runB --run-dir runC
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance


EDGE_NAMES = ["ab", "bc", "cd", "da", "ac", "bd"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correlation analysis for QGAN runs.")
    parser.add_argument("--run-dir", type=Path, action="append", required=True,
                        help="Run-Verzeichnis mit real_samples_final.npy + fake_samples_final.npy "
                             "(kann mehrfach angegeben werden fuer Vergleichs-Plots)")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Toleranz fuer Triangle-Inequality-Check")
    return parser.parse_args()


def correlation_matrix(samples: np.ndarray) -> np.ndarray:
    """Returns 6x6 correlation matrix (Pearson)."""
    return np.corrcoef(samples.T)


def frobenius_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Frobenius-Norm der Differenz zweier Matrizen."""
    return float(np.linalg.norm(a - b))


def triangles_valid_fraction(tuples_6d: np.ndarray, tol: float = 1e-6) -> float:
    """
    Anteil der Tupel, bei denen alle 4 Dreiecke die Dreiecksungleichung erfuellen.
    Edge-Reihenfolge: ab, bc, cd, da, ac, bd
    Dreiecke: abc, abd, acd, bcd
    """
    ab = tuples_6d[:, 0]; bc = tuples_6d[:, 1]; cd = tuples_6d[:, 2]
    da = tuples_6d[:, 3]; ac = tuples_6d[:, 4]; bd = tuples_6d[:, 5]

    triangles = [
        (ab, bc, ac),  # abc: ab + bc >= ac, etc.
        (ab, bd, da),  # abd
        (ac, cd, da),  # acd
        (bc, cd, bd),  # bcd
    ]
    ok = np.ones(len(tuples_6d), dtype=bool)
    for x, y, z in triangles:
        ok &= (x + y + tol >= z) & (x + z + tol >= y) & (y + z + tol >= x)
    return float(ok.mean())


def plot_correlation_heatmaps(corr_real: np.ndarray, corr_fake: np.ndarray,
                              out_path: Path, title: str) -> None:
    """3 Heatmaps nebeneinander: Real, Fake, Differenz."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Symmetrische Farbskala fuer Real und Fake
    vmax = max(np.abs(corr_real).max(), np.abs(corr_fake).max())

    for ax, mat, name in zip(axes[:2], [corr_real, corr_fake], ["Real", "Fake"]):
        im = ax.imshow(mat, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(EDGE_NAMES); ax.set_yticklabels(EDGE_NAMES)
        ax.set_title(f"{name} correlations")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if abs(mat[i, j]) > 0.5 else "black",
                        fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Differenz
    diff = corr_real - corr_fake
    dvmax = max(np.abs(diff).max(), 0.01)
    ax = axes[2]
    im = ax.imshow(diff, vmin=-dvmax, vmax=dvmax, cmap="PiYG")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(EDGE_NAMES); ax.set_yticklabels(EDGE_NAMES)
    ax.set_title(f"Real - Fake (Frobenius={frobenius_diff(corr_real, corr_fake):.3f})")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{diff[i, j]:+.2f}", ha="center", va="center",
                    color="white" if abs(diff[i, j]) > dvmax * 0.5 else "black",
                    fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def evaluate_run(run_dir: Path, tol: float) -> dict:
    """Werte einen Run aus, gib Dict mit Metriken zurueck."""
    real_path = run_dir / "real_samples_final.npy"
    fake_path = run_dir / "fake_samples_final.npy"

    if not real_path.exists() or not fake_path.exists():
        print(f"[WARN] {run_dir.name}: missing npy files, skipping")
        return {}

    real = np.load(real_path)
    fake = np.load(fake_path)
    print(f"\n=== {run_dir.name} ===")
    print(f"Loaded: real {real.shape}, fake {fake.shape}")

    # Korrelationsmatrizen
    corr_real = correlation_matrix(real)
    corr_fake = correlation_matrix(fake)
    frob = frobenius_diff(corr_real, corr_fake)

    # Pro-Element-Differenz
    off_diag_mask = ~np.eye(6, dtype=bool)
    mean_abs_off_diag_diff = float(np.abs(corr_real - corr_fake)[off_diag_mask].mean())

    # Plot
    plot_correlation_heatmaps(
        corr_real, corr_fake,
        run_dir / "correlation_heatmap.png",
        title=f"Correlation analysis: {run_dir.name}"
    )

    # Wasserstein pro Kante (zusaetzlicher Kontext)
    wasserstein_per_edge = {
        EDGE_NAMES[i]: float(wasserstein_distance(real[:, i], fake[:, i]))
        for i in range(6)
    }
    mean_w = float(np.mean(list(wasserstein_per_edge.values())))

    # Triangle inequality
    # Warnung: Bei Simplex-Daten (EXP2) ergibt der absolute Tol-Wert wenig Sinn,
    # weil alle Werte sehr klein sind. Wir berichten den Wert trotzdem.
    real_tri = triangles_valid_fraction(real, tol=tol)
    fake_tri = triangles_valid_fraction(fake, tol=tol)

    # Statistik Real vs Fake fuer Sanity-Check
    print(f"Real:  mean={real.mean():.4f}, std={real.std():.4f}, "
          f"min={real.min():.4f}, max={real.max():.4f}")
    print(f"Fake:  mean={fake.mean():.4f}, std={fake.std():.4f}, "
          f"min={fake.min():.4f}, max={fake.max():.4f}")
    print(f"\nCorrelation matrix REAL:")
    for i in range(6):
        print(f"  {EDGE_NAMES[i]}: " + "  ".join(f"{corr_real[i, j]:+.3f}" for j in range(6)))
    print(f"\nCorrelation matrix FAKE:")
    for i in range(6):
        print(f"  {EDGE_NAMES[i]}: " + "  ".join(f"{corr_fake[i, j]:+.3f}" for j in range(6)))
    print(f"\nFrobenius distance: {frob:.4f}")
    print(f"Mean abs off-diagonal diff: {mean_abs_off_diag_diff:.4f}")
    print(f"Mean Wasserstein per edge: {mean_w:.4f}")
    print(f"Triangle inequality valid (tol={tol}):")
    print(f"  Real: {real_tri*100:.1f}% | Fake: {fake_tri*100:.1f}%")

    # Save numeric summary
    metrics = {
        "run_dir": run_dir.name,
        "frobenius_distance_correlations": frob,
        "mean_abs_off_diagonal_diff": mean_abs_off_diag_diff,
        "mean_wasserstein_per_edge": mean_w,
        "wasserstein_per_edge": wasserstein_per_edge,
        "real_triangle_inequality_valid_fraction": real_tri,
        "fake_triangle_inequality_valid_fraction": fake_tri,
        "real_stats": {
            "mean": float(real.mean()), "std": float(real.std()),
            "min": float(real.min()), "max": float(real.max())
        },
        "fake_stats": {
            "mean": float(fake.mean()), "std": float(fake.std()),
            "min": float(fake.min()), "max": float(fake.max())
        },
        "correlation_matrix_real": corr_real.tolist(),
        "correlation_matrix_fake": corr_fake.tolist(),
    }
    with open(run_dir / "correlation_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def plot_comparison(all_metrics: List[dict], out_path: Path) -> None:
    """Vergleichsplot fuer mehrere Runs: Frobenius, mean-W, triangle-fraction."""
    if len(all_metrics) < 2:
        return

    names = [m["run_dir"] for m in all_metrics]
    frobs = [m["frobenius_distance_correlations"] for m in all_metrics]
    ws = [m["mean_wasserstein_per_edge"] for m in all_metrics]
    fake_tri = [m["fake_triangle_inequality_valid_fraction"] * 100 for m in all_metrics]
    real_tri = [m["real_triangle_inequality_valid_fraction"] * 100 for m in all_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    bars = ax.bar(range(len(names)), frobs, color="steelblue")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Frobenius distance")
    ax.set_title("Correlation matrix Frobenius distance\n(lower = correlations better matched)")
    ax.grid(True, alpha=0.3, axis="y")
    for b, v in zip(bars, frobs):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")

    ax = axes[1]
    bars = ax.bar(range(len(names)), ws, color="darkorange")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Mean Wasserstein distance per edge")
    ax.set_title("Marginal distribution quality\n(lower = marginals better matched)")
    ax.grid(True, alpha=0.3, axis="y")
    for b, v in zip(bars, ws):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")

    ax = axes[2]
    width = 0.35
    x = np.arange(len(names))
    ax.bar(x - width/2, real_tri, width, label="Real", color="steelblue")
    ax.bar(x + width/2, fake_tri, width, label="Fake", color="lightcoral")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("% valid triangles")
    ax.set_title("Triangle inequality satisfied")
    ax.set_ylim(0, 105); ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    all_metrics = []
    for run_dir in args.run_dir:
        if not run_dir.exists():
            print(f"[ERROR] Run dir not found: {run_dir}")
            continue
        m = evaluate_run(run_dir, tol=args.tol)
        if m:
            all_metrics.append(m)

    if len(all_metrics) >= 2:
        # Vergleichsplot in den ersten Run-Ordner legen (oder ins cwd)
        out_path = args.run_dir[0].parent / "comparison_correlations.png"
        plot_comparison(all_metrics, out_path)
        print(f"\n[INFO] Saved comparison plot: {out_path}")

    print(f"\n[INFO] Done. correlation_heatmap.png + correlation_summary.json "
          f"per run directory.")


if __name__ == "__main__":
    main()
