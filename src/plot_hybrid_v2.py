from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from config import MAX_EDGE_LENGTH_KM
except Exception:
    MAX_EDGE_LENGTH_KM = 5000.0


EDGE_COLS = ["e_ab", "e_bc", "e_cd", "e_da", "e_ac", "e_bd"]
EDGE_NAMES = ["ab", "bc", "cd", "da", "ac", "bd"]
REAL_EDGE_COLS = [f"real_{col}" for col in EDGE_COLS]
FAKE_EDGE_COLS = [f"fake_{col}" for col in EDGE_COLS]

# Okabe-Ito colorblind-friendly palette.
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#000000",
}


def require_columns(df: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")


def smooth(values: pd.Series, window: int) -> np.ndarray:
    values = pd.to_numeric(values, errors="coerce")
    if window <= 1:
        return values.to_numpy()
    return (
        values.rolling(window=window, min_periods=1, center=True)
        .mean()
        .to_numpy()
    )


def load_metrics(metrics_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    require_columns(df, ["step", "d_loss", "g_loss"], metrics_path)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["step"]).sort_values("step")
    return df


def select_fake_columns(df: pd.DataFrame, metrics_path: Path) -> list[str]:
    if all(col in df.columns for col in FAKE_EDGE_COLS):
        return FAKE_EDGE_COLS
    require_columns(df, EDGE_COLS, metrics_path)
    return EDGE_COLS


def load_real_samples(
    df_metrics: pd.DataFrame,
    metrics_path: Path,
    valid_tuples_path: Path,
) -> np.ndarray:
    if all(col in df_metrics.columns for col in REAL_EDGE_COLS):
        return df_metrics[REAL_EDGE_COLS].to_numpy(dtype=np.float32)

    df_real = pd.read_csv(valid_tuples_path)
    require_columns(df_real, EDGE_COLS, valid_tuples_path)
    real = df_real[EDGE_COLS].to_numpy(dtype=np.float32)
    real = np.clip(real / MAX_EDGE_LENGTH_KM, 0.0, 1.0)
    print(
        "[WARN] metrics.csv has no real_e_* columns. "
        f"Using full normalized distribution from {valid_tuples_path} instead of exact training real samples."
    )
    return real


def make_training_plot(
    df: pd.DataFrame,
    out_path: Path,
    smooth_window: int,
) -> None:
    steps = df["step"].to_numpy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(
        f"Hybrid V2 Training (smoothed window={smooth_window})",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot(steps, df["d_loss"], color=COLORS["blue"], alpha=0.25, linewidth=1, label="D raw")
    ax.plot(steps, smooth(df["d_loss"], smooth_window), color=COLORS["blue"], linewidth=2, label="D smooth")
    ax.plot(steps, df["g_loss"], color=COLORS["vermillion"], alpha=0.25, linewidth=1, label="G raw")
    ax.plot(
        steps,
        smooth(df["g_loss"], smooth_window),
        color=COLORS["vermillion"],
        linewidth=2,
        linestyle="--",
        label="G smooth",
    )
    ax.set_title("Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    score_cols = [
        ("real_score_d", "Real", COLORS["blue"], "-"),
        ("fake_score_d", "Fake for D", COLORS["orange"], "--"),
        ("fake_score_g", "Fake for G", COLORS["green"], "-."),
    ]
    for col, label, color, linestyle in score_cols:
        if col in df.columns:
            ax.plot(steps, smooth(df[col], smooth_window), color=color, linestyle=linestyle, linewidth=2, label=label)
    ax.axhline(0.5, color=COLORS["black"], linestyle=":", linewidth=1.5, alpha=0.7, label="Chance")
    ax.set_title("Discriminator Scores")
    ax.set_xlabel("Step")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 2]
    raw_cols = [
        ("real_score_d_roh", "Real logit", COLORS["blue"], "-"),
        ("fake_score_d_roh", "Fake D logit", COLORS["orange"], "--"),
        ("fake_score_g_roh", "Fake G logit", COLORS["green"], "-."),
    ]
    plotted_raw = False
    for col, label, color, linestyle in raw_cols:
        if col in df.columns:
            ax.plot(steps, smooth(df[col], smooth_window), color=color, linestyle=linestyle, linewidth=2, label=label)
            plotted_raw = True
    ax.axhline(0.0, color=COLORS["black"], linestyle=":", linewidth=1.5, alpha=0.7)
    ax.set_title("Raw Logits")
    ax.set_xlabel("Step")
    ax.set_ylabel("Logit")
    ax.grid(True, alpha=0.3)
    if plotted_raw:
        ax.legend()

    ax = axes[1, 0]
    grad_cols = [
        ("d_grad_norm", "D grad", COLORS["blue"], "-"),
        ("g_grad_norm", "G grad", COLORS["vermillion"], "--"),
    ]
    plotted_grad = False
    for col, label, color, linestyle in grad_cols:
        if col in df.columns:
            ax.plot(steps, smooth(df[col], smooth_window), color=color, linestyle=linestyle, linewidth=2, label=label)
            plotted_grad = True
    ax.set_title("Gradient Norms")
    ax.set_xlabel("Step")
    ax.set_ylabel("Norm")
    ax.grid(True, alpha=0.3)
    if plotted_grad:
        ax.legend()

    ax = axes[1, 1]
    stat_cols = [
        ("fake_mean", "Fake mean", COLORS["purple"], "-"),
        ("fake_std", "Fake std", COLORS["sky"], "--"),
    ]
    plotted_stats = False
    for col, label, color, linestyle in stat_cols:
        if col in df.columns:
            ax.plot(steps, smooth(df[col], smooth_window), color=color, linestyle=linestyle, linewidth=2, label=label)
            plotted_stats = True
    ax.set_title("Generated Sample Stats")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    if plotted_stats:
        ax.legend()

    ax = axes[1, 2]
    fake_cols = select_fake_columns(df, Path("metrics.csv"))
    for edge_name, col, color in zip(
        EDGE_NAMES,
        fake_cols,
        [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["vermillion"], COLORS["purple"], COLORS["sky"]],
    ):
        ax.plot(steps, smooth(df[col], smooth_window), color=color, linewidth=1.8, label=edge_name)
    ax.set_title("Generated Edges")
    ax.set_xlabel("Step")
    ax.set_ylabel("Normalized edge")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_classic_training_plot(
    df: pd.DataFrame,
    out_path: Path,
    smooth_window: int,
) -> None:
    require_columns(df, ["step", "d_loss", "g_loss"], out_path)
    steps = df["step"].to_numpy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(
        f"Hybrid GAN Training: QGEN + cDISC (Smoothed over {smooth_window} steps)",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot(steps, df["d_loss"], color=COLORS["blue"], alpha=0.22, linewidth=1, label="Raw")
    ax.plot(steps, smooth(df["d_loss"], smooth_window), color=COLORS["orange"], linewidth=2, label="Smoothed")
    ax.set_title("Discriminator Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(steps, df["g_loss"], color=COLORS["blue"], alpha=0.22, linewidth=1, label="Raw")
    ax.plot(steps, smooth(df["g_loss"], smooth_window), color=COLORS["orange"], linewidth=2, label="Smoothed")
    ax.set_title("Generator Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 2]
    ax.plot(steps, smooth(df["d_loss"], smooth_window), color=COLORS["blue"], linewidth=2, label="D Loss")
    ax.plot(steps, smooth(df["g_loss"], smooth_window), color=COLORS["orange"], linewidth=2, label="G Loss")
    ax.set_title("Loss Comparison (D vs G)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    if "real_score_d" in df.columns:
        ax.plot(steps, smooth(df["real_score_d"], smooth_window), color=COLORS["blue"], linewidth=2, label="Real (should->1.0)")
    if "fake_score_d" in df.columns:
        ax.plot(steps, smooth(df["fake_score_d"], smooth_window), color=COLORS["orange"], linewidth=2, label="Fake-Disc (should->0.0)")
    ax.axhline(0.5, color=COLORS["blue"], linestyle="--", linewidth=1.5, alpha=0.7, label="Chance level")
    ax.set_title("Discriminator Classification Scores")
    ax.set_xlabel("Step")
    ax.set_ylabel("D output probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    if "fake_score_g" in df.columns:
        ax.plot(steps, smooth(df["fake_score_g"], smooth_window), color=COLORS["blue"], linewidth=2, label="Fake-Gen (should->1.0)")
    ax.axhline(0.5, color=COLORS["blue"], linestyle="--", linewidth=1.5, alpha=0.7, label="Chance level")
    ax.set_title("Generator Fooling Success")
    ax.set_xlabel("Step")
    ax.set_ylabel("D output probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 2]
    plotted_grad = False
    if "d_grad_norm" in df.columns:
        ax.plot(steps, smooth(df["d_grad_norm"], smooth_window), color=COLORS["blue"], linewidth=2, label="D Grad Norm")
        plotted_grad = True
    if "g_grad_norm" in df.columns:
        ax.plot(steps, smooth(df["g_grad_norm"], smooth_window), color=COLORS["orange"], linewidth=2, label="G Grad Norm")
        plotted_grad = True
    ax.set_title("Gradient Magnitudes")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient norm")
    ax.grid(True, alpha=0.3)
    if plotted_grad:
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_per_edge_distribution_plot(
    real: np.ndarray,
    fake: np.ndarray,
    out_path: Path,
    title_suffix: str = "",
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    title = "Hybrid GAN: Per-edge distributions"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        ax.hist(
            real[:, i],
            bins=30,
            alpha=0.58,
            density=True,
            label="real",
            color=COLORS["blue"],
            edgecolor="white",
            linewidth=0.4,
        )
        ax.hist(
            fake[:, i],
            bins=30,
            alpha=0.58,
            density=True,
            label="fake",
            color=COLORS["orange"],
            edgecolor="white",
            linewidth=0.4,
            hatch="//",
        )
        ax.set_title(EDGE_NAMES[i])
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def filter_last_fraction(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("--last-fraction must be in the interval (0, 1].")
    if fraction >= 1.0:
        return df

    min_step = df["step"].max() - (df["step"].max() - df["step"].min()) * fraction
    return df[df["step"] >= min_step].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create colorblind-friendly training and per-edge distribution plots for hybrid_v2 runs."
    )
    parser.add_argument("log_dir", type=Path, help="Run directory containing metrics.csv")
    parser.add_argument("--valid-tuples", type=Path, default=Path("valid_tuples.csv"))
    parser.add_argument("--smooth-window", type=int, default=10)
    parser.add_argument(
        "--last-fraction",
        type=float,
        default=1.0,
        help="Use only the last fraction of logged training samples for the distribution plot. Example: 0.2",
    )
    args = parser.parse_args()

    metrics_path = args.log_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_path}")

    df = load_metrics(metrics_path)
    fake_cols = select_fake_columns(df, metrics_path)

    make_training_plot(
        df=df,
        out_path=args.log_dir / "training_plot.png",
        smooth_window=args.smooth_window,
    )
    make_classic_training_plot(
        df=df,
        out_path=args.log_dir / "training_plot_classic.png",
        smooth_window=args.smooth_window,
    )

    df_dist = filter_last_fraction(df, args.last_fraction)
    fake = df_dist[fake_cols].to_numpy(dtype=np.float32)
    real = load_real_samples(df_dist, metrics_path, args.valid_tuples)

    suffix = "" if args.last_fraction >= 1.0 else f"last {args.last_fraction:.0%} of training"
    out_dist = args.log_dir / "per_edge_distributions.png"
    make_per_edge_distribution_plot(real=real, fake=fake, out_path=out_dist, title_suffix=suffix)

    print(f"Saved: {args.log_dir / 'training_plot.png'}")
    print(f"Saved: {args.log_dir / 'training_plot_classic.png'}")
    print(f"Saved: {out_dist}")


if __name__ == "__main__":
    main()
