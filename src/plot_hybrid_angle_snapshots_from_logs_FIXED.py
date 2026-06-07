"""
plot_hybrid_angle_snapshots_from_logs_FIXED.py

Erzeugt Vorher/Mitte/Nachher-Per-Edge-Plots aus bereits vorhandenen Logs,
OHNE neues Training.

Fix gegenüber der vorherigen Version:
- Für den FINAL-Step wird NICHT neu aus qgen_final.pt gesampelt,
  sondern direkt fake_samples_final.npy + real_samples_final.npy verwendet,
  falls diese Dateien vorhanden sind.
  Dadurch muss der Final-Plot mit dem im Training gespeicherten
  per_edge_distributions.png übereinstimmen, abgesehen von Titel/Dateiname.

Wichtig:
- Für Zwischenstände wie Step 1000 oder 10000 gibt es nur dann exakt dieselben
  Samples wie im Training, wenn damals auch fake_samples_step_*.npy gespeichert wurde.
  Wenn nicht, kann man den Checkpoint laden und neu sampeln, aber es sind dann
  neue Noise-Samples und damit nicht exakt dieselben Histogramme.
- Falls kein qgen_step_000001.pt existiert, kann Step 1 nicht exakt rekonstruiert
  werden. Dann wird der früheste verfügbare Checkpoint verwendet.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from training_hybrid_qgen_cdisc_classicplot import (
    EDGE_NAMES,
    QGenerator,
    load_real_data,
    sample_latent_torch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create before/middle/after per-edge plots from existing hybrid GAN logs."
    )
    parser.add_argument("--zip", type=Path, default=None)
    parser.add_argument("--root-dir", type=Path, default=None)
    parser.add_argument("--valid-tuples", type=str, required=True)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-root", type=Path, default=Path("hybrid_snapshot_plots_fixed"))
    parser.add_argument("--latent-seed", type=int, default=123)
    parser.add_argument("--real-seed", type=int, default=123)
    return parser.parse_args()


def extract_zip_if_needed(zip_path: Path, output_root: Path) -> Path:
    extract_dir = output_root / f"{zip_path.stem}_extracted"
    if extract_dir.exists():
        return extract_dir
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def discover_experiment_dirs(root_dir: Path, experiment_filter: str | None = None) -> List[Path]:
    exp_dirs: List[Path] = []
    for p in root_dir.rglob("*"):
        if not p.is_dir():
            continue
        if (p / "metrics.csv").exists() and (
            (p / "qgen_final.pt").exists()
            or (p / "fake_samples_final.npy").exists()
            or list(p.glob("qgen_step_*.pt"))
        ):
            if experiment_filter is None or experiment_filter.lower() in p.name.lower():
                exp_dirs.append(p)
    return sorted(exp_dirs)


def try_load_summary(exp_dir: Path) -> dict:
    p = exp_dir / "summary.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_steps_from_name(name: str) -> int | None:
    m = re.search(r"steps(\d+)", name)
    return int(m.group(1)) if m else None


def parse_seed_from_name(name: str) -> int | None:
    m = re.search(r"seed(\d+)", name)
    return int(m.group(1)) if m else None


def get_total_steps(exp_dir: Path, summary: dict) -> int:
    if "steps" in summary:
        return int(summary["steps"])

    parsed = parse_steps_from_name(exp_dir.name)
    if parsed is not None:
        return parsed

    steps = []
    for p in exp_dir.glob("qgen_step_*.pt"):
        m = re.search(r"qgen_step_(\d+)\.pt", p.name)
        if m:
            steps.append(int(m.group(1)))
    if steps:
        return max(steps)

    raise ValueError(f"Could not determine total steps for {exp_dir}")


def collect_qgen_checkpoints(exp_dir: Path, total_steps: int) -> Dict[int, Path]:
    ckpts: Dict[int, Path] = {}
    for p in exp_dir.glob("qgen_step_*.pt"):
        m = re.search(r"qgen_step_(\d+)\.pt", p.name)
        if m:
            ckpts[int(m.group(1))] = p

    final_path = exp_dir / "qgen_final.pt"
    if final_path.exists():
        ckpts[total_steps] = final_path

    return dict(sorted(ckpts.items()))


def choose_checkpoint_for_target(target_step: int, available: Dict[int, Path]) -> Tuple[int, Path]:
    if target_step in available:
        return target_step, available[target_step]

    all_steps = sorted(available.keys())
    if not all_steps:
        raise FileNotFoundError("No qgen checkpoints found.")

    if target_step == 1:
        chosen = all_steps[0]
    else:
        chosen = min(all_steps, key=lambda s: abs(s - target_step))

    return chosen, available[chosen]


def load_generator_from_checkpoint(ckpt_path: Path, device: torch.device) -> QGenerator:
    state = torch.load(ckpt_path, map_location=device)
    if not (isinstance(state, dict) and "weights" in state):
        raise ValueError(f"Unexpected checkpoint format: {ckpt_path}")

    n_layers = int(state["weights"].shape[0])
    gen = QGenerator(n_layer=n_layers, init_std=1.0, seed=0).to(device)
    gen.load_state_dict(state)
    gen.eval()
    return gen


def generate_fake_samples(gen: QGenerator, latent_vectors: torch.Tensor, batch_size: int) -> np.ndarray:
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, latent_vectors.shape[0], batch_size):
            z = latent_vectors[start:start + batch_size]
            outs.append(gen(z).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def make_per_edge_distribution_plot(
    real_batch: np.ndarray,
    fake_batch: np.ndarray,
    out_path: Path,
    title: str,
    force_range_01: bool = False,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    hist_kwargs = dict(bins=30, alpha=0.55, density=True)
    if force_range_01:
        hist_kwargs["range"] = (0.0, 1.0)

    for i, ax in enumerate(axes.flat):
        ax.hist(real_batch[:, i], label="real", **hist_kwargs)
        ax.hist(fake_batch[:, i], label="fake", **hist_kwargs)
        ax.set_title(EDGE_NAMES[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_combined_snapshot_plot(
    real_by_step: Dict[int, np.ndarray],
    fake_by_step: Dict[int, np.ndarray],
    labels: Dict[int, str],
    out_path: Path,
    exp_name: str,
) -> None:
    steps = list(fake_by_step.keys())
    fig, axes = plt.subplots(len(steps), 6, figsize=(22, 4.4 * len(steps)), squeeze=False)
    fig.suptitle(f"{exp_name}: Per-edge distributions before / middle / after", fontsize=18, fontweight="bold")

    for row_idx, step in enumerate(steps):
        real_batch = real_by_step[step]
        fake_batch = fake_by_step[step]

        for col_idx, edge in enumerate(EDGE_NAMES):
            ax = axes[row_idx, col_idx]
            ax.hist(real_batch[:, col_idx], bins=30, alpha=0.50, density=True, label="real")
            ax.hist(fake_batch[:, col_idx], bins=30, alpha=0.50, density=True, label="fake")
            ax.set_title(f"{edge}\n{labels[step]}")
            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_metrics_overview_plot(metrics_csv: Path, out_path: Path, exp_name: str) -> None:
    df = pd.read_csv(metrics_csv)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["step"]).sort_values("step")

    def smooth(series, window=15):
        return pd.Series(series).rolling(window=window, min_periods=1, center=True).mean().to_numpy()

    steps = df["step"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{exp_name}: Training overview", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(steps, smooth(df["d_loss"]), label="d_loss")
    ax.plot(steps, smooth(df["g_loss"]), label="g_loss")
    ax.set_title("Losses")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    for col in ["real_score", "fake_score_d", "fake_score_g", "fake_score"]:
        if col in df.columns:
            ax.plot(steps, smooth(df[col]), label=col)
    ax.axhline(0.5, linestyle="--", alpha=0.6)
    ax.set_title("Scores")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    if "fake_mean" in df.columns:
        ax.plot(steps, smooth(df["fake_mean"]), label="fake_mean")
    if "fake_std" in df.columns:
        ax.plot(steps, smooth(df["fake_std"]), label="fake_std")
    ax.set_title("Global fake stats")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    for edge in EDGE_NAMES:
        col = f"{edge}_mean"
        if col in df.columns:
            ax.plot(steps, smooth(df[col]), label=edge)
    ax.set_title("Per-edge means")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def sample_real_reference(valid_tuples: str, n: int, seed: int) -> np.ndarray:
    real_data = load_real_data(valid_tuples)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(real_data), size=n, replace=True)
    return real_data[idx]


def process_experiment(exp_dir: Path, args: argparse.Namespace, out_root: Path) -> None:
    print(f"\n[INFO] Processing {exp_dir}")

    summary = try_load_summary(exp_dir)
    total_steps = get_total_steps(exp_dir, summary)
    latent_distribution = str(summary.get("latent_distribution", "uniform"))
    experiment_seed = parse_seed_from_name(exp_dir.name)
    latent_seed = args.latent_seed if args.latent_seed is not None else (experiment_seed or 0)

    target_steps = [1, max(1, total_steps // 2), total_steps]
    available = collect_qgen_checkpoints(exp_dir, total_steps)

    out_dir = out_root / exp_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # If final real samples exist, use them. This is necessary to reproduce the final training plot.
    final_real_path = exp_dir / "real_samples_final.npy"
    final_fake_path = exp_dir / "fake_samples_final.npy"

    final_real_samples = np.load(final_real_path) if final_real_path.exists() else None
    final_fake_samples = np.load(final_fake_path) if final_fake_path.exists() else None

    # For non-final generated snapshots use one fixed real reference.
    real_reference = (
        final_real_samples
        if final_real_samples is not None
        else sample_real_reference(args.valid_tuples, args.eval_samples, args.real_seed)
    )

    device = torch.device(args.device)
    torch.manual_seed(latent_seed)
    fixed_latent = sample_latent_torch(args.eval_samples, args.latent_dim, latent_distribution, device)

    real_by_step: Dict[int, np.ndarray] = {}
    fake_by_step: Dict[int, np.ndarray] = {}
    labels: Dict[int, str] = {}
    manifest = []

    for target in target_steps:
        if target == total_steps and final_fake_samples is not None:
            # Critical fix: reproduce the training-final plot from saved samples.
            real_batch = final_real_samples if final_real_samples is not None else real_reference
            fake_batch = final_fake_samples
            used_step = total_steps
            source = "saved fake_samples_final.npy"
            label = f"Snapshot after {used_step} steps"
        else:
            if not available:
                print(f"[WARN] No checkpoint available for {exp_dir}; skipping target {target}")
                continue

            used_step, ckpt = choose_checkpoint_for_target(target, available)
            gen = load_generator_from_checkpoint(ckpt, device)
            fake_batch = generate_fake_samples(gen, fixed_latent, args.batch_size)
            real_batch = real_reference
            source = str(ckpt)
            if used_step == target:
                label = f"Snapshot after {used_step} steps"
            else:
                label = f"Snapshot after {used_step} steps"

        real_by_step[target] = real_batch
        fake_by_step[target] = fake_batch
        labels[target] = label

        manifest.append(
            {
                "target_step": target,
                "used_step": used_step,
                "source": source,
                "label": label,
                "real_samples_source": str(final_real_path) if final_real_path.exists() else "sampled from valid_tuples.csv",
            }
        )

        png_name = f"per_edge_distributions_target_{target:06d}_used_{used_step:06d}.png"
        make_per_edge_distribution_plot(
            real_batch=real_batch,
            fake_batch=fake_batch,
            out_path=out_dir / png_name,
            title=f"{exp_dir.name}: per-edge distributions ({label})",
            force_range_01=False,
        )
        print(f"[INFO] Saved {png_name}")

        # If this is final and saved samples were used, also write an exact-style copy.
        if target == total_steps and final_fake_samples is not None:
            make_per_edge_distribution_plot(
                real_batch=real_batch,
                fake_batch=fake_batch,
                out_path=out_dir / "per_edge_distributions_reproduced_from_saved_npy.png",
                title="Hybrid GAN: Per-edge distributions",
                force_range_01=False,
            )
            print("[INFO] Saved per_edge_distributions_reproduced_from_saved_npy.png")

    if fake_by_step:
        make_combined_snapshot_plot(
            real_by_step=real_by_step,
            fake_by_step=fake_by_step,
            labels=labels,
            out_path=out_dir / "per_edge_distributions_before_middle_after.png",
            exp_name=exp_dir.name,
        )
        print("[INFO] Saved combined before/middle/after plot")

    metrics_csv = exp_dir / "metrics.csv"
    if metrics_csv.exists():
        make_metrics_overview_plot(metrics_csv, out_dir / "metrics_overview_from_logs.png", exp_dir.name)

    (out_dir / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.zip is None and args.root_dir is None:
        raise ValueError("Provide either --zip or --root-dir.")

    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.zip is not None:
        root = extract_zip_if_needed(args.zip, args.output_root)
    else:
        root = args.root_dir

    exp_dirs = discover_experiment_dirs(root, args.experiment)
    if not exp_dirs:
        raise FileNotFoundError(
            f"No experiment folders found under {root}. "
            "Try --root-dir .\\logs or check the folder name."
        )

    print(f"[INFO] Found {len(exp_dirs)} experiment(s).")

    for exp_dir in exp_dirs:
        process_experiment(exp_dir, args, args.output_root)

    print("\n[INFO] Done.")
    print(f"[INFO] Output root: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
