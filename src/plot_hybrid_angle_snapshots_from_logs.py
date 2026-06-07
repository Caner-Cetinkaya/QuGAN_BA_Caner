"""
plot_hybrid_angle_snapshots_from_logs.py

Erzeugt die "vorher / mitte / nachher" Per-Edge-Distribution-Plots
AUS BEREITS GELAUFENEN EXPERIMENTEN / CHECKPOINTS — ohne neues Training.

Unterstützt:
- einen entpackten Experiment-Ordner
- oder direkt eine ZIP-Datei wie hybrid_angle_experiments.zip

Wichtiger Hinweis:
Wenn kein qgen_step_000001.pt existiert, kann Step 1 NICHT exakt rekonstruiert werden.
Dann nimmt das Skript automatisch den frühesten verfügbaren Checkpoint
(z. B. Step 500) und schreibt das auch in den Plot.

Beispiel:
    python .\plot_hybrid_angle_snapshots_from_logs.py ^
        --zip .\hybrid_angle_experiments.zip ^
        --valid-tuples .\valid_tuples.csv

Oder für nur einen bestimmten Run:
    python .\plot_hybrid_angle_snapshots_from_logs.py ^
        --zip .\hybrid_angle_experiments.zip ^
        --valid-tuples .\valid_tuples.csv ^
        --experiment exp02_angle_seed1_bs10_steps2000
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
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
        description="Create hybrid GAN snapshot plots from existing experiment logs/checkpoints."
    )
    parser.add_argument("--zip", type=Path, default=None, help="ZIP archive with experiment folders")
    parser.add_argument("--root-dir", type=Path, default=None, help="Already extracted root directory")
    parser.add_argument("--valid-tuples", type=str, required=True, help="Path to valid_tuples.csv")
    parser.add_argument("--experiment", type=str, default=None, help="Only process experiment folder containing this text")
    parser.add_argument("--eval-samples", type=int, default=1000, help="How many fake samples to generate per snapshot")
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size only for re-sampling the generator during plotting")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-root", type=Path, default=Path("hybrid_snapshot_plots"))
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
    exp_dirs = []
    for p in root_dir.rglob("*"):
        if not p.is_dir():
            continue
        if (p / "metrics.csv").exists() and ((p / "qgen_final.pt").exists() or list(p.glob("qgen_step_*.pt"))):
            if experiment_filter is None or experiment_filter.lower() in p.name.lower():
                exp_dirs.append(p)
    exp_dirs.sort()
    return exp_dirs


def try_load_summary(exp_dir: Path) -> dict:
    summary_path = exp_dir / "summary.json"
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def parse_seed_from_name(name: str) -> int | None:
    m = re.search(r"seed(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def parse_steps_from_name(name: str) -> int | None:
    m = re.search(r"steps(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def get_total_steps(exp_dir: Path, summary: dict) -> int:
    if "steps" in summary:
        return int(summary["steps"])
    parsed = parse_steps_from_name(exp_dir.name)
    if parsed is not None:
        return parsed

    step_files = list(exp_dir.glob("qgen_step_*.pt"))
    steps = []
    for p in step_files:
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


def choose_checkpoint_for_target(target_step: int, available: Dict[int, Path], total_steps: int) -> Tuple[int, Path]:
    if target_step in available:
        return target_step, available[target_step]

    all_steps = sorted(available.keys())
    if not all_steps:
        raise FileNotFoundError("No qgen checkpoints found.")

    # For the "first" snapshot, use the earliest available checkpoint.
    if target_step == 1:
        chosen_step = all_steps[0]
        return chosen_step, available[chosen_step]

    # Otherwise use nearest available step.
    chosen_step = min(all_steps, key=lambda s: abs(s - target_step))
    return chosen_step, available[chosen_step]


def infer_n_layers_from_checkpoint(ckpt_path: Path) -> int:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "weights" in state:
        return int(state["weights"].shape[0])
    raise ValueError(f"Unexpected checkpoint format in {ckpt_path}")


def load_generator_from_checkpoint(ckpt_path: Path, device: torch.device) -> QGenerator:
    state = torch.load(ckpt_path, map_location=device)
    if not (isinstance(state, dict) and "weights" in state):
        raise ValueError(f"Unexpected checkpoint format in {ckpt_path}")

    n_layers = int(state["weights"].shape[0])
    gen = QGenerator(n_layer=n_layers, init_std=1.0, seed=0).to(device)
    gen.load_state_dict(state)
    gen.eval()
    return gen


def load_real_reference(exp_dir: Path, valid_tuples_path: str, eval_samples: int, real_seed: int) -> np.ndarray:
    npy_path = exp_dir / "real_samples_final.npy"
    if npy_path.exists():
        return np.load(npy_path)

    real_data = load_real_data(valid_tuples_path)
    rng = np.random.default_rng(real_seed)
    idx = rng.choice(len(real_data), size=eval_samples, replace=True)
    return real_data[idx]


def build_fixed_latent(eval_samples: int, latent_dim: int, latent_distribution: str, device: torch.device, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return sample_latent_torch(eval_samples, latent_dim, latent_distribution, device)


def generate_fake_samples(gen: QGenerator, latent_vectors: torch.Tensor, batch_size: int) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for start in range(0, latent_vectors.shape[0], batch_size):
            z = latent_vectors[start:start + batch_size]
            fake = gen(z).detach().cpu().numpy()
            outs.append(fake)
    return np.concatenate(outs, axis=0)


def make_single_snapshot_plot(real_batch: np.ndarray, fake_batch: np.ndarray, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        ax.hist(real_batch[:, i], bins=30, alpha=0.55, density=True, label="real")
        ax.hist(fake_batch[:, i], bins=30, alpha=0.55, density=True, label="fake")
        ax.set_title(EDGE_NAMES[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_combined_snapshot_plot(
    real_batch: np.ndarray,
    snapshot_batches: Dict[int, np.ndarray],
    labels: Dict[int, str],
    out_path: Path,
    exp_name: str,
) -> None:
    steps = list(snapshot_batches.keys())
    fig, axes = plt.subplots(len(steps), 6, figsize=(22, 4.4 * len(steps)), squeeze=False)
    fig.suptitle(f"{exp_name}: Per-edge distributions (before / middle / after)", fontsize=18, fontweight="bold")

    for row_idx, step in enumerate(steps):
        fake_batch = snapshot_batches[step]
        row_label = labels[step]

        for col_idx, edge_name in enumerate(EDGE_NAMES):
            ax = axes[row_idx, col_idx]
            ax.hist(real_batch[:, col_idx], bins=30, alpha=0.50, density=True, label="real")
            ax.hist(fake_batch[:, col_idx], bins=30, alpha=0.50, density=True, label="fake")
            ax.set_title(f"{edge_name}\n{row_label}")
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
        return (
            pd.Series(series)
            .rolling(window=window, min_periods=1, center=True)
            .mean()
            .to_numpy()
        )

    steps = df["step"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{exp_name}: Training overview", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(steps, smooth(df["d_loss"]), label="d_loss")
    ax.plot(steps, smooth(df["g_loss"]), label="g_loss")
    ax.set_title("Losses")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    if "real_score" in df.columns:
        ax.plot(steps, smooth(df["real_score"]), label="real_score")
    if "fake_score_d" in df.columns:
        ax.plot(steps, smooth(df["fake_score_d"]), label="fake_score_d")
    if "fake_score_g" in df.columns:
        ax.plot(steps, smooth(df["fake_score_g"]), label="fake_score_g")
    ax.axhline(0.5, linestyle="--", alpha=0.6)
    ax.set_title("Scores")
    ax.set_xlabel("Step")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    if "fake_mean" in df.columns:
        ax.plot(steps, smooth(df["fake_mean"]), label="fake_mean")
    if "fake_std" in df.columns:
        ax.plot(steps, smooth(df["fake_std"]), label="fake_std")
    ax.set_title("Global fake stats")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    for edge in ["ab", "bc", "cd", "da", "ac", "bd"]:
        col = f"{edge}_mean"
        if col in df.columns:
            ax.plot(steps, smooth(df[col]), label=edge)
    ax.set_title("Per-edge means")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def process_experiment(exp_dir: Path, args: argparse.Namespace, out_root: Path) -> None:
    print(f"\n[INFO] Processing: {exp_dir}")

    summary = try_load_summary(exp_dir)
    total_steps = get_total_steps(exp_dir, summary)
    latent_distribution = str(summary.get("latent_distribution", "uniform"))
    seed_from_name = parse_seed_from_name(exp_dir.name)
    latent_seed = args.latent_seed if args.latent_seed is not None else (seed_from_name or 0)

    target_steps = [1, max(1, total_steps // 2), total_steps]
    available_ckpts = collect_qgen_checkpoints(exp_dir, total_steps)

    if not available_ckpts:
        print(f"[WARN] No generator checkpoints found in {exp_dir}, skipping.")
        return

    exp_out = out_root / exp_dir.name
    exp_out.mkdir(parents=True, exist_ok=True)

    real_ref = load_real_reference(
        exp_dir=exp_dir,
        valid_tuples_path=args.valid_tuples,
        eval_samples=args.eval_samples,
        real_seed=args.real_seed,
    )

    fixed_latent = build_fixed_latent(
        eval_samples=args.eval_samples,
        latent_dim=args.latent_dim,
        latent_distribution=latent_distribution,
        device=torch.device(args.device),
        seed=latent_seed,
    )

    snapshot_batches: Dict[int, np.ndarray] = {}
    labels: Dict[int, str] = {}
    manifest_rows = []

    for target in target_steps:
        used_step, ckpt_path = choose_checkpoint_for_target(target, available_ckpts, total_steps)
        gen = load_generator_from_checkpoint(ckpt_path, device=torch.device(args.device))
        fake_ref = generate_fake_samples(gen, fixed_latent, batch_size=args.batch_size)

        key = target
        snapshot_batches[key] = fake_ref

        if used_step == target:
            label = f"target {target} / used {used_step}"
        else:
            label = f"target {target} / used {used_step} (nearest available)"

        labels[key] = label
        manifest_rows.append(
            {
                "target_step": target,
                "used_step": used_step,
                "checkpoint": str(ckpt_path),
                "label": label,
            }
        )

        single_name = f"per_edge_distributions_target_{target:06d}_used_{used_step:06d}.png"
        make_single_snapshot_plot(
            real_batch=real_ref,
            fake_batch=fake_ref,
            out_path=exp_out / single_name,
            title=f"{exp_dir.name}: per-edge distributions ({label})",
        )
        print(f"[INFO] Saved {single_name}")

    make_combined_snapshot_plot(
        real_batch=real_ref,
        snapshot_batches=snapshot_batches,
        labels=labels,
        out_path=exp_out / "per_edge_distributions_before_middle_after.png",
        exp_name=exp_dir.name,
    )
    print("[INFO] Saved combined snapshot plot")

    metrics_csv = exp_dir / "metrics.csv"
    if metrics_csv.exists():
        make_metrics_overview_plot(
            metrics_csv=metrics_csv,
            out_path=exp_out / "metrics_overview_from_logs.png",
            exp_name=exp_dir.name,
        )
        print("[INFO] Saved metrics overview plot")

    (exp_out / "snapshot_manifest.json").write_text(
        json.dumps(
            {
                "experiment": exp_dir.name,
                "source_dir": str(exp_dir),
                "latent_distribution": latent_distribution,
                "total_steps": total_steps,
                "rows": manifest_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if args.zip is None and args.root_dir is None:
        raise ValueError("Please provide either --zip or --root-dir.")

    work_root = args.output_root
    work_root.mkdir(parents=True, exist_ok=True)

    if args.zip is not None:
        root_dir = extract_zip_if_needed(args.zip, work_root)
    else:
        root_dir = args.root_dir

    exp_dirs = discover_experiment_dirs(root_dir, args.experiment)
    if not exp_dirs:
        raise FileNotFoundError("No experiment folders found.")

    print(f"[INFO] Found {len(exp_dirs)} experiment(s).")

    for exp_dir in exp_dirs:
        process_experiment(exp_dir, args, work_root)

    print("\n[INFO] Done.")
    print(f"[INFO] Output root: {work_root.resolve()}")


if __name__ == "__main__":
    main()
