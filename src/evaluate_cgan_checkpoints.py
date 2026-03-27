from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def triangle_triplet_is_valid(a: float, b: float, c: float, tol: float = 1e-8) -> bool:
    return a <= b + c + tol and b <= a + c + tol and c <= a + b + tol


def sample_satisfies_triangle_inequality(sample, tol: float = 1e-8) -> bool:
    # Reihenfolge aus training_qgan.py: ab, bc, cd, da, ac, bd
    ab, bc, cd, da, ac, bd = sample
    return (
        triangle_triplet_is_valid(ab, bc, ac, tol)
        and triangle_triplet_is_valid(ab, bd, da, tol)
        and triangle_triplet_is_valid(ac, cd, da, tol)
        and triangle_triplet_is_valid(bc, cd, bd, tol)
    )


def batch_triangle_valid_mask(batch: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    return np.array([sample_satisfies_triangle_inequality(row, tol) for row in batch], dtype=bool)


def import_training_module(training_script: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("training_cgan_eval_module", training_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Konnte Trainingsskript nicht laden: {training_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_generator(module: Any, checkpoint_path: Path, device: str = "cpu"):
    generator = module.Generator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "generator_state_dict" in checkpoint:
            state_dict = checkpoint["generator_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise RuntimeError("Checkpoint-Format wird nicht erkannt.")
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator


def generate_fake_batch(module: Any, generator: torch.nn.Module, num_samples: int) -> np.ndarray:
    with torch.no_grad():
        fake = module.create_batch_fake(generator, num_samples)
    return fake.detach().cpu().numpy().astype(np.float32)


def generate_real_batch(module: Any, num_samples: int, cities_path: Path, cache_path: Path) -> Tuple[np.ndarray, float]:
    cities = module.load_cities(str(cities_path))
    cache = module.load_distance_cache_dict(str(cache_path))
    batch, clip_fraction = module.create_batch_real(cities, num_samples, cache, module.rng)
    return np.asarray(batch, dtype=np.float32), float(clip_fraction)


def flattened_stats(x: np.ndarray, prefix: str) -> Dict[str, float]:
    flat = x.reshape(-1)
    return {
        f"{prefix}_mean": float(np.mean(flat)),
        f"{prefix}_std": float(np.std(flat)),
        f"{prefix}_min": float(np.min(flat)),
        f"{prefix}_max": float(np.max(flat)),
        f"{prefix}_near_zero_frac": float(np.mean(flat <= 0.05)),
        f"{prefix}_near_one_frac": float(np.mean(flat >= 0.95)),
    }


def parse_step_from_name(path: Path) -> int:
    m = re.search(r"step_(\d+)", path.stem)
    if m:
        return int(m.group(1))
    if "final" in path.stem:
        return -1
    nums = re.findall(r"(\d+)", path.stem)
    return int(nums[-1]) if nums else -1


def resolve_checkpoints(checkpoint: Path | None, checkpoint_dir: Path | None) -> List[Path]:
    paths: List[Path] = []
    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint nicht gefunden: {checkpoint}")
        paths = [checkpoint]
    elif checkpoint_dir is not None:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint-Ordner nicht gefunden: {checkpoint_dir}")
        paths = sorted(checkpoint_dir.glob("generator_step_*.pt"), key=parse_step_from_name)
        final_path = checkpoint_dir / "generator_final.pt"
        if final_path.exists():
            paths.append(final_path)
        if not paths:
            raise FileNotFoundError(f"Keine generator_step_*.pt oder generator_final.pt in {checkpoint_dir} gefunden.")
    else:
        raise ValueError("Entweder --checkpoint oder --checkpoint-dir angeben.")
    return paths


def evaluate_single_checkpoint(module: Any, ckpt: Path, num_samples: int, cities: Path, cache: Path, device: str):
    generator = load_generator(module, ckpt, device=device)
    real_batch, real_clip_frac = generate_real_batch(module, num_samples, cities, cache)
    fake_batch = generate_fake_batch(module, generator, num_samples)

    real_valid = float(np.mean(batch_triangle_valid_mask(real_batch)))
    fake_valid = float(np.mean(batch_triangle_valid_mask(fake_batch)))

    metrics: Dict[str, float] = {
        "checkpoint": ckpt.name,
        "step": parse_step_from_name(ckpt),
        "triangle_valid_frac_real": real_valid,
        "triangle_valid_frac_fake": fake_valid,
        "real_clip_frac": real_clip_frac,
    }
    metrics.update(flattened_stats(real_batch, "real"))
    metrics.update(flattened_stats(fake_batch, "fake"))
    return metrics, real_batch, fake_batch


def make_progress_plots(df: pd.DataFrame, output_dir: Path) -> None:
    df_plot = df.copy()
    # final ans Ende, falls vorhanden
    max_step = df_plot.loc[df_plot["step"] >= 0, "step"].max() if np.any(df_plot["step"] >= 0) else 0
    df_plot.loc[df_plot["step"] < 0, "step"] = max_step + 1
    df_plot = df_plot.sort_values("step")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(df_plot["step"], df_plot["triangle_valid_frac_fake"], marker="o", label="fake valid fraction")
    ax.plot(df_plot["step"], df_plot["triangle_valid_frac_real"], linestyle="--", label="real valid fraction")
    ax.set_title("Triangle-valid fraction over checkpoints")
    ax.set_xlabel("step")
    ax.set_ylabel("fraction")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(df_plot["step"], df_plot["fake_std"], marker="o", label="fake std")
    ax.plot(df_plot["step"], df_plot["real_std"], linestyle="--", label="real std")
    ax.set_title("Edge-weight std over checkpoints")
    ax.set_xlabel("step")
    ax.set_ylabel("std")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "checkpoint_progress.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_final_distribution_plots(real_batch: np.ndarray, fake_batch: np.ndarray, output_dir: Path) -> None:
    edge_names = ["ab", "bc", "cd", "da", "ac", "bd"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CGAN Evaluation: Real vs Fake", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.hist(real_batch.reshape(-1), bins=40, alpha=0.6, density=True, label="real")
    ax.hist(fake_batch.reshape(-1), bins=40, alpha=0.6, density=True, label="fake")
    ax.set_title("All edge values")
    ax.set_xlabel("normalized edge weight")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    x = np.arange(len(edge_names))
    width = 0.35

    ax = axes[0, 1]
    ax.bar(x - width / 2, np.mean(real_batch, axis=0), width, label="real")
    ax.bar(x + width / 2, np.mean(fake_batch, axis=0), width, label="fake")
    ax.set_xticks(x)
    ax.set_xticklabels(edge_names)
    ax.set_title("Per-edge mean")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.bar(x - width / 2, np.std(real_batch, axis=0), width, label="real")
    ax.bar(x + width / 2, np.std(fake_batch, axis=0), width, label="fake")
    ax.set_xticks(x)
    ax.set_xticklabels(edge_names)
    ax.set_title("Per-edge std")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.bar(["real", "fake"], [np.mean(batch_triangle_valid_mask(real_batch)), np.mean(batch_triangle_valid_mask(fake_batch))])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Triangle-valid fraction")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "evaluation_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Per-edge distributions", fontsize=16, fontweight="bold")
    for i, ax in enumerate(axes.flat):
        ax.hist(real_batch[:, i], bins=30, alpha=0.55, density=True, label="real")
        ax.hist(fake_batch[:, i], bins=30, alpha=0.55, density=True, label="fake")
        ax.set_title(edge_names[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "per_edge_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one or many CGAN checkpoints.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path, help="Path to a single generator checkpoint (.pt).")
    group.add_argument("--checkpoint-dir", type=Path, help="Directory containing generator_step_*.pt checkpoints.")
    parser.add_argument("--training-script", type=Path, default=Path("training_cgan.py"), help="Path to training_cgan.py.")
    parser.add_argument("--cities", type=Path, default=Path("cities.csv"), help="Path to cities.csv.")
    parser.add_argument("--cache", type=Path, default=Path("distance_cache.csv"), help="Path to distance_cache.csv.")
    parser.add_argument("--num-samples", type=int, default=5000, help="How many real/fake samples per checkpoint.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("eval_cgan_output"), help="Directory for reports and plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    module = import_training_module(args.training_script)
    checkpoints = resolve_checkpoints(args.checkpoint, args.checkpoint_dir)

    rows: List[Dict[str, float]] = []
    last_real_batch = None
    last_fake_batch = None

    for ckpt in checkpoints:
        metrics, real_batch, fake_batch = evaluate_single_checkpoint(
            module, ckpt, args.num_samples, args.cities, args.cache, args.device
        )
        rows.append(metrics)
        last_real_batch = real_batch
        last_fake_batch = fake_batch
        print(
            f"[{ckpt.name}] step={metrics['step']} | "
            f"valid_real={metrics['triangle_valid_frac_real']:.4f} | "
            f"valid_fake={metrics['triangle_valid_frac_fake']:.4f} | "
            f"real_std={metrics['real_std']:.4f} | "
            f"fake_std={metrics['fake_std']:.4f}"
        )

    df = pd.DataFrame(rows).sort_values("step")
    df.to_csv(args.output_dir / "checkpoint_evaluation_metrics.csv", index=False)
    with open(args.output_dir / "checkpoint_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    make_progress_plots(df, args.output_dir)

    if last_real_batch is not None and last_fake_batch is not None:
        np.save(args.output_dir / "real_samples_last.npy", last_real_batch)
        np.save(args.output_dir / "fake_samples_last.npy", last_fake_batch)
        make_final_distribution_plots(last_real_batch, last_fake_batch, args.output_dir)

    print(f"\nSaved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
