from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# ------------------------------------------------------------
# Geometry helpers (evaluation only, not used for training)
# ------------------------------------------------------------

def triangle_triplet_is_valid(a: float, b: float, c: float, tol: float = 1e-8) -> bool:
    return (
        a <= b + c + tol and
        b <= a + c + tol and
        c <= a + b + tol
    )


def sample_satisfies_triangle_inequality(sample, tol: float = 1e-8) -> bool:
    ab, bc, cd, da, ac, bd = sample

    return (
        triangle_triplet_is_valid(ab, bc, ac, tol) and  # triangle abc
        triangle_triplet_is_valid(ab, bd, da, tol) and  # triangle abd
        triangle_triplet_is_valid(ac, cd, da, tol) and  # triangle acd
        triangle_triplet_is_valid(bc, cd, bd, tol)      # triangle bcd
    )

def count_triangle_valid_samples(batch, tol: float = 1e-8) -> int:
    batch_np = batch.detach().cpu().numpy() if hasattr(batch, "detach") else batch
    valid_count = 0
    for sample in batch_np:
        if sample_satisfies_triangle_inequality(sample, tol=tol):
            valid_count += 1
    return valid_count

def batch_triangle_valid_mask(batch: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    return np.array([sample_satisfies_triangle_inequality(row, tol) for row in batch], dtype=bool)


# ------------------------------------------------------------
# Stats helpers
# ------------------------------------------------------------

def flattened_stats(x: np.ndarray, prefix: str) -> Dict[str, float]:
    flat = x.reshape(-1)
    return {
        f"{prefix}_mean": float(np.mean(flat)),
        f"{prefix}_std": float(np.std(flat)),
        f"{prefix}_min": float(np.min(flat)),
        f"{prefix}_max": float(np.max(flat)),
        f"{prefix}_q05": float(np.quantile(flat, 0.05)),
        f"{prefix}_q25": float(np.quantile(flat, 0.25)),
        f"{prefix}_median": float(np.quantile(flat, 0.50)),
        f"{prefix}_q75": float(np.quantile(flat, 0.75)),
        f"{prefix}_q95": float(np.quantile(flat, 0.95)),
        f"{prefix}_near_zero_frac": float(np.mean(flat <= 0.05)),
        f"{prefix}_near_one_frac": float(np.mean(flat >= 0.95)),
    }


def per_edge_mean_std(x: np.ndarray, prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i in range(x.shape[1]):
        out[f"{prefix}_edge{i}_mean"] = float(np.mean(x[:, i]))
        out[f"{prefix}_edge{i}_std"] = float(np.std(x[:, i]))
    return out


def rounded_uniqueness_ratio(x: np.ndarray, decimals: int = 4) -> float:
    rounded = np.round(x, decimals=decimals)
    unique = np.unique(rounded, axis=0)
    return float(len(unique) / max(len(rounded), 1))


@dataclass
class EvalSummary:
    num_samples: int
    triangle_valid_frac_real: float
    triangle_valid_frac_fake: float
    rounded_uniqueness_real: float
    rounded_uniqueness_fake: float
    l1_mean_gap: float
    l1_std_gap: float
    l1_edge_mean_gap: float
    l1_edge_std_gap: float
    real_clip_frac: float


# ------------------------------------------------------------
# Loading / generation
# ------------------------------------------------------------

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
            # maybe the checkpoint itself is already a plain state_dict
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


def generate_real_batch(module: Any, num_samples: int, valid_tuples_path: Path) -> Tuple[np.ndarray, float]:
    df = pd.read_csv(valid_tuples_path)

    needed = ["e_ab", "e_bc", "e_cd", "e_da", "e_ac", "e_bd"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r} in {valid_tuples_path}")

    arr_km = df[needed].to_numpy(dtype=np.float32)
    arr_norm = np.clip(arr_km / module.MAX_EDGE_LENGTH_KM, 0.0, 1.0)

    if len(arr_norm) == 0:
        raise RuntimeError(f"No valid tuples found in {valid_tuples_path}")

    idx = module.rng.choice(len(arr_norm), size=num_samples, replace=True)
    batch = arr_norm[idx]
    clip_fraction = float(np.mean(batch >= 1.0))
    return np.asarray(batch, dtype=np.float32), clip_fraction

# ------------------------------------------------------------
# Reporting / plots
# ------------------------------------------------------------

def build_summary(real_batch: np.ndarray, fake_batch: np.ndarray, real_clip_frac: float) -> Tuple[EvalSummary, Dict[str, float]]:
    metrics: Dict[str, float] = {}
    metrics.update(flattened_stats(real_batch, "real"))
    metrics.update(flattened_stats(fake_batch, "fake"))
    metrics.update(per_edge_mean_std(real_batch, "real"))
    metrics.update(per_edge_mean_std(fake_batch, "fake"))

    real_valid = batch_triangle_valid_mask(real_batch)
    fake_valid = batch_triangle_valid_mask(fake_batch)
    metrics["triangle_valid_frac_real"] = float(np.mean(real_valid))
    metrics["triangle_valid_frac_fake"] = float(np.mean(fake_valid))
    metrics["rounded_uniqueness_real"] = rounded_uniqueness_ratio(real_batch)
    metrics["rounded_uniqueness_fake"] = rounded_uniqueness_ratio(fake_batch)

    l1_mean_gap = abs(metrics["real_mean"] - metrics["fake_mean"])
    l1_std_gap = abs(metrics["real_std"] - metrics["fake_std"])
    edge_mean_gap = float(np.mean(np.abs(np.mean(real_batch, axis=0) - np.mean(fake_batch, axis=0))))
    edge_std_gap = float(np.mean(np.abs(np.std(real_batch, axis=0) - np.std(fake_batch, axis=0))))

    summary = EvalSummary(
        num_samples=int(real_batch.shape[0]),
        triangle_valid_frac_real=float(np.mean(real_valid)),
        triangle_valid_frac_fake=float(np.mean(fake_valid)),
        rounded_uniqueness_real=metrics["rounded_uniqueness_real"],
        rounded_uniqueness_fake=metrics["rounded_uniqueness_fake"],
        l1_mean_gap=float(l1_mean_gap),
        l1_std_gap=float(l1_std_gap),
        l1_edge_mean_gap=edge_mean_gap,
        l1_edge_std_gap=edge_std_gap,
        real_clip_frac=float(real_clip_frac),
    )
    return summary, metrics


def save_examples(real_batch: np.ndarray, fake_batch: np.ndarray, output_dir: Path, n: int = 10) -> None:
    examples = pd.DataFrame(
        np.vstack([real_batch[:n], fake_batch[:n]]),
        columns=["ab", "bc", "cd", "da", "ac", "bd"],
    )
    examples.insert(0, "kind", ["real"] * min(n, len(real_batch)) + ["fake"] * min(n, len(fake_batch)))
    examples.to_csv(output_dir / "sample_examples.csv", index=False)


def make_plots(real_batch: np.ndarray, fake_batch: np.ndarray, output_dir: Path) -> None:
    edge_names = ["ab", "bc", "cd", "da", "ac", "bd"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CGAN Evaluation: Real vs Fake", fontsize=16, fontweight="bold")

    # Global histogram
    ax = axes[0, 0]
    ax.hist(real_batch.reshape(-1), bins=40, alpha=0.6, density=True, label="real")
    ax.hist(fake_batch.reshape(-1), bins=40, alpha=0.6, density=True, label="fake")
    ax.set_title("All edge values")
    ax.set_xlabel("normalized edge weight")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Per-edge means
    ax = axes[0, 1]
    x = np.arange(len(edge_names))
    real_means = np.mean(real_batch, axis=0)
    fake_means = np.mean(fake_batch, axis=0)
    width = 0.35
    ax.bar(x - width / 2, real_means, width, label="real")
    ax.bar(x + width / 2, fake_means, width, label="fake")
    ax.set_xticks(x)
    ax.set_xticklabels(edge_names)
    ax.set_title("Per-edge mean")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Per-edge std
    ax = axes[1, 0]
    real_stds = np.std(real_batch, axis=0)
    fake_stds = np.std(fake_batch, axis=0)
    ax.bar(x - width / 2, real_stds, width, label="real")
    ax.bar(x + width / 2, fake_stds, width, label="fake")
    ax.set_xticks(x)
    ax.set_xticklabels(edge_names)
    ax.set_title("Per-edge std")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Triangle validity bars
    ax = axes[1, 1]
    real_valid = np.mean(batch_triangle_valid_mask(real_batch))
    fake_valid = np.mean(batch_triangle_valid_mask(fake_batch))
    ax.bar(["real", "fake"], [real_valid, fake_valid])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Triangle-valid fraction")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "evaluation_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Separate per-edge histograms
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


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CGAN generator against real graph-edge samples.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to generator checkpoint (.pt).")
    parser.add_argument("--training-script", type=Path, default=Path("training_cgan.py"), help="Path to training_cgan.py.")
    parser.add_argument("--valid-tuples", type=Path, default=Path("valid_tuples.csv"), help="Path to valid_tuples.csv.")
    parser.add_argument("--num-samples", type=int, default=5000, help="How many real/fake samples to compare.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("eval_cgan_output"), help="Directory for reports and plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint nicht gefunden: {args.checkpoint}\n"
            "Hinweis: Dein Training speichert aktuell sehr wahrscheinlich noch kein Generator-Checkpoint.\n"
            "Speichere nach dem Training z.B. mit:\n"
            "torch.save({'generator_state_dict': generator.state_dict()}, log_dir / 'generator_final.pt')"
        )

    module = import_training_module(args.training_script)
    generator = load_generator(module, args.checkpoint, device=args.device)

    real_batch, real_clip_frac = generate_real_batch(module, args.num_samples, args.valid_tuples)
    fake_batch = generate_fake_batch(module, generator, args.num_samples)

    summary, metrics = build_summary(real_batch, fake_batch, real_clip_frac)

    with open(args.output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    with open(args.output_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame([metrics]).to_csv(args.output_dir / "evaluation_metrics.csv", index=False)
    save_examples(real_batch, fake_batch, args.output_dir, n=10)
    np.save(args.output_dir / "real_samples.npy", real_batch)
    np.save(args.output_dir / "fake_samples.npy", fake_batch)
    make_plots(real_batch, fake_batch, args.output_dir)

    print("\n=== CGAN EVALUATION SUMMARY ===")
    print(f"num_samples:               {summary.num_samples}")
    print(f"triangle_valid_frac_real:  {summary.triangle_valid_frac_real:.4f}")
    print(f"triangle_valid_frac_fake:  {summary.triangle_valid_frac_fake:.4f}")
    print(f"rounded_uniqueness_real:   {summary.rounded_uniqueness_real:.4f}")
    print(f"rounded_uniqueness_fake:   {summary.rounded_uniqueness_fake:.4f}")
    print(f"l1_mean_gap:               {summary.l1_mean_gap:.4f}")
    print(f"l1_std_gap:                {summary.l1_std_gap:.4f}")
    print(f"l1_edge_mean_gap:          {summary.l1_edge_mean_gap:.4f}")
    print(f"l1_edge_std_gap:           {summary.l1_edge_std_gap:.4f}")
    print(f"real_clip_frac:            {summary.real_clip_frac:.4f}")
    print(f"\nSaved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
