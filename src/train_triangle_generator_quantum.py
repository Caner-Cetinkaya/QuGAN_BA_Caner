from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import pandas as pd
from config import MAX_EDGE_LENGTH_KM

# Reuse the user's existing quantum generator implementation.
from generator import QGenerator


@dataclass
class Summary:
    steps: int
    batch_size: int
    learning_rate: float
    latent_distribution: str
    n_layers: int
    init_std: float | None
    final_loss: float
    final_triangle_penalty: float
    final_triangle_valid_frac: float
    final_fake_mean: float
    final_fake_std: float
    best_triangle_valid_frac: float


def sample_latent(
    rng: np.random.Generator,
    batch_size: int,
    latent_dim: int,
    distribution: str,
) -> np.ndarray:
    distribution = distribution.lower().strip()
    if distribution == "uniform":
        # Default latent for the current generator: small non-periodic angles in [0, 1] rad.
        return rng.uniform(0.0, 1.0, size=(batch_size, latent_dim)).astype(float)
    if distribution == "normal":
        return rng.normal(0, 1.0, size=(batch_size, latent_dim)).astype(float)
    elif distribution == "uniform_pm1":
        # Full-period phase input. This is kept only for explicit experiments:
        # feeding U(0, 2pi) directly into AngleEmbedding produces outputs that are
        # naturally centered around 0.5 because the measurement averages over a full period.
        return rng.uniform(0, 2 * np.pi, size=(batch_size, latent_dim)).astype(float)
    raise ValueError(f"Unknown latent distribution: {distribution!r}")


def smooth_relu(x, beta: float = 20.0):
    return pnp.log1p(pnp.exp(beta * x)) / beta


def triangle_penalty_q(batch) -> pnp.ndarray:
    ab = batch[:, 0]
    bc = batch[:, 1]
    cd = batch[:, 2]
    da = batch[:, 3]
    ac = batch[:, 4]
    bd = batch[:, 5]

    # Differentiable hinge-style penalty: >0 only when triangle inequality is violated.
    violation = (
        smooth_relu(ab - (bc + ac)) +
        smooth_relu(bc - (ab + ac)) +
        smooth_relu(ac - (ab + bc)) +

        smooth_relu(ab - (da + bd)) +
        smooth_relu(da - (ab + bd)) +
        smooth_relu(bd - (ab + da)) +

        smooth_relu(ac - (cd + da)) +
        smooth_relu(cd - (ac + da)) +
        smooth_relu(da - (ac + cd)) +

        smooth_relu(bc - (cd + bd)) +
        smooth_relu(cd - (bc + bd)) +
        smooth_relu(bd - (bc + cd))
    )

    return pnp.mean(violation) / 12.0

"""
def triangle_penalty_q(batch) -> pnp.ndarray:
    ab = batch[:, 0]
    bc = batch[:, 1]
    cd = batch[:, 2]
    da = batch[:, 3]
    ac = batch[:, 4]
    bd = batch[:, 5]

# BSP 0,8 <= 0,6 + 0,3 --> valid keine violations
# aber 0,8 <= 0,6 + 0,1 --> violation von 0,1 --> Da 0,8-0,7 = 0,1 violation
# am ende Summe aller violations / 12 --> je kleiner desto besser, 0 = perfekt, >0 = verletzt, max wäre 1 wenn alle 12 Bedingungen maximal verletzt wären (zB 0,8 <= 0,01 + 0,01 --> violation von 0,78 pro Bedingung)
    violation = (
        pnp.maximum(0.0, ab - (bc + ac)) +
        pnp.maximum(0.0, bc - (ab + ac)) +
        pnp.maximum(0.0, ac - (ab + bc)) +

        pnp.maximum(0.0, ab - (da + bd)) +
        pnp.maximum(0.0, da - (ab + bd)) +
        pnp.maximum(0.0, bd - (ab + da)) +

        pnp.maximum(0.0, ac - (cd + da)) +
        pnp.maximum(0.0, cd - (ac + da)) +
        pnp.maximum(0.0, da - (ac + cd)) +

        pnp.maximum(0.0, bc - (cd + bd)) +
        pnp.maximum(0.0, cd - (bc + bd)) +
        pnp.maximum(0.0, bd - (bc + cd))
    )

    return pnp.mean(violation) / 12.0
    """

def triangle_valid_mask_np(batch: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    ab = batch[:, 0]
    bc = batch[:, 1]
    cd = batch[:, 2]
    da = batch[:, 3]
    ac = batch[:, 4]
    bd = batch[:, 5]

    def tri_ok(x, y, z):
        return (x <= y + z + tol) & (y <= x + z + tol) & (z <= x + y + tol)

    return (
        tri_ok(ab, bc, ac)
        & tri_ok(ab, da, bd)
        & tri_ok(ac, cd, da)
        & tri_ok(bc, cd, bd)
    )


def rounded_uniqueness_ratio(x: np.ndarray, decimals: int = 4) -> float:
    rounded = np.round(x, decimals=decimals)
    unique = np.unique(rounded, axis=0)
    return float(len(unique) / max(len(rounded), 1))


def compute_metrics(fake_batch: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "triangle_penalty": float(triangle_penalty_q(pnp.array(fake_batch, dtype=float))),
        "triangle_valid_frac": float(np.mean(triangle_valid_mask_np(fake_batch))),
        "fake_mean": float(np.mean(fake_batch)),
        "fake_std": float(np.std(fake_batch)),
        "fake_min": float(np.min(fake_batch)),
        "fake_max": float(np.max(fake_batch)),
        "fake_near_zero_frac": float(np.mean(fake_batch <= 0.05)),
        "fake_near_one_frac": float(np.mean(fake_batch >= 0.95)),
        "rounded_uniqueness_fake": rounded_uniqueness_ratio(fake_batch),
        "ab_mean": float(np.mean(fake_batch[:, 0])),
        "bc_mean": float(np.mean(fake_batch[:, 1])),
        "cd_mean": float(np.mean(fake_batch[:, 2])),
        "da_mean": float(np.mean(fake_batch[:, 3])),
        "ac_mean": float(np.mean(fake_batch[:, 4])),
        "bd_mean": float(np.mean(fake_batch[:, 5])),
        "ab_std": float(np.std(fake_batch[:, 0])),
        "bc_std": float(np.std(fake_batch[:, 1])),
        "cd_std": float(np.std(fake_batch[:, 2])),
        "da_std": float(np.std(fake_batch[:, 3])),
        "ac_std": float(np.std(fake_batch[:, 4])),
        "bd_std": float(np.std(fake_batch[:, 5])),
    }
    edge_names = ["ab", "bc", "cd", "da", "ac", "bd"]
    for i, name in enumerate(edge_names):
        metrics[f"{name}_mean"] = float(np.mean(fake_batch[:, i]))
        metrics[f"{name}_std"] = float(np.std(fake_batch[:, i]))
    return metrics

def load_real_samples(valid_tuples_path: str, num_samples: int, rng):
    df = pd.read_csv(valid_tuples_path)

    needed = ["e_ab", "e_bc", "e_cd", "e_da", "e_ac", "e_bd"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r} in {valid_tuples_path}")

    real = df[needed].to_numpy(dtype=np.float64)
    real = np.clip(real / MAX_EDGE_LENGTH_KM, 0.0, 1.0)

    idx = rng.choice(len(real), size=num_samples, replace=True)
    return real[idx]
def load_real_edge_stats(valid_tuples_path: str):
    df = pd.read_csv(valid_tuples_path)

    needed = ["e_ab", "e_bc", "e_cd", "e_da", "e_ac", "e_bd"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r} in {valid_tuples_path}")

    real = df[needed].to_numpy(dtype=np.float64)
    real = np.clip(real / MAX_EDGE_LENGTH_KM, 0.0, 1.0)

    real_mean = pnp.array(real.mean(axis=0), dtype=float)
    real_std = pnp.array(real.std(axis=0), dtype=float)

    return real_mean, real_std


def distribution_match_loss_q(fake_batch, real_mean, real_std):
    fake_mean = pnp.mean(fake_batch, axis=0)
    fake_std = pnp.std(fake_batch, axis=0)

    mean_loss = pnp.mean((fake_mean - real_mean) ** 2)
    std_loss = pnp.mean((fake_std - real_std) ** 2)

    return mean_loss + std_loss

def make_plot(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["step"]).sort_values("step")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Quantum Generator: Triangle-Only Training", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(df["step"], df["loss"], label="loss")
    ax.plot(df["step"], df["triangle_loss"], label="triangle_loss")
    ax.plot(df["step"], df["dist_loss"], label="dist_loss")
    ax.plot(df["step"], df["triangle_penalty"], label="triangle_penalty")
    ax.set_title("Loss / penalty")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(df["step"], df["triangle_valid_frac"], label="triangle_valid_frac")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Triangle-valid fraction")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(df["step"], df["fake_mean"], label="fake_mean")
    ax.plot(df["step"], df["fake_std"], label="fake_std")
    ax.set_title("Batch stats")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(df["step"], df["grad_norm"], label="grad_norm")
    ax.set_title("Gradient norm")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def make_per_edge_distribution_plot(real_batch: np.ndarray, fake_batch: np.ndarray, out_path: Path) -> None:
    edge_names = ["ab", "bc", "cd", "da", "ac", "bd"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Quantum Generator: Per-edge distributions", fontsize=16, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        ax.hist(real_batch[:, i], bins=30, alpha=0.55, density=True, label="real")
        ax.hist(fake_batch[:, i], bins=30, alpha=0.55, density=True, label="fake")
        ax.set_title(edge_names[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train only the quantum generator to satisfy triangle inequality.")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--latent-dim", type=int, default=6)
    #parser.add_argument("--latent-distribution", type=str, default="uniform", choices=["uniform", "normal"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--init-std", type=float, default=None, help="Optional reinitialization std for QGenerator weights")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--valid-tuples", type=str, default="valid_tuples.csv")
    parser.add_argument("--lambda-dist", type=float, default=10.0)
    parser.add_argument("--latent-distribution", type=str, default="uniform", choices=["uniform", "normal", "uniform_pm1"])
    return parser.parse_args()

csv_columns = [
    "step",
    "loss",
    "triangle_loss",
    "dist_loss",
    "triangle_penalty",
    "triangle_valid_frac",
    "fake_mean",
    "fake_std",
    "fake_min",
    "fake_max",
    "fake_near_zero_frac",
    "fake_near_one_frac",
    "rounded_uniqueness_fake",
    "grad_norm",
    "ab_mean",
    "bc_mean",
    "cd_mean",
    "da_mean",
    "ac_mean",
    "bd_mean",
    "ab_std",
    "bc_std",
    "cd_std",
    "da_std",
    "ac_std",
    "bd_std",
]

def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"logs/triangle_gen_quantum_{timestamp}")
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = QGenerator(n_layer=2, seed=27)
    gen.weights = gen.rng.normal(0.0, 3.0, size=gen.weights.shape)
    weights = pnp.array(gen.weights, dtype=float, requires_grad=True)

    fixed_noise = pnp.array([
        [2.828448, 2.872291, 5.939522, 3.379326, 0.392954, 0.189158],
        [2.565836, 2.577120, 1.326812, 0.109693, 5.580918, 1.951064],
        [5.964916, 1.630786, 0.063053, 3.518623, 1.826606, 3.006534],
    ], dtype=float)

    real_mean, real_std = load_real_edge_stats(args.valid_tuples)
    """
    gen = QGenerator(n_layer=args.layers, seed=args.seed)
    if args.init_std is not None:
        gen.weights = gen.rng.normal(0.0, args.init_std, size=gen.weights.shape)
        print(f"[INFO] Reinitialized quantum generator weights with std={args.init_std}")

    real_mean, real_std = load_real_edge_stats(args.valid_tuples)

    weights = pnp.array(gen.weights, dtype=float, requires_grad=True)
    """
    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        

    def triangle_loss_fn(w, noise_batch):
        fake = gen.batch_forward(noise_batch, weights=w)
        return triangle_penalty_q(fake)

    def dist_loss_fn(w, noise_batch):
        fake = gen.batch_forward(noise_batch, weights=w)
        return distribution_match_loss_q(fake, real_mean, real_std)

    def total_loss_fn(w, noise_batch):
        #tri = triangle_loss_fn(w, noise_batch)
        #dist = dist_loss_fn(w, noise_batch)
        #return dist + args.lambda_dist * tri
        return triangle_loss_fn(w, noise_batch)

    grad_fn = qml.grad(total_loss_fn)

    best_valid_frac = 0.0
    last_metrics: Dict[str, float] = {}

    print(f"[INFO] Output dir: {output_dir}")
    print("[INFO] Training quantum generator with triangle + distribution loss")
    if args.latent_distribution == "uniform_pm1":
        print(
            "[WARN] latent_distribution=uniform_pm1 feeds full-period angles in [0, 2pi] directly into "
            "the generator. For this architecture that centers raw outputs near 0.5 by symmetry. "
            "Use --latent-distribution uniform unless you intentionally want a phase experiment."
        )

    for step in range(1, args.steps + 1):
        #noise_batch = sample_latent(rng, args.batch_size, args.latent_dim, args.latent_distribution)
        noise_batch = fixed_noise
        triangle_loss = triangle_loss_fn(weights, noise_batch)
        #dist_loss = dist_loss_fn(weights, noise_batch)
        #loss = triangle_loss + args.lambda_dist * dist_loss
        #loss = dist_loss + args.lambda_dist * triangle_loss
        loss = triangle_loss
        dist_loss = 0.0

        """
        grad = grad_fn(weights, noise_batch)
        grad_norm = float(pnp.sqrt(pnp.sum(grad ** 2)))

        weights = weights - args.lr * grad
        """
        grad = grad_fn(weights, noise_batch)

        # Falls PennyLane ein Tupel zurückgibt: erster Eintrag = Gradient nach weights
        if isinstance(grad, tuple):
            grad = grad[0]

        grad_norm = float(pnp.sqrt(pnp.sum(grad ** 2)))
        weights = weights - args.lr * grad
        gen.weights = np.array(weights, dtype=float)

        #noise_eval = sample_latent(rng, args.eval_batch_size, args.latent_dim, args.latent_distribution)
        #fake_eval = np.asarray(gen.batch_forward(noise_eval), dtype=float)
        fake_eval = np.asarray(gen.batch_forward(fixed_noise, weights=weights), dtype=float)

        metrics = compute_metrics(fake_eval)
        metrics["step"] = step
        metrics["loss"] = float(loss)
        metrics["triangle_loss"] = float(triangle_loss)
        metrics["dist_loss"] = float(dist_loss)
        metrics["grad_norm"] = grad_norm

        last_metrics = metrics
        best_valid_frac = max(best_valid_frac, metrics["triangle_valid_frac"])

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writerow({col: last_metrics.get(col, float("nan")) for col in csv_columns})

        if step == 1 or step % 100 == 0:
            print(
                f"[Step {step:5d}] loss={metrics['loss']:.6f} | "
                f"penalty={metrics['triangle_penalty']:.6f} | "
                f"triangle_loss={metrics['triangle_loss']:.6f} | "
                f"dist_loss={metrics['dist_loss']:.6f} | "
                f"valid_frac={metrics['triangle_valid_frac']:.4f} | "
                f"fake_mean={metrics['fake_mean']:.4f} | "
                f"fake_std={metrics['fake_std']:.4f} | "
                f"grad_norm={metrics['grad_norm']:.6f}"
            )

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            np.save(output_dir / f"generator_weights_step_{step:06d}.npy", np.array(weights, dtype=float))

    np.save(output_dir / "generator_weights_final.npy", np.array(weights, dtype=float))

        # Final samples for inspection
    noise_final = sample_latent(
        rng,
        args.eval_batch_size,
        args.latent_dim,
        args.latent_distribution,
    )
    fake_final = np.asarray(gen.batch_forward(noise_final), dtype=float)
    real_final = load_real_samples(args.valid_tuples, args.eval_batch_size, rng)

    np.save(output_dir / "fake_samples_final.npy", fake_final)
    np.save(output_dir / "real_samples_final.npy", real_final)

    make_per_edge_distribution_plot(
        real_final,
        fake_final,
        output_dir / "per_edge_distributions.png",
    )

    summary = Summary(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        latent_distribution=args.latent_distribution,
        n_layers=args.layers,
        init_std=args.init_std,
        final_loss=float(last_metrics["loss"]),
        final_triangle_penalty=float(last_metrics["triangle_penalty"]),
        final_triangle_valid_frac=float(last_metrics["triangle_valid_frac"]),
        final_fake_mean=float(last_metrics["fake_mean"]),
        final_fake_std=float(last_metrics["fake_std"]),
        best_triangle_valid_frac=float(best_valid_frac),
    )
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    make_plot(csv_path, output_dir / "training_plot.png")
    print("\n=== FINAL SUMMARY ===")
    print(json.dumps(asdict(summary), indent=2))
    print(f"[INFO] Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
