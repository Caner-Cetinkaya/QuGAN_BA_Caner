from __future__ import annotations

import argparse
import csv
from html import parser
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

# Reuse the user's existing classical generator implementation.
from config import MAX_EDGE_LENGTH_KM
from training_cgan import Generator


@dataclass
class Summary:
    steps: int
    batch_size: int
    learning_rate: float
    latent_distribution: str
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
    device: torch.device,
) -> torch.Tensor:
    distribution = distribution.lower().strip()
    if distribution == "uniform":
        z_np = rng.uniform(0.0, 1.0, size=(batch_size, latent_dim)).astype(np.float32)
        return torch.from_numpy(z_np).to(device)
    if distribution == "normal":
        z_np = rng.normal(0.0, 1.0, size=(batch_size, latent_dim)).astype(np.float32)
        return torch.from_numpy(z_np).to(device)
    raise ValueError(f"Unknown latent distribution: {distribution!r}")


def triangle_penalty_torch(batch: torch.Tensor) -> torch.Tensor:
    ab = batch[:, 0]
    bc = batch[:, 1]
    cd = batch[:, 2]
    da = batch[:, 3]
    ac = batch[:, 4]
    bd = batch[:, 5]

    violation = (
        # triangle abc
        F.relu(ab - (bc + ac)) +
        F.relu(bc - (ab + ac)) +
        F.relu(ac - (ab + bc)) +

        # triangle abd
        F.relu(ab - (da + bd)) +
        F.relu(da - (ab + bd)) +
        F.relu(bd - (ab + da)) +

        # triangle acd
        F.relu(ac - (cd + da)) +
        F.relu(cd - (ac + da)) +
        F.relu(da - (ac + cd)) +

        # triangle bcd
        F.relu(bc - (cd + bd)) +
        F.relu(cd - (bc + bd)) +
        F.relu(bd - (bc + cd))
    )

    return violation.mean() / 12.0


def triangle_valid_mask_torch(batch: torch.Tensor, tol: float = 1e-8) -> torch.Tensor:
    ab = batch[:, 0]
    bc = batch[:, 1]
    cd = batch[:, 2]
    da = batch[:, 3]
    ac = batch[:, 4]
    bd = batch[:, 5]

    def tri_ok(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
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

import pandas as pd

def load_real_edge_stats(valid_tuples_path: str):
    df = pd.read_csv(valid_tuples_path)

    needed = ["e_ab", "e_bc", "e_cd", "e_da", "e_ac", "e_bd"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r} in {valid_tuples_path}")

    real = df[needed].to_numpy(dtype=np.float32)
    real = np.clip(real / MAX_EDGE_LENGTH_KM, 0.0, 1.0)

    real_mean = torch.tensor(real.mean(axis=0), dtype=torch.float32)
    real_std = torch.tensor(real.std(axis=0), dtype=torch.float32)

    return real_mean, real_std


def distribution_match_loss_torch(
    fake_batch: torch.Tensor,
    real_mean: torch.Tensor,
    real_std: torch.Tensor,
) -> torch.Tensor:
    fake_mean = fake_batch.mean(dim=0)
    fake_std = fake_batch.std(dim=0, unbiased=False)

    mean_loss = F.mse_loss(fake_mean, real_mean.to(fake_batch.device))
    std_loss = F.mse_loss(fake_std, real_std.to(fake_batch.device))

    return mean_loss + std_loss


def compute_metrics(fake_batch: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        penalty = float(triangle_penalty_torch(fake_batch).item())
        valid_frac = float(triangle_valid_mask_torch(fake_batch).float().mean().item())
        fake_np = fake_batch.detach().cpu().numpy()
        metrics: Dict[str, float] = {
            "triangle_penalty": penalty,
            "triangle_valid_frac": valid_frac,
            "fake_mean": float(fake_batch.mean().item()),
            "fake_std": float(fake_batch.std().item()),
            "fake_min": float(fake_batch.min().item()),
            "fake_max": float(fake_batch.max().item()),
            "fake_near_zero_frac": float((fake_batch <= 0.05).float().mean().item()),
            "fake_near_one_frac": float((fake_batch >= 0.95).float().mean().item()),
            "rounded_uniqueness_fake": rounded_uniqueness_ratio(fake_np),
        }
        edge_names = ["ab", "bc", "cd", "da", "ac", "bd"]
        for i, name in enumerate(edge_names):
            metrics[f"{name}_mean"] = float(fake_np[:, i].mean())
            metrics[f"{name}_std"] = float(fake_np[:, i].std())
    return metrics


def make_plot(csv_path: Path, out_path: Path) -> None:
    import pandas as pd

    df = pd.read_csv(csv_path)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["step"]).sort_values("step")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Classical Generator: Triangle-Only Training", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(df["step"], df["loss"], label="loss")
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
    ax.plot(df["step"], df["rounded_uniqueness_fake"], label="rounded_uniqueness_fake")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Rounded uniqueness")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train only the classical generator to satisfy triangle inequality.")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--latent-distribution", type=str, default="uniform", choices=["uniform", "normal"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--valid-tuples", type=str, default="valid_tuples.csv")
    parser.add_argument("--lambda-dist", type=float, default=10.0)
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"logs/triangle_gen_classical_{timestamp}")
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    generator = Generator(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    real_mean, real_std = load_real_edge_stats(args.valid_tuples)

    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

    best_valid_frac = 0.0
    last_metrics: Dict[str, float] = {}

    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Training classical generator only for triangle inequality")

    for step in range(1, args.steps + 1):
        generator.train()
        optimizer.zero_grad(set_to_none=True)

        z = sample_latent(rng, args.batch_size, args.latent_dim, args.latent_distribution, device)
        fake = generator(z)

        triangle_loss = triangle_penalty_torch(fake)
        dist_loss = distribution_match_loss_torch(fake, real_mean, real_std)
        loss = triangle_loss + args.lambda_dist * dist_loss

        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10.0).item())
        optimizer.step()

        generator.eval()
        with torch.no_grad():
            z_eval = sample_latent(rng, args.eval_batch_size, args.latent_dim, args.latent_distribution, device)
            fake_eval = generator(z_eval)

            metrics = compute_metrics(fake_eval)
            metrics["step"] = step
            metrics["loss"] = float(loss.detach().item())
            metrics["grad_norm"] = grad_norm
            metrics["triangle_loss"] = float(triangle_loss.detach().item())
            metrics["dist_loss"] = float(dist_loss.detach().item())

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
                f"grad_norm={metrics['grad_norm']:.6f} | "
            )

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            torch.save(
                {"generator_state_dict": generator.state_dict(), "step": step},
                output_dir / f"generator_step_{step:06d}.pt",
            )

    torch.save(
        {"generator_state_dict": generator.state_dict(), "step": args.steps},
        output_dir / "generator_final.pt",
    )

    summary = Summary(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        latent_distribution=args.latent_distribution,
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
