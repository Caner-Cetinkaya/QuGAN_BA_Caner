"""
Hybrid Quantum GAN with WGAN-GP training (v2).

Changes vs v1:
  - Loss function: BCE -> Wasserstein with gradient penalty (Gulrajani et al. 2017)
  - Discriminator is now a Critic: outputs unbounded real scores, NOT probabilities.
  - n_critic > 1: critic is updated multiple times per generator update.
  - Adam betas changed to (0.0, 0.9) per WGAN-GP recommendation.
  - Default learning rates set to 1e-4 for both (no asymmetric LR hack).
  - Default init_std raised to 0.3 to start outside the small-angle regime.
  - CSV column 'gp' (gradient penalty) added; 'real_score'/'fake_score_*'
    now contain raw critic outputs (no sigmoid).
  - Training plot updated: no ylim(0,1) on score panel, no chance-level line.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from config import MAX_EDGE_LENGTH_KM
except Exception:
    MAX_EDGE_LENGTH_KM = 5000.0


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """
    Classical Critic for WGAN-GP.

    Returns raw real-valued scores (NOT probabilities). Do NOT apply sigmoid.
    Do NOT use BatchNorm (forbidden in WGAN-GP); LeakyReLU is fine.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QGenerator(nn.Module):
    """
    Torch-compatible 6-qubit quantum generator.

    Output is shape (batch, 6), values in [0, 1].
    Each edge is p(|1>) from qml.probs(wires=i).
    """

    def __init__(self, n_layer: int = 2, init_std: float = 0.3, seed: int = 0):
        super().__init__()
        self.n_qubits = 6
        self.n_layer = n_layer
        self.rng = np.random.default_rng(seed)
        self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=None)

        init = torch.tensor(
            self.rng.normal(0.0, init_std, size=(n_layer, self.n_qubits, 3)),
            dtype=torch.float32,
        )
        self.weights = nn.Parameter(init)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(noise_vector: torch.Tensor, weights: torch.Tensor):
            qml.AngleEmbedding(noise_vector, wires=range(self.n_qubits), rotation="Y")

            for layer in range(self.n_layer):
                for qubit in range(self.n_qubits):
                    qml.RX(weights[layer, qubit, 0], wires=qubit)
                    qml.RY(weights[layer, qubit, 1], wires=qubit)
                    qml.RZ(weights[layer, qubit, 2], wires=qubit)

                for qubit in range(self.n_qubits):
                    qml.CNOT(wires=[qubit, (qubit + 1) % self.n_qubits])

            return [qml.probs(wires=i) for i in range(self.n_qubits)]
        
        self.circuit = circuit
        self.output_scale = nn.Parameter(torch.ones(6) * 3.0)
        self.output_bias  = nn.Parameter(torch.full((6,), -1.5))
    def forward(self, noise_batch: torch.Tensor) -> torch.Tensor:
        noise_batch = noise_batch.to(dtype=torch.float32)

        # Handle single-sample case
        single_sample = noise_batch.ndim == 1
        if single_sample:
            noise_batch = noise_batch.unsqueeze(0)

        # ONE batched QNode call instead of a Python loop
        probs_values = self.circuit(noise_batch, self.weights)

        # Output shape depends on PennyLane version:
        # - list of n_qubits tensors, each (batch, 2), OR
        # - tensor of shape (n_qubits, batch, 2)
        if isinstance(probs_values, (list, tuple)):
            stacked = torch.stack(
                [p.to(dtype=torch.float32) for p in probs_values], dim=0
            )  # (n_qubits, batch, 2)
        else:
            stacked = probs_values.to(dtype=torch.float32)

        # p(|1>) per qubit, then (n_qubits, batch) -> (batch, n_qubits)
        output = stacked[..., 1].transpose(0, 1).contiguous()

        if single_sample:
            output = output.squeeze(0)
        # raw hat Shape (batch, 6), Werte in [0,1]
        return torch.sigmoid(output * self.output_scale + self.output_bias)
        #return output
    """
    def _single_forward(self, noise_vector: torch.Tensor) -> torch.Tensor:
        probs_values = self.circuit(noise_vector, self.weights)

        if isinstance(probs_values, (list, tuple)):
            return torch.stack([p[1] for p in probs_values], dim=0).to(dtype=torch.float32)

        if probs_values.ndim == 2 and probs_values.shape == (self.n_qubits, 2):
            return probs_values[:, 1].to(dtype=torch.float32)

        raise ValueError(f"Unexpected QNode output shape: {tuple(probs_values.shape)}")

    def forward(self, noise_batch: torch.Tensor) -> torch.Tensor:
        noise_batch = noise_batch.to(dtype=torch.float32)

        if noise_batch.ndim == 1:
            return self._single_forward(noise_batch)

        outputs = [self._single_forward(noise_batch[i]) for i in range(noise_batch.shape[0])]
        return torch.stack(outputs, dim=0)
    """

# ---------------------------------------------------------------------------
# Summary dataclass
# ---------------------------------------------------------------------------

@dataclass
class Summary:
    steps: int
    batch_size: int
    g_lr: float
    d_lr: float
    n_critic: int
    lambda_gp: float
    latent_distribution: str
    n_layers: int
    init_std: float
    final_d_loss: float
    final_g_loss: float
    final_gp: float
    final_real_score: float
    final_fake_score_d: float
    final_fake_score_g: float
    final_d_grad_norm: float
    final_g_grad_norm: float
    final_fake_mean: float
    final_fake_std: float


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid WGAN-GP: QGEN + classical Critic, with training plot."
    )
    parser.add_argument("--steps", type=int, default=3000,
                        help="Number of generator updates (outer loop).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--g-lr", type=float, default=1e-4,
                        help="WGAN-GP standard: 1e-4 for both G and D.")
    parser.add_argument("--d-lr", type=float, default=1e-4)
    parser.add_argument("--n-critic", type=int, default=5,
                        help="Critic updates per generator update (WGAN standard: 5).")
    parser.add_argument("--lambda-gp", type=float, default=10.0,
                        help="Gradient penalty coefficient (WGAN-GP standard: 10).")
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument(
        "--latent-distribution",
        type=str,
        default="uniform",
        choices=["uniform", "normal", "angle", "uniform_pm1"],
    )
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-std", type=float, default=0.3,
                        help="Std of normal init for quantum weights. >=0.3 avoids tiny-gradient start.")
    parser.add_argument("--valid-tuples", type=str, default="valid_tuples.csv")
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--smooth-window", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGE_COLS = ["e_ab", "e_bc", "e_cd", "e_da", "e_ac", "e_bd"]
EDGE_NAMES = ["ab", "bc", "cd", "da", "ac", "bd"]

CSV_COLUMNS = [
    "step",
    "d_loss",
    "g_loss",
    "gp",
    "real_score",
    "fake_score_d",
    "fake_score_g",
    "d_grad_norm",
    "g_grad_norm",
    "fake_mean",
    "fake_std",
    "ab_mean", "bc_mean", "cd_mean", "da_mean", "ac_mean", "bd_mean",
    "ab_std", "bc_std", "cd_std", "da_std", "ac_std", "bd_std",
    "real_e_ab", "real_e_bc", "real_e_cd", "real_e_da", "real_e_ac", "real_e_bd",
    "fake_e_ab", "fake_e_bc", "fake_e_cd", "fake_e_da", "fake_e_ac", "fake_e_bd",
]


# ---------------------------------------------------------------------------
# Data + sampling helpers
# ---------------------------------------------------------------------------

def load_real_data(valid_tuples_path: str) -> np.ndarray:
    df = pd.read_csv(valid_tuples_path)
    missing = [col for col in EDGE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {valid_tuples_path}: {missing}")

    real = df[EDGE_COLS].to_numpy(dtype=np.float32)
    real = np.clip(real / MAX_EDGE_LENGTH_KM, 0.0, 1.0)
    return real


def sample_real_batch(real_data: np.ndarray, batch_size: int, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    idx = rng.choice(len(real_data), size=batch_size, replace=True)
    return torch.tensor(real_data[idx], dtype=torch.float32, device=device)


def sample_latent_torch(batch_size: int, latent_dim: int, distribution: str, device: torch.device) -> torch.Tensor:
    distribution = distribution.lower().strip()
    if distribution == "uniform":
        return torch.rand(batch_size, latent_dim, device=device)
    if distribution == "normal":
        return torch.randn(batch_size, latent_dim, device=device)
    if distribution in {"angle", "uniform_pm1"}:
        return 2.0 * torch.pi * torch.rand(batch_size, latent_dim, device=device)
    raise ValueError(f"Unknown latent distribution: {distribution!r}")


def grad_norm_torch(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(torch.sum(p.grad.detach() ** 2).item())
    return total ** 0.5


def compute_batch_stats(fake_batch: torch.Tensor) -> Dict[str, float]:
    arr = fake_batch.detach().cpu().numpy()
    stats: Dict[str, float] = {
        "fake_mean": float(arr.mean()),
        "fake_std": float(arr.std()),
    }
    for i, edge in enumerate(EDGE_NAMES):
        stats[f"{edge}_mean"] = float(arr[:, i].mean())
        stats[f"{edge}_std"] = float(arr[:, i].std())
    return stats


# ---------------------------------------------------------------------------
# WGAN-GP: gradient penalty
# ---------------------------------------------------------------------------

def gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Two-sided gradient penalty from Gulrajani et al. 2017.

    Samples random interpolations between real and fake (both detached),
    computes the critic's gradient w.r.t. the input at those points, and
    penalizes deviation of the L2 norm from 1.
    """
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    # broadcast alpha to data dims
    alpha = alpha.expand_as(real)

    interp = alpha * real + (1.0 - alpha) * fake
    interp.requires_grad_(True)

    d_interp = critic(interp)

    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grads = grads.view(batch_size, -1)
    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def smooth(y, window: int = 10) -> np.ndarray:
    y = np.asarray(y, dtype=float)

    if window <= 1 or len(y) < 2:
        return y

    return (
        pd.Series(y)
        .rolling(window=window, min_periods=1, center=True)
        .mean()
        .to_numpy()
    )


def make_training_plot(csv_path: Path, out_path: Path, smooth_window: int = 10) -> None:
    df = pd.read_csv(csv_path)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["step"]).sort_values("step")

    steps = df["step"].to_numpy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(
        f"Hybrid WGAN-GP Training: QGEN + Critic (smoothed over {smooth_window} steps)",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot(steps, df["d_loss"], alpha=0.25, label="Raw")
    ax.plot(steps, smooth(df["d_loss"], smooth_window), linewidth=2, label="Smoothed")
    ax.set_title("Critic Loss (lower = better separation)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(steps, df["g_loss"], alpha=0.25, label="Raw")
    ax.plot(steps, smooth(df["g_loss"], smooth_window), linewidth=2, label="Smoothed")
    ax.set_title("Generator Loss (-E[D(fake)])")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 2]
    # Wasserstein estimate ~ E[D(real)] - E[D(fake)]
    wdist = df["real_score"] - df["fake_score_d"]
    ax.plot(steps, wdist, alpha=0.25, label="Raw")
    ax.plot(steps, smooth(wdist, smooth_window), linewidth=2, label="Smoothed")
    ax.axhline(0.0, linestyle="--", alpha=0.5, color="gray")
    ax.set_title("Wasserstein Estimate (E[D(real)] - E[D(fake)])")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance estimate")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(steps, smooth(df["real_score"], smooth_window), label="Critic(real)")
    ax.plot(steps, smooth(df["fake_score_d"], smooth_window), label="Critic(fake)")
    ax.set_title("Critic Output Scores (raw, unbounded)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    if "gp" in df.columns:
        ax.plot(steps, df["gp"], alpha=0.25, label="Raw")
        ax.plot(steps, smooth(df["gp"], smooth_window), linewidth=2, label="Smoothed")
        ax.axhline(0.0, linestyle="--", alpha=0.5, color="gray",
                   label="ideal (||grad||=1)")
    ax.set_title("Gradient Penalty (should approach 0)")
    ax.set_xlabel("Step")
    ax.set_ylabel("GP value")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 2]
    ax.plot(steps, smooth(df["d_grad_norm"], smooth_window), label="D Grad Norm")
    ax.plot(steps, smooth(df["g_grad_norm"], smooth_window), label="G Grad Norm")
    ax.set_title("Parameter Gradient Magnitudes")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient norm")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_per_edge_distribution_plot(real_batch: np.ndarray, fake_batch: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Hybrid GAN: Per-edge distributions", fontsize=16, fontweight="bold")

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


def generate_fake_samples(
    gen: QGenerator,
    num_samples: int,
    batch_size: int,
    latent_dim: int,
    latent_distribution: str,
    device: torch.device,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    remaining = num_samples
    gen.eval()
    while remaining > 0:
        bs = min(batch_size, remaining)
        z = sample_latent_torch(bs, latent_dim, latent_distribution, device)
        fake = gen(z).detach().cpu().numpy()
        outputs.append(fake)
        remaining -= bs
    return np.concatenate(outputs, axis=0)


# ---------------------------------------------------------------------------
# Main training loop (WGAN-GP)
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device)

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"logs/hybrid_qgen_cdisc_wgangp_{timestamp}")
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Training hybrid WGAN-GP: QGenerator + classical Critic")
    print(f"[INFO] n_critic={args.n_critic}, lambda_gp={args.lambda_gp}, "
          f"g_lr={args.g_lr}, d_lr={args.d_lr}, init_std={args.init_std}")

    real_data = load_real_data(args.valid_tuples)

    gen = QGenerator(n_layer=args.layers, init_std=args.init_std, seed=args.seed).to(device)
    disc = Discriminator(hidden_dim=args.hidden_dim).to(device)

    # WGAN-GP standard: betas=(0.0, 0.9)
    g_optimizer = optim.Adam(gen.parameters(), lr=args.g_lr, betas=(0.0, 0.9))
    d_optimizer = optim.Adam(disc.parameters(), lr=args.d_lr, betas=(0.0, 0.9))

    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

    # Dump full args as JSON for reproducibility
    with open(output_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()}, f, indent=2)

    real_used_samples = []
    fake_used_samples = []
    last_metrics: Dict[str, float] = {}

    for step in range(1, args.steps + 1):
        # =========================================================
        # CRITIC UPDATES (n_critic times per generator step)
        # =========================================================
        disc.train()

        # We track the last critic update's values for logging.
        last_d_loss = 0.0
        last_gp = 0.0
        last_real_score = 0.0
        last_fake_score_d = 0.0
        last_real_batch = None
        last_fake_batch_d = None
        last_d_grad_norm = 0.0

        for _ in range(args.n_critic):
            d_optimizer.zero_grad(set_to_none=True)

            real_batch = sample_real_batch(real_data, args.batch_size, rng, device)
            z_d = sample_latent_torch(args.batch_size, args.latent_dim,
                                      args.latent_distribution, device)
            with torch.no_grad():
                # No need to backprop through G during critic updates.
                fake_batch_d = gen(z_d)

            real_logits = disc(real_batch)
            fake_logits_d = disc(fake_batch_d)

            # WGAN loss: maximize D(real) - D(fake)  <=>  minimize -[D(real) - D(fake)]
            d_loss_wass = fake_logits_d.mean() - real_logits.mean()
            gp = gradient_penalty(disc, real_batch, fake_batch_d, device)
            d_loss = d_loss_wass + args.lambda_gp * gp

            d_loss.backward()
            d_grad_norm_val = grad_norm_torch(disc.parameters())
            d_optimizer.step()

            # cache for logging
            last_d_loss = float(d_loss.item())
            last_gp = float(gp.item())
            last_real_score = float(real_logits.mean().item())
            last_fake_score_d = float(fake_logits_d.mean().item())
            last_real_batch = real_batch
            last_fake_batch_d = fake_batch_d
            last_d_grad_norm = float(d_grad_norm_val)

        # Keep samples for later distribution plotting (one batch per outer step)
        real_used_samples.append(last_real_batch.detach().cpu().numpy())
        fake_used_samples.append(last_fake_batch_d.detach().cpu().numpy())

        # =========================================================
        # GENERATOR UPDATE (once per outer step)
        # =========================================================
        g_optimizer.zero_grad(set_to_none=True)

        z_g = sample_latent_torch(args.batch_size, args.latent_dim,
                                  args.latent_distribution, device)
        fake_batch_g = gen(z_g)
        fake_logits_g = disc(fake_batch_g)

        # WGAN generator loss: maximize D(fake)  <=>  minimize -D(fake)
        g_loss = -fake_logits_g.mean()

        g_loss.backward()
        g_grad_norm = grad_norm_torch(gen.parameters())
        g_optimizer.step()

        # =========================================================
        # Logging
        # =========================================================
        with torch.no_grad():
            fake_score_g = float(fake_logits_g.mean().item())
            stats = compute_batch_stats(fake_batch_g)

        metrics = {
            "step": step,
            "d_loss": last_d_loss,
            "g_loss": float(g_loss.item()),
            "gp": last_gp,
            "real_score": last_real_score,
            "fake_score_d": last_fake_score_d,
            "fake_score_g": fake_score_g,
            "d_grad_norm": last_d_grad_norm,
            "g_grad_norm": float(g_grad_norm),
            **stats,
            "real_e_ab": float(last_real_batch[:, 0].mean().item()),
            "real_e_bc": float(last_real_batch[:, 1].mean().item()),
            "real_e_cd": float(last_real_batch[:, 2].mean().item()),
            "real_e_da": float(last_real_batch[:, 3].mean().item()),
            "real_e_ac": float(last_real_batch[:, 4].mean().item()),
            "real_e_bd": float(last_real_batch[:, 5].mean().item()),

            "fake_e_ab": float(last_fake_batch_d[:, 0].mean().item()),
            "fake_e_bc": float(last_fake_batch_d[:, 1].mean().item()),
            "fake_e_cd": float(last_fake_batch_d[:, 2].mean().item()),
            "fake_e_da": float(last_fake_batch_d[:, 3].mean().item()),
            "fake_e_ac": float(last_fake_batch_d[:, 4].mean().item()),
            "fake_e_bd": float(last_fake_batch_d[:, 5].mean().item()),
        }
        last_metrics = metrics

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow(metrics)

        if step == 1 or step % 100 == 0:
            wdist = metrics["real_score"] - metrics["fake_score_d"]
            print(
                f"[Step {step:5d}] "
                f"d_loss={metrics['d_loss']:.4f} | "
                f"g_loss={metrics['g_loss']:.4f} | "
                f"gp={metrics['gp']:.4f} | "
                f"W_est={wdist:+.4f} | "
                f"D(real)={metrics['real_score']:+.3f} | "
                f"D(fake)={metrics['fake_score_d']:+.3f} | "
                f"d_grad={metrics['d_grad_norm']:.3f} | "
                f"g_grad={metrics['g_grad_norm']:.3f} | "
                f"fake_mean={metrics['fake_mean']:.3f} | "
                f"fake_std={metrics['fake_std']:.3f}"
            )

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            torch.save(gen.state_dict(), output_dir / f"qgen_step_{step:06d}.pt")
            torch.save(disc.state_dict(), output_dir / f"cdisc_step_{step:06d}.pt")

    # =========================================================
    # Post-training: eval, save, plot
    # =========================================================
    fake_final = generate_fake_samples(
        gen=gen,
        num_samples=args.eval_samples,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        latent_distribution=args.latent_distribution,
        device=device,
    )
    real_final = sample_real_batch(real_data, args.eval_samples, rng, device).cpu().numpy()

    np.save(output_dir / "fake_samples_final.npy", fake_final)
    np.save(output_dir / "real_samples_final.npy", real_final)

    real_used = np.concatenate(real_used_samples, axis=0)
    fake_used = np.concatenate(fake_used_samples, axis=0)

    make_training_plot(
        csv_path,
        output_dir / "training_plot.png",
        smooth_window=args.smooth_window,
    )

    make_per_edge_distribution_plot(
        real_final,
        fake_final,
        output_dir / "per_edge_distributions_final.png",
    )

    make_per_edge_distribution_plot(
        real_used,
        fake_used,
        output_dir / "per_edge_distributions_training_samples.png",
    )

    summary = Summary(
        steps=args.steps,
        batch_size=args.batch_size,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        n_critic=args.n_critic,
        lambda_gp=args.lambda_gp,
        latent_distribution=args.latent_distribution,
        n_layers=args.layers,
        init_std=args.init_std,
        final_d_loss=float(last_metrics["d_loss"]),
        final_g_loss=float(last_metrics["g_loss"]),
        final_gp=float(last_metrics["gp"]),
        final_real_score=float(last_metrics["real_score"]),
        final_fake_score_d=float(last_metrics["fake_score_d"]),
        final_fake_score_g=float(last_metrics["fake_score_g"]),
        final_d_grad_norm=float(last_metrics["d_grad_norm"]),
        final_g_grad_norm=float(last_metrics["g_grad_norm"]),
        final_fake_mean=float(last_metrics["fake_mean"]),
        final_fake_std=float(last_metrics["fake_std"]),
    )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    torch.save(gen.state_dict(), output_dir / "qgen_final.pt")
    torch.save(disc.state_dict(), output_dir / "cdisc_final.pt")

    print("\n=== FINAL SUMMARY ===")
    print(json.dumps(asdict(summary), indent=2))
    print(f"[INFO] Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()