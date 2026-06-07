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


class Discriminator(nn.Module):
    """
    Returns a single logit per sample, representing the probability of being real (close to 1) or fake (close to 0)
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
    6-qubit quantum generator:
    - AngleEmbedding on Y
    - RX / RY / RZ trainable layers
    - circular CNOT + skip-1 CNOT
    - measurement: qml.probs(wires=i), edge_i = p(|1>)
    """

    def __init__(self, n_layer: int = 2, init_std: float = 1.0, seed: int = 0):
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

    # verarbeitet einen einzigen Noise-Vektor und gibt die 6 edge-Wahrscheinlichkeiten zurück
    def _single_forward(self, noise_vector: torch.Tensor) -> torch.Tensor:
        probs_values = self.circuit(noise_vector, self.weights)

        if isinstance(probs_values, (list, tuple)):
            edge_values = torch.stack([p[1] for p in probs_values], dim=0)
            return edge_values.to(dtype=torch.float32)

        probs_tensor = probs_values
        if probs_tensor.ndim == 2 and probs_tensor.shape == (self.n_qubits, 2):
            return probs_tensor[:, 1].to(dtype=torch.float32)

        raise ValueError(f"Unexpected probs output shape: {tuple(probs_tensor.shape)}")

    # verarbeitet einen Batch von Noise-Vektoren, indem er sie einzeln durch die Schaltung schickt und die Ergebnisse stapelt
    def forward(self, noise_batch: torch.Tensor) -> torch.Tensor:
        noise_batch = noise_batch.to(dtype=torch.float32)

        if noise_batch.ndim == 1:
            return self._single_forward(noise_batch)

        outputs: List[torch.Tensor] = []
        for i in range(noise_batch.shape[0]):
            outputs.append(self._single_forward(noise_batch[i]))
        return torch.stack(outputs, dim=0)


@dataclass
class Summary:
    steps: int
    batch_size: int
    g_lr: float
    d_lr: float
    latent_distribution: str
    n_layers: int
    init_std: float
    final_d_loss: float
    final_g_loss: float
    final_real_score: float
    final_fake_score: float
    final_fake_mean: float
    final_fake_std: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid GAN: quantum generator + classical discriminator on real 4-city edge data."
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--g-lr", type=float, default=0.01)
    parser.add_argument("--d-lr", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--latent-distribution", type=str, default="uniform", choices=["uniform", "normal", "angle"])
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-std", type=float, default=1.0)
    parser.add_argument("--valid-tuples", type=str, default="valid_tuples.csv")
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()

# Erzeugt Zufallsrauschen für den Generator, entweder gleichverteilt, normalverteilt oder als Winkel (0 bis 2π)
def sample_latent_torch(
    batch_size: int,
    latent_dim: int,
    distribution: str,
    device: torch.device,
) -> torch.Tensor:
    distribution = distribution.lower().strip()
    if distribution == "uniform":
        return torch.rand(batch_size, latent_dim, device=device)
    if distribution == "normal":
        return torch.randn(batch_size, latent_dim, device=device)
    if distribution == "angle":
        return 2.0 * torch.pi * torch.rand(batch_size, latent_dim, device=device)
    raise ValueError(f"Unknown latent distribution: {distribution!r}")


def load_real_data(valid_tuples_path: str) -> np.ndarray:
    df = pd.read_csv(valid_tuples_path)
    needed = ["e_ab", "e_bc", "e_cd", "e_da", "e_ac", "e_bd"]

    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r} in {valid_tuples_path}")

    real = df[needed].to_numpy(dtype=np.float32)
    real = np.clip(real / MAX_EDGE_LENGTH_KM, 0.0, 1.0)
    return real


def sample_real_batch(
    real_data: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    idx = rng.choice(len(real_data), size=batch_size, replace=True)
    return torch.tensor(real_data[idx], dtype=torch.float32, device=device)


def compute_batch_stats(fake_batch: torch.Tensor) -> Dict[str, float]:
    fake_np = fake_batch.detach().cpu().numpy()
    return {
        "fake_mean": float(fake_np.mean()),
        "fake_std": float(fake_np.std()),
        "ab_mean": float(fake_np[:, 0].mean()),
        "bc_mean": float(fake_np[:, 1].mean()),
        "cd_mean": float(fake_np[:, 2].mean()),
        "da_mean": float(fake_np[:, 3].mean()),
        "ac_mean": float(fake_np[:, 4].mean()),
        "bd_mean": float(fake_np[:, 5].mean()),
        "ab_std": float(fake_np[:, 0].std()),
        "bc_std": float(fake_np[:, 1].std()),
        "cd_std": float(fake_np[:, 2].std()),
        "da_std": float(fake_np[:, 3].std()),
        "ac_std": float(fake_np[:, 4].std()),
        "bd_std": float(fake_np[:, 5].std()),
    }


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


def make_training_plot(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["step"]).sort_values("step")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Hybrid GAN: QGEN + cDISC", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(df["step"], df["d_loss"], label="d_loss")
    ax.plot(df["step"], df["g_loss"], label="g_loss")
    ax.set_title("Adversarial losses")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(df["step"], df["real_score"], label="real_score")
    ax.plot(df["step"], df["fake_score"], label="fake_score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Discriminator scores")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(df["step"], df["fake_mean"], label="fake_mean")
    ax.plot(df["step"], df["fake_std"], label="fake_std")
    ax.set_title("Global fake stats")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    for name in ["ab_mean", "bc_mean", "cd_mean", "da_mean", "ac_mean", "bd_mean"]:
        ax.plot(df["step"], df[name], label=name)
    ax.set_title("Per-edge means")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_per_edge_distribution_plot(real_batch: np.ndarray, fake_batch: np.ndarray, out_path: Path) -> None:
    edge_names = ["ab", "bc", "cd", "da", "ac", "bd"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Hybrid GAN: Per-edge distributions", fontsize=16, fontweight="bold")

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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cpu")
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"logs/hybrid_qgen_cdisc_{timestamp}")
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Device: {device}")
    print("[INFO] Training hybrid GAN: QGenerator + classical Discriminator")

    real_data = load_real_data(args.valid_tuples)

    gen = QGenerator(
        n_layer=args.layers,
        init_std=args.init_std,
        seed=args.seed,
    ).to(device)

    disc = Discriminator(hidden_dim=args.hidden_dim).to(device)

    g_optimizer = optim.Adam(gen.parameters(), lr=args.g_lr)
    d_optimizer = optim.Adam(disc.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    csv_columns = [
        "step",
        "d_loss",
        "g_loss",
        "real_score",
        "fake_score",
        "fake_mean",
        "fake_std",
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

    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

    last_metrics: Dict[str, float] = {}

    for step in range(1, args.steps + 1):
        disc.train()
        d_optimizer.zero_grad(set_to_none=True)

        real_batch = sample_real_batch(real_data, args.batch_size, rng, device)
        z = sample_latent_torch(args.batch_size, args.latent_dim, args.latent_distribution, device)
        fake_batch = gen(z)

        real_logits = disc(real_batch)
        fake_logits = disc(fake_batch.detach())

        real_targets = torch.ones_like(real_logits)
        fake_targets = torch.zeros_like(fake_logits)

        d_loss_real = criterion(real_logits, real_targets)
        d_loss_fake = criterion(fake_logits, fake_targets)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        d_loss.backward()
        d_optimizer.step()

        g_optimizer.zero_grad(set_to_none=True)

        z = sample_latent_torch(args.batch_size, args.latent_dim, args.latent_distribution, device)
        fake_batch = gen(z)
        fake_logits_for_g = disc(fake_batch)

        g_targets = torch.ones_like(fake_logits_for_g)
        g_loss = criterion(fake_logits_for_g, g_targets)

        g_loss.backward()
        #print("qgen grad norm:", gen.weights.grad.norm().item())
        g_optimizer.step()

        with torch.no_grad():
            real_score = torch.sigmoid(real_logits).mean().item()
            fake_score = torch.sigmoid(fake_logits).mean().item()
            stats = compute_batch_stats(fake_batch)

        metrics = {
            "step": step,
            "d_loss": float(d_loss.item()),
            "g_loss": float(g_loss.item()),
            "real_score": float(real_score),
            "fake_score": float(fake_score),
            **stats,
        }
        last_metrics = metrics

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writerow(metrics)

        if step == 1 or step % 100 == 0:
            print(
                f"[Step {step:5d}] "
                f"d_loss={metrics['d_loss']:.6f} | "
                f"g_loss={metrics['g_loss']:.6f} | "
                f"real_score={metrics['real_score']:.4f} | "
                f"fake_score={metrics['fake_score']:.4f} | "
                f"fake_mean={metrics['fake_mean']:.4f} | "
                f"fake_std={metrics['fake_std']:.4f}"
            )

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            torch.save(gen.state_dict(), output_dir / f"qgen_step_{step:06d}.pt")
            torch.save(disc.state_dict(), output_dir / f"cdisc_step_{step:06d}.pt")

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

    make_training_plot(csv_path, output_dir / "training_plot.png")
    make_per_edge_distribution_plot(real_final, fake_final, output_dir / "per_edge_distributions.png")

    summary = Summary(
        steps=args.steps,
        batch_size=args.batch_size,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        latent_distribution=args.latent_distribution,
        n_layers=args.layers,
        init_std=args.init_std,
        final_d_loss=float(last_metrics["d_loss"]),
        final_g_loss=float(last_metrics["g_loss"]),
        final_real_score=float(last_metrics["real_score"]),
        final_fake_score=float(last_metrics["fake_score"]),
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
