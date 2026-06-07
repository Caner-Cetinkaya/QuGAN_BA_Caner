"""
Hybrid GAN: QGenerator + classical Discriminator — EXPERIMENT 1: BASELINE + MBD

Diese Variante baut auf der Baseline auf und fuegt Minibatch Discrimination hinzu:
- MIT Minibatch Discrimination im Discriminator (8 Features)
- KEIN trainierbarer Post-Processing-Adapter
- Label Smoothing aktiv (D und G beide gegen 0.9)
- (z+1)/2-Mapping (parameterfrei)
- Discriminator-Groesse beibehalten (hidden_dim=8 wie Baseline), mit MBD jetzt 14 Inputs

Ziel: pruefen, ob MBD allein den Generator stabilisiert und Mode Collapse
verhindert, ohne andere Aenderungen.
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


# =============================================================================
# Minibatch Discrimination
# =============================================================================
class MinibatchDiscrimination(nn.Module):
    """
    Gibt dem Discriminator ein Feature pro Sample, das misst,
    wie aehnlich das Sample zu den anderen Samples im Batch ist.
    Bestraft Mode Collapse direkt im Gradienten.
    """
    def __init__(self, in_features: int, out_features: int, kernel_dims: int):
        super().__init__()
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M = torch.einsum("bi,ijk->bjk", x, self.T)
        L1 = torch.abs(M.unsqueeze(0) - M.unsqueeze(1)).sum(dim=3)
        c = torch.exp(-L1)
        mask = 1 - torch.eye(x.size(0), device=x.device).unsqueeze(2)
        o_b = (c * mask).mean(dim=1)
        return torch.cat([x, o_b], dim=1)


# =============================================================================
# Classical Discriminator — MIT Minibatch Discrimination
# =============================================================================
class Discriminator(nn.Module):
    """Classical discriminator MIT MBD. Returns raw logits; use BCEWithLogitsLoss.

    Discriminator-Groesse beibehalten wie Baseline (hidden_dim=8),
    aber Input ist jetzt 6 + 8 (MBD out_features) = 14.
    """

    def __init__(self, hidden_dim: int = 8, mbd_out_features: int = 8):
        super().__init__()
        self.mbd = MinibatchDiscrimination(
            in_features=6, out_features=mbd_out_features, kernel_dims=3
        )
        self.net = nn.Sequential(
            nn.Linear(6 + mbd_out_features, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mbd(x)
        return self.net(x)


# =============================================================================
# Quantum Generator — KEIN Post-Processing-Adapter
# =============================================================================
class QGenerator(nn.Module):
    """
    6-Qubit Quantum-Generator mit echtem Batching.

    Input  forward(noise_batch): (batch, 6) oder (6,)
    Output:                      (batch, 6) oder (6,) in [0, 1]
                                 via (PauliZ_expval + 1) / 2
    """

    def __init__(self, n_layer: int = 4, init_std: float = 0.1, seed: int = 0):
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
        def circuit(noise_batch: torch.Tensor, weights: torch.Tensor):
            # XYZ-Encoding wie Projekt A (komplexer Initialzustand)
            qml.AngleEmbedding(noise_batch, wires=range(self.n_qubits), rotation="X")
            qml.AngleEmbedding(noise_batch, wires=range(self.n_qubits), rotation="Y")
            qml.AngleEmbedding(noise_batch, wires=range(self.n_qubits), rotation="Z")

            for layer in range(self.n_layer):
                for qubit in range(self.n_qubits):
                    qml.RX(weights[layer, qubit, 0], wires=qubit)
                    qml.RY(weights[layer, qubit, 1], wires=qubit)
                    qml.RZ(weights[layer, qubit, 2], wires=qubit)
                for qubit in range(self.n_qubits):
                    qml.CNOT(wires=[qubit, (qubit + 1) % self.n_qubits])

            # Pauli-Z-Erwartungswert je Qubit: Werte in [-1, 1]
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def forward(self, noise_batch: torch.Tensor) -> torch.Tensor:
        noise_batch = noise_batch.to(dtype=torch.float32)

        squeeze_output = False
        if noise_batch.ndim == 1:
            noise_batch = noise_batch.unsqueeze(0)
            squeeze_output = True

        # Ein einziger QNode-Call mit dem ganzen Batch
        z_vals = self.circuit(noise_batch, self.weights)
        z = torch.stack(
            [v if v.ndim > 0 else v.unsqueeze(0) for v in z_vals],
            dim=-1,
        ).to(dtype=torch.float32)

        # Lineare Range-Konvertierung [-1, 1] -> [0, 1] (parameterfrei, kein Postprocessing-NN)
        out = (z + 1.0) / 2.0

        if squeeze_output:
            out = out.squeeze(0)
        return out


@dataclass
class Summary:
    steps: int
    batch_size: int
    g_lr: float
    d_lr: float
    latent_distribution: str
    n_layers: int
    init_std: float
    label_smoothing: float
    final_d_loss: float
    final_g_loss: float
    final_real_score: float
    final_fake_score_d: float
    final_fake_score_g: float
    final_d_grad_norm: float
    final_g_grad_norm: float
    final_fake_mean: float
    final_fake_std: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EXP1: Hybrid GAN: QGEN + cDISC mit MBD, ohne post-adapter."
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--g-lr", type=float, default=1e-3)
    parser.add_argument("--d-lr", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument(
        "--latent-distribution",
        type=str,
        default="angle",
        choices=["uniform", "normal", "angle", "uniform_pm1"],
    )
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-std", type=float, default=0.1)
    parser.add_argument("--valid-tuples", type=str, default="valid_tuples.csv")
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--smooth-window", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--label-smoothing", type=float, default=0.1,
        help="Real-Target = 1 - smoothing (D und G), Fake-Target = smoothing (nur D).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


EDGE_COLS = ["e_ab", "e_bc", "e_cd", "e_da", "e_ac", "e_bd"]
EDGE_NAMES = ["ab", "bc", "cd", "da", "ac", "bd"]

CSV_COLUMNS = [
    "step",
    "d_loss",
    "g_loss",
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
        return torch.pi * torch.rand(batch_size, latent_dim, device=device)
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
        f"EXP1 (MBD) Hybrid GAN: QGEN + cDISC (Smoothed over {smooth_window} steps)",
        fontsize=16, fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot(steps, df["d_loss"], alpha=0.25, label="Raw")
    ax.plot(steps, smooth(df["d_loss"], smooth_window), linewidth=2, label="Smoothed")
    ax.set_title("Discriminator Loss"); ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[0, 1]
    ax.plot(steps, df["g_loss"], alpha=0.25, label="Raw")
    ax.plot(steps, smooth(df["g_loss"], smooth_window), linewidth=2, label="Smoothed")
    ax.set_title("Generator Loss"); ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[0, 2]
    ax.plot(steps, smooth(df["d_loss"], smooth_window), label="D Loss")
    ax.plot(steps, smooth(df["g_loss"], smooth_window), label="G Loss")
    ax.set_title("Loss Comparison (D vs G)"); ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1, 0]
    ax.plot(steps, smooth(df["real_score"], smooth_window), label="Real (should->1.0)")
    ax.plot(steps, smooth(df["fake_score_d"], smooth_window), label="Fake-Disc (should->0.0)")
    ax.axhline(0.5, linestyle="--", alpha=0.7, label="Chance level")
    ax.set_title("Discriminator Classification Scores"); ax.set_xlabel("Step")
    ax.set_ylabel("D output probability"); ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1, 1]
    ax.plot(steps, smooth(df["fake_score_g"], smooth_window), label="Fake-Gen (should->1.0)")
    ax.axhline(0.5, linestyle="--", alpha=0.7, label="Chance level")
    ax.set_title("Generator Fooling Success"); ax.set_xlabel("Step")
    ax.set_ylabel("D output probability"); ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1, 2]
    ax.plot(steps, smooth(df["d_grad_norm"], smooth_window), label="D Grad Norm")
    ax.plot(steps, smooth(df["g_grad_norm"], smooth_window), label="G Grad Norm")
    ax.set_title("Gradient Magnitudes"); ax.set_xlabel("Step"); ax.set_ylabel("Gradient norm")
    ax.grid(True, alpha=0.3); ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_per_edge_distribution_plot(real_batch: np.ndarray, fake_batch: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("EXP1 (MBD) Hybrid GAN: Per-edge distributions", fontsize=16, fontweight="bold")
    for i, ax in enumerate(axes.flat):
        ax.hist(real_batch[:, i], bins=30, alpha=0.55, density=True, label="real")
        ax.hist(fake_batch[:, i], bins=30, alpha=0.55, density=True, label="fake")
        ax.set_title(EDGE_NAMES[i]); ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_histogram_plot(values: np.ndarray, out_path: Path, title: str, xlabel: str) -> None:
    values = np.asarray(values).flatten()
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=20, density=False)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


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
    with torch.no_grad():
        while remaining > 0:
            bs = min(batch_size, remaining)
            z = sample_latent_torch(bs, latent_dim, latent_distribution, device)
            fake = gen(z).detach().cpu().numpy()
            if fake.ndim == 1:
                fake = fake.reshape(1, -1)
            outputs.append(fake)
            remaining -= bs
    gen.train()
    return np.concatenate(outputs, axis=0)


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device)

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"logs/exp1_mbd_qgen_cdisc_{timestamp}")
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] === EXPERIMENT 1: MBD RUN ===")
    print(f"[INFO] batch_size={args.batch_size}, steps={args.steps}")
    print(f"[INFO] g_lr={args.g_lr}, d_lr={args.d_lr}, layers={args.layers}, init_std={args.init_std}")
    print(f"[INFO] Label smoothing={args.label_smoothing} (Real-Target={1.0 - args.label_smoothing} for D AND G)")
    print(f"[INFO] NO Minibatch Discrimination, NO Post-Processing Adapter")

    real_data = load_real_data(args.valid_tuples)

    gen = QGenerator(n_layer=args.layers, init_std=args.init_std, seed=args.seed).to(device)
    disc = Discriminator(hidden_dim=args.hidden_dim).to(device)

    initial_weights = gen.weights.detach().cpu().numpy().copy()
    np.save(output_dir / "initial_qgen_weights.npy", initial_weights)
    make_histogram_plot(
        initial_weights,
        output_dir / "initial_qgen_weights_histogram.png",
        title="Initial QGEN Weights Distribution",
        xlabel="Initial weight value",
    )

    g_optimizer = optim.Adam(gen.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(disc.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    real_label_val = 1.0 - args.label_smoothing   # 0.9 default
    fake_label_val = args.label_smoothing          # 0.1 default

    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

    real_used_samples = []
    fake_used_samples = []
    latent_used_samples = []
    weight_snapshot = []
    last_metrics: Dict[str, float] = {}

    for step in range(1, args.steps + 1):
        # -------------------------
        # Train Discriminator
        # -------------------------
        disc.train()
        d_optimizer.zero_grad(set_to_none=True)

        real_batch = sample_real_batch(real_data, args.batch_size, rng, device)
        z_d = sample_latent_torch(args.batch_size, args.latent_dim, args.latent_distribution, device)
        fake_batch_d = gen(z_d)
        latent_used_samples.append(z_d.detach().cpu().numpy())
        real_used_samples.append(real_batch.detach().cpu().numpy())
        fake_used_samples.append(fake_batch_d.detach().cpu().numpy())

        real_logits = disc(real_batch)
        fake_logits_d = disc(fake_batch_d.detach())

        # Label-Smoothing fuer D
        real_labels = torch.full_like(real_logits, real_label_val)
        fake_labels = torch.full_like(fake_logits_d, fake_label_val)

        d_loss_real = criterion(real_logits, real_labels)
        d_loss_fake = criterion(fake_logits_d, fake_labels)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        d_loss.backward()
        d_grad_norm = grad_norm_torch(disc.parameters())
        d_optimizer.step()

        # -------------------------
        # Train Generator (1x pro D-Step)
        # -------------------------
        g_optimizer.zero_grad(set_to_none=True)

        z_g = sample_latent_torch(args.batch_size, args.latent_dim, args.latent_distribution, device)
        fake_batch_g = gen(z_g)
        latent_used_samples.append(z_g.detach().cpu().numpy())
        fake_logits_g = disc(fake_batch_g)

        # Generator-Target = real_label_val (0.9 mit Smoothing) — wie in Lauf 2
        gen_labels = torch.full_like(fake_logits_g, real_label_val)
        g_loss = criterion(fake_logits_g, gen_labels)

        g_loss.backward()
        g_grad_norm = grad_norm_torch(gen.parameters())
        g_optimizer.step()

        # Weight-Snapshot (nur EIN Block — Duplikat-Bug behoben)
        if step == 1 or step == args.steps or (
            args.checkpoint_every > 0 and step % args.checkpoint_every == 0
        ):
            weight_snapshot.append({
                "step": step,
                "weights": gen.weights.detach().cpu().numpy().copy(),
            })

        # -------------------------
        # Logging
        # -------------------------
        with torch.no_grad():
            real_score = torch.sigmoid(real_logits).mean().item()
            fake_score_d = torch.sigmoid(fake_logits_d).mean().item()
            fake_score_g = torch.sigmoid(fake_logits_g).mean().item()
            stats = compute_batch_stats(fake_batch_g)

        metrics = {
            "step": step,
            "d_loss": float(d_loss.item()),
            "g_loss": float(g_loss.item()),
            "real_score": float(real_score),
            "fake_score_d": float(fake_score_d),
            "fake_score_g": float(fake_score_g),
            "d_grad_norm": float(d_grad_norm),
            "g_grad_norm": float(g_grad_norm),
            **stats,
            "real_e_ab": float(real_batch[:, 0].mean().item()),
            "real_e_bc": float(real_batch[:, 1].mean().item()),
            "real_e_cd": float(real_batch[:, 2].mean().item()),
            "real_e_da": float(real_batch[:, 3].mean().item()),
            "real_e_ac": float(real_batch[:, 4].mean().item()),
            "real_e_bd": float(real_batch[:, 5].mean().item()),
            "fake_e_ab": float(fake_batch_d[:, 0].mean().item()),
            "fake_e_bc": float(fake_batch_d[:, 1].mean().item()),
            "fake_e_cd": float(fake_batch_d[:, 2].mean().item()),
            "fake_e_da": float(fake_batch_d[:, 3].mean().item()),
            "fake_e_ac": float(fake_batch_d[:, 4].mean().item()),
            "fake_e_bd": float(fake_batch_d[:, 5].mean().item()),
        }
        last_metrics = metrics

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow(metrics)

        if step == 1 or step % 100 == 0:
            print(
                f"[Step {step:5d}] "
                f"d_loss={metrics['d_loss']:.4f} | g_loss={metrics['g_loss']:.4f} | "
                f"real_score={metrics['real_score']:.3f} | "
                f"fake_d={metrics['fake_score_d']:.3f} | "
                f"fake_g={metrics['fake_score_g']:.3f} | "
                f"d_grad={metrics['d_grad_norm']:.3f} | "
                f"g_grad={metrics['g_grad_norm']:.3f} | "
                f"fake_mean={metrics['fake_mean']:.3f} | "
                f"fake_std={metrics['fake_std']:.3f}"
            )

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            torch.save(gen.state_dict(), output_dir / f"qgen_step_{step:06d}.pt")
            torch.save(disc.state_dict(), output_dir / f"cdisc_step_{step:06d}.pt")

    # -------------------------
    # Post-training analyses
    # -------------------------
    final_weights = gen.weights.detach().cpu().numpy().copy()
    np.save(output_dir / "qgen_weights_final.npy", final_weights)
    make_histogram_plot(
        final_weights,
        output_dir / "qgen_weights_final_distribution.png",
        title="Final QGenerator weight distribution",
        xlabel="Final weight value",
    )

    weight_change = final_weights - initial_weights
    np.save(output_dir / "qgen_weights_delta.npy", weight_change)
    make_histogram_plot(
        weight_change,
        output_dir / "qgen_weights_delta_distribution.png",
        title="QGenerator weight changes: final - initial",
        xlabel="Weight change",
    )

    latent_used = np.concatenate(latent_used_samples, axis=0)
    np.save(output_dir / "latent_noise_used_training.npy", latent_used)
    make_histogram_plot(
        latent_used,
        output_dir / "latent_noise_used_training_distribution.png",
        title=f"Latent noise used during training ({args.latent_distribution})",
        xlabel="Noise value",
    )

    for snapshot in weight_snapshot:
        snapshot_step = snapshot["step"]
        snapshot_weights = snapshot["weights"]
        np.save(
            output_dir / f"qgen_weights_step_{snapshot_step:06d}.npy",
            snapshot_weights,
        )
        make_histogram_plot(
            snapshot_weights,
            output_dir / f"qgen_weights_step_{snapshot_step:06d}_distribution.png",
            title=f"QGenerator weight distribution at step {snapshot_step}",
            xlabel="Weight value",
        )

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

    make_training_plot(csv_path, output_dir / "training_plot.png", smooth_window=args.smooth_window)
    make_per_edge_distribution_plot(real_final, fake_final, output_dir / "per_edge_distributions_final.png")
    make_per_edge_distribution_plot(real_used, fake_used, output_dir / "per_edge_distributions_training_samples.png")

    summary = Summary(
        steps=args.steps,
        batch_size=args.batch_size,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        latent_distribution=args.latent_distribution,
        n_layers=args.layers,
        init_std=args.init_std,
        label_smoothing=args.label_smoothing,
        final_d_loss=float(last_metrics["d_loss"]),
        final_g_loss=float(last_metrics["g_loss"]),
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

    print("\n=== EXP1 (MBD) FINAL SUMMARY ===")
    print(json.dumps(asdict(summary), indent=2))
    print(f"[INFO] Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
