"""
Temporal Evaluation: Wie hat sich der Generator UEBER DIE ZEIT entwickelt?

Laedt die gespeicherten qgen_step_*.pt Checkpoints aus einem Trainings-Run und
generiert fuer jeden Checkpoint eine frische Stichprobe Fake-Samples.
Ziel: rausfinden, ob der finale Stand der beste war oder ob das Modell
zwischenzeitlich besser war (-> Argument fuer Early Stopping / Best Checkpoint).

Outputs:
  - per_edge_distributions_over_time.png  (Subplot pro Kante, Linien pro Checkpoint)
  - wasserstein_over_time.png             (Wasserstein-Distanz Real vs Fake pro Step)
  - summary_temporal.json                 (numerische Zusammenfassung)
  - best_checkpoint.txt                   (welcher Step war am besten)

Aufruf:
  python eval_temporal.py --run-dir logs/baseline_qgen_cdisc_20260524_171345 \
                          --valid-tuples valid_tuples.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance


try:
    from config import MAX_EDGE_LENGTH_KM
except Exception:
    MAX_EDGE_LENGTH_KM = 5000.0


EDGE_COLS = ["e_ab", "e_bc", "e_cd", "e_da", "e_ac", "e_bd"]
EDGE_NAMES = ["ab", "bc", "cd", "da", "ac", "bd"]


# =============================================================================
# QGenerator - muss IDENTISCH zur Trainings-Datei sein, damit state_dict passt
# =============================================================================
class QGenerator(nn.Module):
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
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def forward(self, noise_batch: torch.Tensor) -> torch.Tensor:
        noise_batch = noise_batch.to(dtype=torch.float32)
        squeeze_output = False
        if noise_batch.ndim == 1:
            noise_batch = noise_batch.unsqueeze(0)
            squeeze_output = True
        z_vals = self.circuit(noise_batch, self.weights)
        z = torch.stack(
            [v if v.ndim > 0 else v.unsqueeze(0) for v in z_vals],
            dim=-1,
        ).to(dtype=torch.float32)
        out = (z + 1.0) / 2.0
        if squeeze_output:
            out = out.squeeze(0)
        return out


# =============================================================================
# Helpers
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal evaluation of QGAN checkpoints."
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Trainings-Output-Verzeichnis mit qgen_step_*.pt")
    parser.add_argument("--valid-tuples", type=str, default="valid_tuples.csv")
    parser.add_argument("--samples-per-checkpoint", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--latent-distribution", type=str, default="angle",
                        choices=["uniform", "normal", "angle"])
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--init-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_real_data(valid_tuples_path: str) -> np.ndarray:
    df = pd.read_csv(valid_tuples_path)
    real = df[EDGE_COLS].to_numpy(dtype=np.float32)
    real = np.clip(real / MAX_EDGE_LENGTH_KM, 0.0, 1.0)
    return real


def sample_latent(batch_size: int, latent_dim: int, distribution: str,
                  device: torch.device, rng: np.random.Generator) -> torch.Tensor:
    if distribution == "uniform":
        return torch.tensor(rng.random((batch_size, latent_dim)),
                            dtype=torch.float32, device=device)
    if distribution == "normal":
        return torch.tensor(rng.normal(size=(batch_size, latent_dim)),
                            dtype=torch.float32, device=device)
    # angle
    return torch.tensor(np.pi * rng.random((batch_size, latent_dim)),
                        dtype=torch.float32, device=device)


def generate_fake_samples(gen: QGenerator, num_samples: int, batch_size: int,
                          latent_dim: int, latent_distribution: str,
                          device: torch.device,
                          rng: np.random.Generator) -> np.ndarray:
    outputs: List[np.ndarray] = []
    remaining = num_samples
    gen.eval()
    with torch.no_grad():
        while remaining > 0:
            bs = min(batch_size, remaining)
            z = sample_latent(bs, latent_dim, latent_distribution, device, rng)
            fake = gen(z).detach().cpu().numpy()
            if fake.ndim == 1:
                fake = fake.reshape(1, -1)
            outputs.append(fake)
            remaining -= bs
    return np.concatenate(outputs, axis=0)


def find_checkpoints(run_dir: Path) -> List[Tuple[int, Path]]:
    """Returns sorted list of (step, path) tuples for all qgen_step_*.pt files."""
    pattern = str(run_dir / "qgen_step_*.pt")
    files = glob.glob(pattern)
    result = []
    for f in files:
        m = re.search(r"qgen_step_(\d+)\.pt", f)
        if m:
            result.append((int(m.group(1)), Path(f)))
    # Add final if present
    final_path = run_dir / "qgen_final.pt"
    if final_path.exists():
        # Try to get step from summary
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as fp:
                final_step = json.load(fp).get("steps", -1)
            if final_step > 0 and (final_step, final_path) not in result:
                result.append((final_step, final_path))
    result.sort(key=lambda x: x[0])
    return result


def compute_wasserstein_per_edge(real: np.ndarray, fake: np.ndarray) -> Dict[str, float]:
    """Returns dict {edge_name: wasserstein_distance}."""
    return {
        EDGE_NAMES[i]: float(wasserstein_distance(real[:, i], fake[:, i]))
        for i in range(6)
    }


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    print(f"[INFO] Run dir: {run_dir}")
    device = torch.device("cpu")

    # 1. Real-Daten laden (einmal, deterministisch)
    real_data = load_real_data(args.valid_tuples)
    rng_real = np.random.default_rng(args.seed)
    real_idx = rng_real.choice(len(real_data), size=args.samples_per_checkpoint,
                               replace=True)
    real_samples = real_data[real_idx]
    print(f"[INFO] Loaded {len(real_data)} real samples, using {len(real_samples)} for evaluation")

    # 2. Checkpoints finden
    checkpoints = find_checkpoints(run_dir)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {run_dir}")
    print(f"[INFO] Found {len(checkpoints)} checkpoints: steps {[s for s, _ in checkpoints]}")

    # 3. Pro Checkpoint Fakes generieren und Wasserstein berechnen
    rng_gen = np.random.default_rng(args.seed)  # gleicher Seed für alle Checkpoints

    fake_samples_per_step: Dict[int, np.ndarray] = {}
    wasserstein_per_step: Dict[int, Dict[str, float]] = {}

    for step, ckpt_path in checkpoints:
        # Reset RNG so dass alle Checkpoints die SELBEN Latent-Vectors sehen
        # (fairer Vergleich: nur Generator-Gewichte unterscheiden sich)
        rng_gen_local = np.random.default_rng(args.seed)

        gen = QGenerator(n_layer=args.layers, init_std=args.init_std,
                         seed=args.seed).to(device)
        try:
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(ckpt_path, map_location=device)
        gen.load_state_dict(state_dict)

        fakes = generate_fake_samples(
            gen=gen,
            num_samples=args.samples_per_checkpoint,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            latent_distribution=args.latent_distribution,
            device=device,
            rng=rng_gen_local,
        )
        fake_samples_per_step[step] = fakes
        w = compute_wasserstein_per_edge(real_samples, fakes)
        wasserstein_per_step[step] = w
        w_mean = float(np.mean(list(w.values())))
        print(f"[Step {step:5d}] mean W-distance = {w_mean:.4f} | "
              f"per edge: {', '.join(f'{k}={v:.3f}' for k, v in w.items())}")

    # 4. Plot 1: Wasserstein über Zeit
    steps_sorted = sorted(wasserstein_per_step.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    for edge in EDGE_NAMES:
        vals = [wasserstein_per_step[s][edge] for s in steps_sorted]
        ax.plot(steps_sorted, vals, marker="o", label=edge)
    mean_vals = [np.mean(list(wasserstein_per_step[s].values())) for s in steps_sorted]
    ax.plot(steps_sorted, mean_vals, marker="s", linewidth=3, color="black",
            label="mean", linestyle="--")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Wasserstein distance (Real vs Fake)")
    ax.set_title("Wasserstein distance per edge over training")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "wasserstein_over_time.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved wasserstein_over_time.png")

    # 5. Plot 2: Per-edge Histogramme nebeneinander - frueher / mid / final
    # Pick first, middle, and last checkpoints for clarity
    n_ckpt = len(steps_sorted)
    show_steps = [steps_sorted[0],
                  steps_sorted[n_ckpt // 4],
                  steps_sorted[n_ckpt // 2],
                  steps_sorted[3 * n_ckpt // 4],
                  steps_sorted[-1]]
    show_steps = sorted(set(show_steps))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Per-edge distributions: Real vs Fake at different training steps",
                 fontsize=14, fontweight="bold")
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(show_steps)))

    for i, ax in enumerate(axes.flat):
        ax.hist(real_samples[:, i], bins=30, alpha=0.4, density=True,
                label="real", color="steelblue")
        for ci, step in enumerate(show_steps):
            ax.hist(fake_samples_per_step[step][:, i], bins=30, alpha=0.35,
                    density=True, label=f"fake @ step {step}",
                    color=colors[ci], histtype="step", linewidth=2)
        ax.set_title(EDGE_NAMES[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(run_dir / "per_edge_distributions_over_time.png",
                dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved per_edge_distributions_over_time.png")

    # 6. Best Checkpoint identifizieren
    best_step = min(steps_sorted,
                    key=lambda s: np.mean(list(wasserstein_per_step[s].values())))
    final_step = steps_sorted[-1]
    best_w = np.mean(list(wasserstein_per_step[best_step].values()))
    final_w = np.mean(list(wasserstein_per_step[final_step].values()))

    print()
    print("=" * 60)
    print(f"BEST CHECKPOINT:  step {best_step}, mean W-distance = {best_w:.4f}")
    print(f"FINAL CHECKPOINT: step {final_step}, mean W-distance = {final_w:.4f}")
    if best_step != final_step:
        improvement = (final_w - best_w) / final_w * 100
        print(f"==> Best was {improvement:.1f}% better than final")
        print(f"==> CONSIDER EARLY STOPPING at step {best_step}")
    else:
        print(f"==> Final IS the best checkpoint")
    print("=" * 60)

    # 7. Speichern: zusammenfassende JSON + bestes Checkpoint
    summary = {
        "checkpoints_evaluated": steps_sorted,
        "wasserstein_per_step": {
            str(s): wasserstein_per_step[s] for s in steps_sorted
        },
        "mean_wasserstein_per_step": {
            str(s): float(np.mean(list(wasserstein_per_step[s].values())))
            for s in steps_sorted
        },
        "best_checkpoint_step": int(best_step),
        "best_checkpoint_mean_wasserstein": float(best_w),
        "final_checkpoint_step": int(final_step),
        "final_checkpoint_mean_wasserstein": float(final_w),
    }
    with open(run_dir / "summary_temporal.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Saved summary_temporal.json")

    with open(run_dir / "best_checkpoint.txt", "w", encoding="utf-8") as f:
        f.write(f"Best checkpoint: step {best_step}\n")
        f.write(f"Mean Wasserstein distance: {best_w:.6f}\n")
        f.write(f"Path: qgen_step_{best_step:06d}.pt\n")
        f.write(f"\nFinal checkpoint: step {final_step}\n")
        f.write(f"Final mean Wasserstein: {final_w:.6f}\n")

    # 8. Bestes Per-Edge Histogramm zum Vergleich
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"BEST checkpoint (step {best_step}, W={best_w:.4f}) vs FINAL (step {final_step}, W={final_w:.4f})",
        fontsize=14, fontweight="bold"
    )
    for i, ax in enumerate(axes.flat):
        ax.hist(real_samples[:, i], bins=30, alpha=0.55, density=True,
                label="real", color="steelblue")
        ax.hist(fake_samples_per_step[best_step][:, i], bins=30, alpha=0.55,
                density=True, label=f"fake @ best (step {best_step})", color="green")
        ax.hist(fake_samples_per_step[final_step][:, i], bins=30, alpha=0.35,
                density=True, label=f"fake @ final (step {final_step})", color="red",
                histtype="step", linewidth=2)
        ax.set_title(EDGE_NAMES[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(run_dir / "per_edge_best_vs_final.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved per_edge_best_vs_final.png")

    print(f"\n[INFO] All outputs written to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
