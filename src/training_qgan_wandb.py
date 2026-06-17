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

# Optional W&B import — falls nicht installiert, laeuft das Skript normal weiter
# wandb-Import absichern.
# Problem: W&B legt im Arbeitsverzeichnis einen Ordner 'wandb/' fuer Run-Daten an.
# Wenn man dann 'import wandb' macht, kann Python diesen lokalen Ordner als
# Namespace-Package interpretieren statt das echte Pip-Paket. Das fuehrt zu
# AttributeError: module 'wandb' has no attribute 'init'.
# Loesung: Wir entfernen das aktuelle Arbeitsverzeichnis und alle Varianten davon
# voruebergehend von sys.path, importieren wandb und stellen den Pfad wieder her.
import os as _os
import sys as _sys

WANDB_AVAILABLE = False
wandb = None

def _import_wandb_robust():
    """Importiert wandb so, dass ein lokales wandb/-Verzeichnis ignoriert wird."""
    global wandb, WANDB_AVAILABLE

    # ---- Versuch 1: Pfad-basierte Filterung ----
    cwd = _os.path.abspath(_os.getcwd())
    blacklist = {"", ".", cwd, _os.getcwd()}
    saved_path = list(_sys.path)
    new_path = []
    for p in saved_path:
        if not p:
            continue
        try:
            abs_p = _os.path.abspath(p)
        except Exception:
            abs_p = p
        if abs_p in blacklist or p in blacklist:
            continue
        new_path.append(p)

    # Eventuell schon geladenes (falsches) wandb-Modul aus Cache loeschen
    for mod_name in list(_sys.modules.keys()):
        if mod_name == "wandb" or mod_name.startswith("wandb."):
            del _sys.modules[mod_name]

    _sys.path = new_path
    try:
        import wandb as _w
        if hasattr(_w, "init"):
            return _w
    except ImportError:
        pass
    finally:
        _sys.path = saved_path

    # ---- Versuch 2: Direkter Import aus site-packages via importlib ----
    # Cache nochmal leeren
    for mod_name in list(_sys.modules.keys()):
        if mod_name == "wandb" or mod_name.startswith("wandb."):
            del _sys.modules[mod_name]

    import importlib.util
    import site

    # Finde alle site-packages-Verzeichnisse
    candidates = []
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass
    # venv-spezifische Pfade abdecken
    for p in _sys.path:
        if "site-packages" in p:
            candidates.append(p)

    for sp_dir in candidates:
        wandb_init = _os.path.join(sp_dir, "wandb", "__init__.py")
        if _os.path.isfile(wandb_init):
            spec = importlib.util.spec_from_file_location("wandb", wandb_init)
            if spec and spec.loader:
                try:
                    mod = importlib.util.module_from_spec(spec)
                    _sys.modules["wandb"] = mod
                    spec.loader.exec_module(mod)
                    if hasattr(mod, "init"):
                        print(f"[INFO] wandb geladen via importlib aus: {sp_dir}")
                        return mod
                except Exception as e:
                    print(f"[WARN] importlib-Versuch fuer {sp_dir} fehlgeschlagen: {e}")
                    if "wandb" in _sys.modules:
                        del _sys.modules["wandb"]

    # Diagnose-Ausgabe, falls beide Versuche scheitern
    print(f"[DEBUG] sys.path: {_sys.path}")
    print(f"[DEBUG] cwd: {cwd}")
    print(f"[DEBUG] cwd hat wandb/-Ordner: {_os.path.isdir(_os.path.join(cwd, 'wandb'))}")
    return None


_w = _import_wandb_robust()
if _w is not None:
    wandb = _w
    WANDB_AVAILABLE = True
else:
    print("[WARN] wandb konnte nicht geladen werden — W&B-Logging deaktiviert.")
    print("[WARN] Falls wandb installiert ist: pruefe ob ein 'wandb/'-Ordner im cwd liegt,")
    print("[WARN] der das Package shadowed. Loesung: 'pip show wandb' und pruefen ob Location passt.")

try:
    from scipy.stats import wasserstein_distance
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    wasserstein_distance = None

try:
    from config import MAX_EDGE_LENGTH_KM
except Exception:
    MAX_EDGE_LENGTH_KM = 5000.0


# =============================================================================
# Minibatch Discrimination (gegen Mode Collapse)
# =============================================================================
class MinibatchDiscrimination(nn.Module):
    """
    Gibt dem Discriminator ein Feature pro Sample, das misst,
    wie ähnlich das Sample zu den anderen Samples im Batch ist.
    Bestraft Mode Collapse direkt im Gradienten.
    Funktioniert nur sinnvoll mit batch_size >= 8.
    """
    def __init__(self, in_features: int, out_features: int, kernel_dims: int):
        super().__init__()
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        M = torch.einsum("bi,ijk->bjk", x, self.T)                  # (batch, out, kdim)
        L1 = torch.abs(M.unsqueeze(0) - M.unsqueeze(1)).sum(dim=3)  # (batch, batch, out)
        c = torch.exp(-L1)
        mask = 1 - torch.eye(x.size(0), device=x.device).unsqueeze(2)
        o_b = (c * mask).mean(dim=1)                                # (batch, out)
        return torch.cat([x, o_b], dim=1)


# =============================================================================
# Classical Discriminator
# =============================================================================
class Discriminator(nn.Module):
    """Classical discriminator. Returns raw logits; use BCEWithLogitsLoss."""

    def __init__(self, hidden_dim: int = 64, use_mbd: bool = True, mbd_features: int = 8):
        super().__init__()
        self.use_mbd = use_mbd
        if use_mbd:
            self.mbd = MinibatchDiscrimination(
                in_features=6, out_features=mbd_features, kernel_dims=3
            )
            in_feats = 6 + mbd_features
        else:
            self.mbd = None
            in_feats = 6

        self.net = nn.Sequential(
            nn.Linear(in_feats, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_mbd:
            x = self.mbd(x)
        return self.net(x)


# =============================================================================
# Quantum Generator: jetzt mit echtem Batching und Pauli-Z-Output
# =============================================================================
class QGenerator(nn.Module):
    """
    6-Qubit Quantum-Generator mit echtem Batching.

    Input  forward(noise_batch): (batch, 6) oder (6,)
    Output:                      (batch, 6) oder (6,) in [0, 1]
    """

    def __init__(self, n_layer: int = 2, init_std: float = 0.3, seed: int = 0):
        super().__init__()
        self.n_qubits = 6
        self.n_layer = n_layer
        self.rng = np.random.default_rng(seed)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        init = torch.tensor(
            self.rng.normal(0.0, init_std, size=(n_layer, self.n_qubits, 3)),
            dtype=torch.float32,
        )
        self.weights = nn.Parameter(init)
        # Klassischer Post-Processing-Adapter
        #self.post = nn.Linear(self.n_qubits, self.n_qubits)
        #nn.init.eye_(self.post.weight)      # startet als Identität
        #nn.init.zeros_(self.post.bias)      # startet ohne Bias

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(noise_batch: torch.Tensor, weights: torch.Tensor):
            # noise_batch: (batch, n_qubits) — default.qubit broadcastet automatisch
            qml.AngleEmbedding(noise_batch, wires=range(self.n_qubits), rotation="Y")
            
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
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def embedding_circuit(noise_batch: torch.Tensor):
            qml.AngleEmbedding(noise_batch, wires=range(self.n_qubits), rotation="Y")
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.embedding_circuit = embedding_circuit
    def forward(self, noise_batch: torch.Tensor) -> torch.Tensor:
        noise_batch = noise_batch.to(dtype=torch.float32)

        squeeze_output = False
        if noise_batch.ndim == 1:
            noise_batch = noise_batch.unsqueeze(0)
            squeeze_output = True

        # Ein einziger QNode-Call mit dem ganzen Batch
        z_vals = self.circuit(noise_batch, self.weights)
        # z_vals: Liste von n_qubits Tensoren — bei batch=1 evtl. 0-dim, dann unsqueezen
        z = torch.stack(
            [v if v.ndim > 0 else v.unsqueeze(0) for v in z_vals],
            dim=-1,
        ).to(dtype=torch.float32)

        # [-1, 1] → [0, 1]
        out = (z + 1.0) / 2.0
        #out = torch.sigmoid(self.post(z))

        if squeeze_output:
            out = out.squeeze(0)
        return out


@dataclass
class Summary:
    steps: int
    batch_size: int
    seed: int
    g_lr: float
    d_lr: float
    latent_distribution: str
    n_layers: int
    init_std: float
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
        description="Hybrid GAN: QGEN + cDISC with classical-style training plot."
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=32)
    parser.add_argument("--g-lr", "--g_lr", type=float, default=1e-3)
    parser.add_argument("--d-lr", "--d_lr", type=float, default=1e-4)
    parser.add_argument("--latent-dim", "--latent_dim", type=int, default=6)
    parser.add_argument(
        "--latent-distribution", "--latent_distribution",
        type=str,
        default="angle",
        choices=["uniform", "normal", "angle", "uniform_pm1"],
    )
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--n-critic-g", "--n_critic_g", type=int, default=1,
                    help="Anzahl Generator-Updates pro Discriminator-Update.")
    parser.add_argument("--hidden-dim", "--hidden_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-std", "--init_std", type=float, default=0.1)
    parser.add_argument("--valid-tuples", "--valid_tuples", type=str, default="valid_tuples.csv")
    parser.add_argument("--eval-samples", "--eval_samples", type=int, default=1000)
    parser.add_argument("--checkpoint-every", "--checkpoint_every", type=int, default=500)
    parser.add_argument("--smooth-window", "--smooth_window", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use-mbd", "--use_mbd", action="store_true", default=True,
                        help="Aktiviert Minibatch Discrimination (default an).")
    parser.add_argument("--no-mbd", "--no_mbd", dest="use_mbd", action="store_false")
    parser.add_argument("--label-smoothing", "--label_smoothing", type=float, default=0.1,
                        help="Real-Label = 1 - smoothing, Fake-Label = smoothing.")
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=None)
    parser.add_argument(
        "--show-circuit", "--show_circuit",
        action="store_true",
        help="Print the PennyLane circuit at startup for debugging.",
    )

    # ---- W&B-Argumente ----
    parser.add_argument("--wandb-project", "--wandb_project", type=str, default="qugan-ba-hp-sweep",
                        help="W&B-Projektname (None oder leer = W&B deaktiviert).")
    parser.add_argument("--wandb-entity", "--wandb_entity", type=str, default=None,
                        help="W&B-Entity (Benutzer/Team). None = persoenlich.")
    parser.add_argument("--wandb-run-name", "--wandb_run_name", type=str, default=None,
                        help="Optionaler Run-Name. Bei None wird automatisch generiert.")
    parser.add_argument("--no-wandb", "--no_wandb", action="store_true",
                        help="Deaktiviert W&B-Logging.")
    parser.add_argument("--wasserstein-eval-every", "--wasserstein_eval_every", type=int, default=200,
                        help="Alle N Steps Wasserstein-Distanz berechnen "
                             "(als Ziel-Metrik fuer den Sweep).")
    parser.add_argument("--wasserstein-eval-samples", "--wasserstein_eval_samples", type=int, default=500,
                        help="Anzahl Samples fuer Wasserstein-Zwischenauswertung.")
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
        f"Hybrid GAN Training: QGEN + cDISC (Smoothed over {smooth_window} steps)",
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
    ax.plot(steps, smooth(df["real_score"], smooth_window), label="Real (should→1.0)")
    ax.plot(steps, smooth(df["fake_score_d"], smooth_window), label="Fake-Disc (should→0.0)")
    ax.axhline(0.5, linestyle="--", alpha=0.7, label="Chance level")
    ax.set_title("Discriminator Classification Scores"); ax.set_xlabel("Step")
    ax.set_ylabel("D output probability"); ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1, 1]
    ax.plot(steps, smooth(df["fake_score_g"], smooth_window), label="Fake-Gen (should→1.0)")
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
    fig.suptitle("Hybrid GAN: Per-edge distributions", fontsize=16, fontweight="bold")
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


def evaluate_wasserstein(
    gen,
    real_data: np.ndarray,
    n_samples: int,
    batch_size: int,
    latent_dim: int,
    latent_distribution: str,
    device: torch.device,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Schnelle Zwischenauswertung der Wasserstein-Distanz pro Kante.

    Verwendet einen festen Real-Subset (fuer Vergleichbarkeit zwischen Steps).
    Generiert frische Fakes mit dem aktuellen Generator-Zustand.

    Returns dict mit per-edge Wasserstein-Distanzen + Mittelwert.
    """
    if not SCIPY_AVAILABLE:
        return {}

    # Real-Stichprobe (deterministisch waehlbar wenn rng konsistent)
    idx = rng.choice(len(real_data), size=n_samples, replace=True)
    real_subset = real_data[idx]

    # Fake-Stichprobe
    outputs: List[np.ndarray] = []
    remaining = n_samples
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
    fakes = np.concatenate(outputs, axis=0)[:n_samples]

    per_edge = {}
    for i, name in enumerate(EDGE_NAMES):
        per_edge[f"w_{name}"] = float(wasserstein_distance(real_subset[:, i], fakes[:, i]))
    per_edge["w_mean"] = float(np.mean(list(per_edge.values())))
    return per_edge


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

    # Sanity-Check: MBD braucht ausreichend grossen Batch
    if args.use_mbd and args.batch_size < 8:
        print(f"[WARN] Minibatch Discrimination ist aktiv, aber batch_size={args.batch_size} ist zu klein.")
        print("[WARN] Empfohlen: batch_size >= 16. MBD wird automatisch deaktiviert.")
        args.use_mbd = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device)

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"logs/hybrid_qgen_cdisc_{timestamp}")
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] batch_size={args.batch_size}, steps={args.steps}, MBD={args.use_mbd}")
    print(f"[INFO] g_lr={args.g_lr}, d_lr={args.d_lr}, layers={args.layers}, init_std={args.init_std}")
    print("[INFO] Training hybrid GAN: QGenerator + classical Discriminator")

    # ---- W&B-Initialisierung ----
    use_wandb = WANDB_AVAILABLE and (not args.no_wandb) and bool(args.wandb_project)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "steps": args.steps,
                "batch_size": args.batch_size,
                "g_lr": args.g_lr,
                "d_lr": args.d_lr,
                "layers": args.layers,
                "init_std": args.init_std,
                "n_critic_g": args.n_critic_g,
                "use_mbd": args.use_mbd,
                "hidden_dim": args.hidden_dim,
                "label_smoothing": args.label_smoothing,
                "latent_dim": args.latent_dim,
                "latent_distribution": args.latent_distribution,
                "seed": args.seed,
                "model": "QGEN_PauliZ_cDISC_MBD",
                "diff_method": "backprop",
                "shots": "None",
            },
        )
        print(f"[INFO] W&B aktiviert: project={args.wandb_project}, run={wandb.run.name}")
    elif not WANDB_AVAILABLE and not args.no_wandb:
        print("[WARN] wandb nicht installiert. Mit 'pip install wandb' installieren.")
        print("[WARN] Training laeuft ohne W&B-Logging weiter.")
    else:
        print("[INFO] W&B-Logging deaktiviert.")

    real_data = load_real_data(args.valid_tuples)

    gen = QGenerator(n_layer=args.layers, init_std=args.init_std, seed=args.seed).to(device)
    z = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32)

    if args.show_circuit:
        print(qml.draw(gen.circuit)(z, gen.weights).encode("ascii", "replace").decode("ascii"))
    disc = Discriminator(hidden_dim=args.hidden_dim, use_mbd=args.use_mbd).to(device)

    initial_weights = gen.weights.detach().cpu().numpy().copy()
    np.save(output_dir / "initial_qgen_weights.npy", initial_weights)
    make_histogram_plot(
        initial_weights,
        output_dir / "initial_qgen_weights_histogram.png",
        title="Initial QGEN Weights Distribution",
        xlabel="Initial weight value",
    )

    g_optimizer = optim.Adam(gen.parameters(), lr=args.g_lr, betas=(0.5, 0.999), weight_decay=0.005)
    d_optimizer = optim.Adam(disc.parameters(), lr=args.d_lr, betas=(0.5, 0.999), weight_decay=0.005)
    criterion = nn.BCEWithLogitsLoss()

    real_label_val = 1.0 - args.label_smoothing
    fake_label_val = args.label_smoothing

    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

    real_used_samples = []
    fake_used_samples = []
    latent_used_samples = []
    weight_snapshot = []
    embedding_used_samples = []
    last_metrics: Dict[str, float] = {}

    # ---- Best-Wasserstein-Tracking ueber das gesamte Training ----
    # Wir behalten den niedrigsten beobachteten Wasserstein-Wert und den Step,
    # an dem er auftrat. Das ist die Ziel-Metrik fuer den Sweep.
    best_wasserstein = float("inf")
    best_wasserstein_step = 0
    # Fester Eval-RNG, damit Wasserstein-Messungen zwischen Steps vergleichbar sind
    eval_rng = np.random.default_rng(args.seed + 9999)

    for step in range(1, args.steps + 1):
        # -------------------------
        # Train Discriminator
        # -------------------------
        disc.train()
        d_optimizer.zero_grad(set_to_none=True)

        real_batch = sample_real_batch(real_data, args.batch_size, rng, device)
        z_d = sample_latent_torch(args.batch_size, args.latent_dim, args.latent_distribution, device)
        with torch.no_grad():
            emb_d_vals = gen.embedding_circuit(z_d)
            emb_d = torch.stack(
                [v if v.ndim > 0 else v.unsqueeze(0) for v in emb_d_vals],
                dim=-1,
            )
        embedding_used_samples.append(emb_d.detach().cpu().numpy())
        fake_batch_d = gen(z_d)
        latent_used_samples.append(z_d.detach().cpu().numpy())
        real_used_samples.append(real_batch.detach().cpu().numpy())
        fake_used_samples.append(fake_batch_d.detach().cpu().numpy())

        real_logits = disc(real_batch)
        fake_logits_d = disc(fake_batch_d.detach())

        # Label-Smoothing
        real_labels = torch.full_like(real_logits, real_label_val)
        fake_labels = torch.full_like(fake_logits_d, fake_label_val)

        d_loss_real = criterion(real_logits, real_labels)
        d_loss_fake = criterion(fake_logits_d, fake_labels)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        d_loss.backward()
        d_grad_norm = grad_norm_torch(disc.parameters())
        d_optimizer.step()

        # -------------------------
        # Train Generator
        # -------------------------
        for _ in range(args.n_critic_g):
            g_optimizer.zero_grad(set_to_none=True)

            z_g = sample_latent_torch(args.batch_size, args.latent_dim, args.latent_distribution, device)
            with torch.no_grad():
                emb_g_vals = gen.embedding_circuit(z_g)
                emb_g = torch.stack(
                    [v if v.ndim > 0 else v.unsqueeze(0) for v in emb_g_vals],
                    dim=-1,
                )
            embedding_used_samples.append(emb_g.detach().cpu().numpy())
            fake_batch_g = gen(z_g)
            latent_used_samples.append(z_g.detach().cpu().numpy())
            fake_logits_g = disc(fake_batch_g)

            # Generator will, dass D denkt "real" → Label 1 (mit smoothing 0.9)
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

        # ---- W&B-Logging fuer diesen Step ----
        wandb_log_dict = {
            "train/d_loss": metrics["d_loss"],
            "train/g_loss": metrics["g_loss"],
            "scores/real": metrics["real_score"],
            "scores/fake_disc": metrics["fake_score_d"],
            "scores/fake_gen": metrics["fake_score_g"],
            "grad/d_norm": metrics["d_grad_norm"],
            "grad/g_norm": metrics["g_grad_norm"],
            "fake/mean": metrics["fake_mean"],
            "fake/std": metrics["fake_std"],
        }

        # ---- Wasserstein-Zwischenauswertung alle N Steps ----
        if (
            step % args.wasserstein_eval_every == 0
            or step == 1
            or step == args.steps
        ):
            w_metrics = evaluate_wasserstein(
                gen=gen,
                real_data=real_data,
                n_samples=args.wasserstein_eval_samples,
                batch_size=args.batch_size,
                latent_dim=args.latent_dim,
                latent_distribution=args.latent_distribution,
                device=device,
                rng=eval_rng,
            )
            if w_metrics:
                for k, v in w_metrics.items():
                    wandb_log_dict[f"wasserstein/{k}"] = v

                # Best-Tracking
                current_w = w_metrics["w_mean"]
                if current_w < best_wasserstein:
                    best_wasserstein = current_w
                    best_wasserstein_step = step
                wandb_log_dict["wasserstein/best_so_far"] = best_wasserstein
                wandb_log_dict["wasserstein/best_step"] = best_wasserstein_step

        if use_wandb:
            wandb.log(wandb_log_dict, step=step)

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
        seed=args.seed,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        latent_distribution=args.latent_distribution,
        n_layers=args.layers,
        init_std=args.init_std,
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

    # ---- Finale Wasserstein-Auswertung auf den Final-Samples ----
    # Diese Zahl ist die offizielle Bewertung des finalen Modells.
    final_w_per_edge = {}
    if SCIPY_AVAILABLE:
        for i, name in enumerate(EDGE_NAMES):
            final_w_per_edge[name] = float(
                wasserstein_distance(real_final[:, i], fake_final[:, i])
            )
        final_w_mean = float(np.mean(list(final_w_per_edge.values())))
    else:
        final_w_mean = None

    # ---- Finales W&B-Logging ----
    if use_wandb:
        wandb_summary = {
            "final/d_loss": float(last_metrics["d_loss"]),
            "final/g_loss": float(last_metrics["g_loss"]),
            "final/real_score": float(last_metrics["real_score"]),
            "final/fake_score_d": float(last_metrics["fake_score_d"]),
            "final/fake_score_g": float(last_metrics["fake_score_g"]),
            "final/fake_mean": float(last_metrics["fake_mean"]),
            "final/fake_std": float(last_metrics["fake_std"]),
            # Wichtigste Metriken fuer den Sweep:
            "best_wasserstein": float(best_wasserstein),
            "best_wasserstein_step": int(best_wasserstein_step),
        }
        if final_w_mean is not None:
            wandb_summary["final/wasserstein_mean"] = final_w_mean
            for name, val in final_w_per_edge.items():
                wandb_summary[f"final/wasserstein_{name}"] = val

        # In W&B 'summary' fest verankern, damit der Sweep-Optimizer
        # darauf zugreifen kann
        for k, v in wandb_summary.items():
            wandb.summary[k] = v

        # Hauptplots als W&B-Artefakte hochladen (optional, aber huebsch)
        try:
            wandb.log({
                "plots/training": wandb.Image(str(output_dir / "training_plot.png")),
                "plots/per_edge_final": wandb.Image(
                    str(output_dir / "per_edge_distributions_final.png")
                ),
            })
        except Exception as e:
            print(f"[WARN] Konnte Plots nicht zu W&B hochladen: {e}")

        wandb.finish()

    print("\n=== FINAL SUMMARY ===")
    print(json.dumps(asdict(summary), indent=2))
    print(f"\n[BEST] Wasserstein (mean ueber Training): {best_wasserstein:.4f} @ step {best_wasserstein_step}")
    if final_w_mean is not None:
        print(f"[FINAL] Wasserstein (final samples): {final_w_mean:.4f}")
    print(f"[INFO] Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
