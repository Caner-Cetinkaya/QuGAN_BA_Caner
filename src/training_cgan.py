"""
Classical GAN Training for TSP Edge Distribution Learning
QuGAN-comparable variant:
- same real-data logic (4 cities -> 6 normalized edges)
- same latent sampling support (uniform [0,1] by default)
- same loss semantics as training_qgan.py for mse/log/pce on probabilities
- same adversarial training structure (D: real->LABEL_REAL, fake->LABEL_FAKE; G: fake->LABEL_REAL)
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from training_qgan import (
    load_cities,
    sample_edges_from_cache,
)
from config import (
    BATCH_SIZE,
    TRAINING_STEPS,
    DISC_STEPS_PER_GEN,
    DISC_LEARNING_RATE,
    GEN_LEARNING_RATE,
    LEARNING_RATE,
    SEED,
    LOSS_TYPE,
    DISC_WARMUP_STEPS,
    MAX_EDGE_LENGTH_KM,
    LABEL_REAL,
    LABEL_FAKE,
)

# Reproducibility
np.random.seed(SEED)
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

LATENT_DIM = 6
GRAD_CLIP_NORM = 10.0
# LATENT_DISTRIBUTION = "uniform"  # set to "normal" if you explicitly want torch.randn
LATENT_DISTRIBUTION = os.getenv("CGAN_LATENT_DISTRIBUTION", "uniform").strip().lower()

class Generator(nn.Module):
    """Maps 6-D latent noise to 6 normalized edge values in [0,1]."""

    def __init__(self, latent_dim: int = LATENT_DIM, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, 6),
            nn.Sigmoid(),  # outputs probabilities-like normalized edges in [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """Outputs probability in [0,1], matching qGAN loss semantics."""

    def __init__(self, hidden_dim: int = 128):
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


class GANLoss:
    """Probability-based loss semantics aligned with training_qgan._loss_fn."""

"""     def __init__(self, loss_type: str):
        loss_type = loss_type.lower().strip()
        if loss_type == "bce":
            # Map classical "bce" config to qGAN-style "log" semantics.
            loss_type = "log"
        if loss_type not in {"mse", "log", "pce"}:
            raise ValueError(
                f"Unsupported LOSS_TYPE='{loss_type}'. Use 'mse', 'log', 'pce' or 'bce'."
            )
        self.loss_type = loss_type

    def __call__(self, probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = torch.clamp(probs, 1e-12, 1.0 - 1e-12)
        labels = labels.to(dtype=probs.dtype)

        if self.loss_type == "mse":
            return torch.mean((probs - labels) ** 2)

        if self.loss_type == "pce":
            eps = 1e-9
            denom = torch.where(labels > eps, labels + eps, torch.ones_like(labels))
            return torch.mean(((probs - labels) ** 2) / denom)

        # qGAN "log" loss
        return -torch.mean(labels * torch.log(probs) + (1.0 - labels) * torch.log(1.0 - probs))
"""

# LOSS_FN = GANLoss(LOSS_TYPE)
LOSS_FN = nn.BCEWithLogitsLoss()


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def load_distance_cache_dict(cache_path: str = "distance_cache.csv") -> Dict[Tuple[str, str], float]:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache nicht gefunden: {cache_path}")

    cache = {}
    with open(cache_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cache[(row["k1"], row["k2"])] = float(row["distance_km"])
    return cache

def load_valid_real_tuples(path: str = "valid_tuples.csv") -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Valid tuples file not found: {path}")

    df = pd.read_csv(path)

    needed = ["e_ab", "e_bc", "e_cd", "e_da", "e_ac", "e_bd"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r} in {path}")

    arr_km = df[needed].to_numpy(dtype=np.float32)
    arr_norm = np.clip(arr_km / MAX_EDGE_LENGTH_KM, 0.0, 1.0)

    if len(arr_norm) == 0:
        raise RuntimeError(f"No valid tuples found in {path}")

    return arr_norm

def compute_tensor_stats(x: torch.Tensor, prefix: str) -> Dict[str, float]:
    x_detached = x.detach()
    stats = {
        f"{prefix}_mean": float(x_detached.mean().item()),
        f"{prefix}_std": float(x_detached.std().item()),
        f"{prefix}_min": float(x_detached.min().item()),
        f"{prefix}_max": float(x_detached.max().item()),
        f"{prefix}_near_zero_frac": float((x_detached <= 0.05).float().mean().item()),
        f"{prefix}_near_one_frac": float((x_detached >= 0.95).float().mean().item()),
    }
    return stats

def create_batch_real(valid_real_tuples: np.ndarray, batch_size: int, rng_obj):
    idx = rng_obj.choice(len(valid_real_tuples), size=batch_size, replace=True)
    batch_np = np.asarray(valid_real_tuples[idx], dtype=np.float32)
    clipped_fraction = float(np.mean(batch_np >= 1.0))
    return batch_np, clipped_fraction
"""
def create_batch_real(cities, batch_size, cache, rng_obj):
    batch = []
    clipped_value_count = 0
    total_value_count = 0
    max_retries = batch_size * 10
    attempt = 0

    while len(batch) < batch_size and attempt < max_retries:
        success, edges_km = sample_edges_from_cache(cities, cache, rng_obj)
        attempt += 1
        if not success:
            continue
        normalized_unclipped = edges_km / MAX_EDGE_LENGTH_KM
        clipped_value_count += int(np.sum(normalized_unclipped > 1.0))
        total_value_count += int(normalized_unclipped.size)

        edges_normalized = np.clip(normalized_unclipped, 0.0, 1.0)
        batch.append(edges_normalized.astype(np.float32))

    if len(batch) < batch_size:
        raise RuntimeError(
            f"Could not create full batch: got {len(batch)}/{batch_size} samples after {max_retries} attempts"
        )
    batch_np = np.asarray(batch, dtype=np.float32)
    clipped_fraction = float(clipped_value_count / max(total_value_count,1))
    return batch_np, clipped_fraction
"""

def sample_latent(batch_size: int, latent_dim: int = LATENT_DIM) -> torch.Tensor:
    if LATENT_DISTRIBUTION == "uniform":
        z_np = rng.uniform(0.0, 1.0, size=(batch_size, latent_dim)).astype(np.float32)
        return torch.from_numpy(z_np).to(DEVICE)
    if LATENT_DISTRIBUTION == "normal":
        return torch.randn(batch_size, latent_dim, device=DEVICE)
    raise ValueError(f"Unknown LATENT_DISTRIBUTION={LATENT_DISTRIBUTION!r}")


@torch.no_grad()
def create_batch_fake(generator: nn.Module, batch_size: int) -> torch.Tensor:
    return generator(sample_latent(batch_size))


def create_batch_fake_with_grad(generator: nn.Module, batch_size: int, z: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if z is None:
        z = sample_latent(batch_size)
    return generator(z), z


def train_discriminator_step(generator, discriminator, optimizer_d, valid_real_tuples, rng_obj, batch_size):
    discriminator.train()
    generator.train()
    set_requires_grad(discriminator, True)

    real_batch, clip_fraction_real = create_batch_real(valid_real_tuples, batch_size, rng_obj)
    real_tensor = torch.as_tensor(real_batch, dtype=torch.float32, device=DEVICE)
    fake_tensor = create_batch_fake(generator, batch_size).detach()

    target_real = torch.full((batch_size, 1), float(LABEL_REAL), device=DEVICE)
    target_fake = torch.full((batch_size, 1), float(LABEL_FAKE), device=DEVICE)

    optimizer_d.zero_grad(set_to_none=True)

    real_logits = discriminator(real_tensor)
    fake_logits = discriminator(fake_tensor)

    disc_loss_real = LOSS_FN(real_logits, target_real)
    disc_loss_fake = LOSS_FN(fake_logits, target_fake)
    loss_d = 0.5 * (disc_loss_real + disc_loss_fake)

    loss_d.backward()
    grad_norm_d = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=GRAD_CLIP_NORM)
    optimizer_d.step()


    #with torch.no_grad():
       # real_probs = discriminator(real_tensor)
       # fake_probs = discriminator(fake_tensor)

   # with torch.no_grad():
    #    real_probs = torch.sigmoid(discriminator(real_tensor))
     #   fake_probs = torch.sigmoid(discriminator(fake_tensor))

    with torch.no_grad():
        real_logits_eval = discriminator(real_tensor)
        fake_logits_eval = discriminator(fake_tensor)
        real_probs = torch.sigmoid(real_logits_eval)
        fake_probs = torch.sigmoid(fake_logits_eval)

    metrics = {
    "loss_d": float(loss_d.detach().item()),
    "loss_d_real": float(disc_loss_real.detach().item()),
    "loss_d_fake": float(disc_loss_fake.detach().item()),
    "score_real": float(real_probs.mean().item()),
    "score_fake_d": float(fake_probs.mean().item()),
    "real_logit_mean": float(real_logits_eval.mean().item()),
    "fake_logit_mean": float(fake_logits_eval.mean().item()),
    "grad_norm_d": float(grad_norm_d.item()),
    "real_clip_frac": float(clip_fraction_real),
}

    metrics.update(compute_tensor_stats(real_tensor, "real"))
    metrics.update(compute_tensor_stats(fake_tensor, "fake_d"))
    return metrics


def train_generator_step(generator, discriminator, optimizer_g, batch_size):
    generator.train()
    discriminator.train()
    set_requires_grad(discriminator, False)

    target_real = torch.full((batch_size, 1), float(LABEL_REAL), device=DEVICE)
    optimizer_g.zero_grad(set_to_none=True)

    z = sample_latent(batch_size)
    fake_tensor_before, _ = create_batch_fake_with_grad(generator, batch_size, z=z)
    fake_probs_before = discriminator(fake_tensor_before)
    loss_g = LOSS_FN(fake_probs_before, target_real)
    loss_g.backward()
    grad_norm_g = torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=GRAD_CLIP_NORM)
    optimizer_g.step()

   # with torch.no_grad():
    #    fake_tensor_after = generator(z)
     #   fake_probs_after = discriminator(fake_tensor_after)

    with torch.no_grad():
        fake_tensor_after = generator(z)
        fake_probs_after = torch.sigmoid(discriminator(fake_tensor_after))

    set_requires_grad(discriminator, True)
    metrics = {
        "loss_g": float(loss_g.detach().item()),
        #"score_fake_g": float(fake_probs_after.mean().item()),
        "score_fake_g": float(fake_probs_after.mean().item()),
        "grad_norm_g": float(grad_norm_g.item()),
    }
    metrics.update(compute_tensor_stats(fake_tensor_after, "fake_g"))
    return metrics


def main():
    valid_real_tuples = load_valid_real_tuples("valid_tuples.csv")
    print(f"[INFO] Loaded {len(valid_real_tuples)} valid real tuples")
    #cities = load_cities("cities.csv")
    #cache = load_distance_cache_dict("distance_cache.csv")

    #print(f"[INFO] Loaded {len(cities)} cities")
    #print(f"[INFO] Distance cache entries: {len(cache)}")

    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=GEN_LEARNING_RATE if GEN_LEARNING_RATE is not None else LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=DISC_LEARNING_RATE if DISC_LEARNING_RATE is not None else LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"logs/cgan_comparable_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "metrics.csv"
    config_path = log_dir / "config.json"

    config_dict = {
        "timestamp": timestamp,
        "batch_size": BATCH_SIZE,
        "training_steps": TRAINING_STEPS,
        "disc_steps_per_gen": DISC_STEPS_PER_GEN,
        "disc_learning_rate": DISC_LEARNING_RATE,
        "gen_learning_rate": GEN_LEARNING_RATE,
        "seed": SEED,
        "loss_type": LOSS_TYPE,
        "disc_warmup_steps": DISC_WARMUP_STEPS,
        "device": str(DEVICE),
        "model_type": "classical_gan_comparable_to_qgan",
        "discriminator_outputs": "logits",
        "gradient_clip_norm": GRAD_CLIP_NORM,
        "latent_distribution": LATENT_DISTRIBUTION,
        "label_real": LABEL_REAL,
        "label_fake": LABEL_FAKE,
        "max_edge_length_km": MAX_EDGE_LENGTH_KM,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    csv_columns = [
      "step",
        "loss_d",
        "loss_d_real",
        "loss_d_fake",
        "loss_g",
        "score_real",
        "score_fake_d",
        "real_logit_mean",
        "fake_logit_mean",
        "score_fake_g",
        "grad_norm_d",
        "grad_norm_g",
        "real_mean",
        "real_std",
        "real_min",
        "real_max",
        "real_near_zero_frac",
        "real_near_one_frac",
        "real_clip_frac",
        "fake_d_mean",
        "fake_d_std",
        "fake_d_min",
        "fake_d_max",
        "fake_d_near_zero_frac",
        "fake_d_near_one_frac",
        "fake_g_mean",
        "fake_g_std",
        "fake_g_min",
        "fake_g_max",
        "fake_g_near_zero_frac",
        "fake_g_near_one_frac",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

    loss_d_history = []
    loss_g_history = []
    score_real_history = []
    score_fake_d_history = []
    score_fake_g_history = []

    print(f"[INFO] Logging to {log_dir}")
    print(f"[INFO] Starting comparable training for {TRAINING_STEPS} steps...")
    print(f"[CONFIG] LOSS_TYPE={LOSS_TYPE} (mapped to qGAN-style semantics)")
    print(f"[CONFIG] LATENT_DISTRIBUTION={LATENT_DISTRIBUTION}")

    for step in range(1, TRAINING_STEPS + 1):
        d_runs = []

        for _ in range(int(DISC_STEPS_PER_GEN)):
          d_runs.append(
                train_discriminator_step(
                    generator=generator,
                    discriminator=discriminator,
                    optimizer_d=optimizer_d,
                    valid_real_tuples=valid_real_tuples,
                    rng_obj=rng,
                    batch_size=BATCH_SIZE,
                )
            )
        d_metrics = {
            key: float(np.mean([run[key] for run in d_runs]))
            for key in d_runs[0].keys()
        }
        if step > int(DISC_WARMUP_STEPS):
            g_metrics = train_generator_step(
                generator=generator,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                batch_size=BATCH_SIZE,
            )
        else:
            g_metrics = {
                "loss_g": float("nan"),
                "score_fake_g": float("nan"),
                "grad_norm_g": float("nan"),
                "fake_g_mean": float("nan"),
                "fake_g_std": float("nan"),
                "fake_g_min": float("nan"),
                "fake_g_max": float("nan"),
                "fake_g_near_zero_frac": float("nan"),
                "fake_g_near_one_frac": float("nan"),
            }

        loss_d_history.append(d_metrics["loss_d"])
        loss_g_history.append(g_metrics["loss_g"])
        score_real_history.append(d_metrics["score_real"])
        score_fake_d_history.append(d_metrics["score_fake_d"])
        score_fake_g_history.append(g_metrics["score_fake_g"])

        row = {
            "step": step,
            **{k: d_metrics.get(k, float("nan")) for k in d_metrics.keys()},
            **{k: g_metrics.get(k, float("nan")) for k in g_metrics.keys()},
        }

        if step % 100 == 0 or step == 1:
            print(
                f"[Step {step:5d}] "
                f"Loss_D: {d_metrics['loss_d']:.6f} | "
                f"Loss_G: {g_metrics['loss_g']:.6f} | "
                f"Score_Real: {d_metrics['score_real']:.4f} | "
                f"Score_Fake_D: {d_metrics['score_fake_d']:.4f} | "
                f"Score_Fake_G: {g_metrics['score_fake_g']:.4f} | "
                f"GradNorm_D: {d_metrics['grad_norm_d']:.6f} | "
                f"GradNorm_G: {g_metrics['grad_norm_g']:.6f} | "
                f"RealClipFrac: {d_metrics['real_clip_frac']:.4f} | "
                f"RealMean: {d_metrics['real_mean']:.4f} | "
                f"FakeMean: {g_metrics['fake_g_mean']:.4f} | "
                f"Loss_D_Real: {d_metrics['loss_d_real']:.6f} | "
                f"Loss_D_Fake: {d_metrics['loss_d_fake']:.6f} | "
                f"RealLogit: {d_metrics['real_logit_mean']:.4f} | "
                f"FakeLogit: {d_metrics['fake_logit_mean']:.4f} | "
            )


        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writerow({col: row.get(col, float("nan")) for col in csv_columns})

        if step % 500 == 0:
            torch.save(
                {"generator_state_dict": generator.state_dict()},
                log_dir / f"generator_step_{step:06d}.pt",
    )

    torch.save(
    {"generator_state_dict": generator.state_dict()},
    log_dir / "generator_final.pt",
)
    print("\n=== TRAINING SUMMARY ===")
    print(f"Final Loss D: {loss_d_history[-1]:.6f}")
    print(f"Final Loss G: {loss_g_history[-1]:.6f}" if not np.isnan(loss_g_history[-1]) else "Final Loss G: NaN")
    print(f"Final Score Real: {score_real_history[-1]:.4f}")
    print(f"Final Score Fake D: {score_fake_d_history[-1]:.4f}")
    print(
        f"Final Score Fake G: {score_fake_g_history[-1]:.4f}"
        if not np.isnan(score_fake_g_history[-1])
        else "Final Score Fake G: NaN"
    )
    print(f"Mean Loss D: {np.nanmean(loss_d_history):.6f}")
    print(f"Mean Loss G: {np.nanmean(loss_g_history):.6f}")
    print(f"[INFO] Metrics saved to {csv_path}")


if __name__ == "__main__":
    main()
