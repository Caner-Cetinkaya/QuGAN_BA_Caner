"""
Standalone Generator Training Loop (wie training_qdis.py für Discriminator)

Zweck:
- Trainiere nur QGenerator auf 10.000 Schritte
- Teste, ob Generator realistisch aussehende Kanten generiert
- Baseline: Wie verhält sich Generator ohne Discriminator-Feedback?

Architektur:
- Generator: 6 Qubits, 2 Layer, 36 Parameter
- Input: Random Noise ∈ [0,1]^6
- Output: 6 normalisierte Kanten ∈ [0,1]
- Loss: MSE gegenüber echten Edge-Verteilung

Ausgabe:
- logs/qgen_TIMESTAMP/
  ├─ config.json (Hyperparameter)
  ├─ metrics.csv (Step, Loss, Grad-Norm, etc.)
  └─ plot_qgen_training.py (Visualisierung)
"""

import os
import json
import csv
import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from datetime import datetime
from pathlib import Path

from config import (
    LEARNING_RATE,
    GEN_LEARNING_RATE,
    BATCH_SIZE,
    TRAINING_STEPS,
    DEVICE_NAME,
    SEED,
    MAX_EDGE_LENGTH_KM,
    LOSS_TYPE,
)
from generator import QGenerator
from training_qgan import load_cities, load_distance_cache, sample_edges_from_cache


def mse_loss(preds, target):
    """Mean Squared Error"""
    return pnp.mean((preds - target) ** 2)


def pce_loss(preds, target, eps=1e-9):
    """Pearson Chi-Square Error: sum((pred-target)^2 / target)"""
    return pnp.sum((preds - target) ** 2 / (target + eps))


def mae_loss(preds, target):
    """Mean Absolute Error"""
    return pnp.mean(pnp.abs(preds - target))


def log_loss(preds, target, eps=1e-12):
    """Cross-Entropy / Log Loss"""
    return -pnp.mean(target * pnp.log(pnp.clip(preds, eps, 1.0)))


def _loss_fn(preds, targets, loss_type: str):
    """Wähle Loss-Funktion basierend auf loss_type"""
    if loss_type == "mse":
        return mse_loss(preds, targets)
    elif loss_type == "pce":
        return pce_loss(preds, targets)
    elif loss_type == "mae":
        return mae_loss(preds, targets)
    elif loss_type == "log":
        return log_loss(preds, targets)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Choose from: mse, pce, mae, log")


def load_real_edge_distribution(cities, cache, rng, n_samples=1000):
    """
    Sampelt n_samples reale Kanten-Samples um die empirische Verteilung zu lernen.
    
    Returns: (mean_edges, std_edges) für jede Kante (6,)
    """
    samples = []
    for _ in range(n_samples):
        success, edges_km = sample_edges_from_cache(cities, cache, rng)
        if success:
            edges_normalized = np.clip(edges_km / MAX_EDGE_LENGTH_KM, 0, 1)
            samples.append(edges_normalized)
    
    samples = np.array(samples)  # (n_samples, 6)
    mean_edges = np.mean(samples, axis=0)
    std_edges = np.std(samples, axis=0)
    
    return mean_edges, std_edges


def train_generator_step(gen, real_target, rng, loss_type="mse"):
    """
    Trainiere Generator für einen Step.
    
    Args:
        gen: QGenerator
        real_target: Target edge distribution (6,)
        rng: Random generator
        loss_type: Loss-Funktion ("mse", "pce", "mae", "log")
    
    Returns:
        loss, grad_norm
    """
    # Generiere Batch von Fake Edges
    noise_batch = rng.uniform(0, 1, size=(BATCH_SIZE, 6))
    target = pnp.array(real_target, dtype=float)
    target_batch = pnp.tile(target, (BATCH_SIZE, 1))  # (BATCH_SIZE, 6)
    
    def loss_inner(weights):
        """Loss function that depends on weights (wichtig für PennyLane!)"""
        # Nutze weights-Argument, nicht gen.weights!
        batch_fake = pnp.array(gen.batch_forward(noise_batch, weights=weights), dtype=float)
        return _loss_fn(batch_fake, target_batch, loss_type)
    
    # Explizit argnums=0 für weights-Parameter
    grad_fn = qml.grad(loss_inner, argnums=0)
    
    # Berechne Loss - sichere Konvertierung aus ArrayBox
    loss_val = loss_inner(gen.weights)
    loss = float(pnp.asarray(loss_val))
    
    # Berechne Gradient
    grad = grad_fn(gen.weights)
    
    # Debug: Check gradient shape
    grad_shape = getattr(grad, "shape", None)
    if grad_shape == () or grad_shape == (0,):
        print(f"WARNING: Gradient has unexpected shape: {grad_shape}")
        print(f"  Loss type: {loss_type}, Loss value: {loss}")
    
    # Gradient norm
    grad_norm = float(pnp.linalg.norm(grad))
    
    # Update Generator (pnp arithmetic!)
    lr = GEN_LEARNING_RATE if GEN_LEARNING_RATE is not None else LEARNING_RATE
    gen.weights = gen.weights - lr * grad
    
    return loss, grad_norm


def main():
    print("=" * 60)
    print("QGenerator Standalone Training Loop")
    print("=" * 60)
    
    # Initialize generator
    gen = QGenerator(n_layer=2, seed=SEED)
    
    # Load cities and cache
    cities = load_cities("cities.csv")
    cache = load_distance_cache("distance_cache.csv")
    print(f"Loaded {len(cities)} cities")
    print(f"Cache contains {len(cache)} pairwise distances\n")
    
    # Load real edge distribution
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)
    
    print(f"Sampling {1000} real edges to get target distribution...")
    real_mean, real_std = load_real_edge_distribution(cities, cache, rng, n_samples=1000)
    print(f"Real edge mean: {real_mean}")
    print(f"Real edge std:  {real_std}\n")
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/qgen_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Save config
    config = {
        "LEARNING_RATE": LEARNING_RATE,
        "GEN_LEARNING_RATE": GEN_LEARNING_RATE,
        "BATCH_SIZE": BATCH_SIZE,
        "TRAINING_STEPS": TRAINING_STEPS,
        "LOSS_TYPE": LOSS_TYPE,
        "DEVICE": DEVICE_NAME,
        "SEED": SEED,
        "N_CITIES": len(cities),
        "TIMESTAMP": timestamp,
        "real_edge_mean": real_mean.tolist(),
        "real_edge_std": real_std.tolist(),
    }
    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # CSV logging
    csv_path = f"{log_dir}/metrics.csv"
    csv_header = [
        "step",
        "gen_loss",
        "gen_grad_norm",
        "fake_mean_0", "fake_mean_1", "fake_mean_2",
        "fake_mean_3", "fake_mean_4", "fake_mean_5",
    ]
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
    
    print(f"Training directory: {log_dir}")
    print(f"Training for {TRAINING_STEPS} steps\n")
    
    # Training loop
    for step in range(1, int(TRAINING_STEPS) + 1):
        # Train generator
        loss, grad_norm = train_generator_step(gen, real_mean, rng, loss_type=LOSS_TYPE)
        
        # Generate sample for statistics
        noise_sample = rng.uniform(0, 1, size=(16, 6))
        fake_batch = gen.batch_forward(noise_sample)
        fake_mean = np.mean(fake_batch, axis=0)
        
        # Log metrics
        row = [step, loss, grad_norm] + fake_mean.tolist()
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # Print progress
        if step % 100 == 0:
            print(f"Step {step:5d} | Loss: {loss:.6f} | Grad: {grad_norm:.6f}")
        
        # Save checkpoint every 1000 steps
        if step % 1000 == 0:
            ckpt_path = f"{log_dir}/ckpt_{step:06d}.npy"
            np.save(ckpt_path, gen.weights)
            print(f"  → Checkpoint saved: {ckpt_path}")
    
    print(f"\n✓ Training complete!")
    print(f"Logs saved to: {log_dir}")
    
    # Final statistics
    final_noise = rng.uniform(0, 1, size=(100, 6))
    final_fake = gen.batch_forward(final_noise)
    final_mean = np.mean(final_fake, axis=0)
    final_std = np.std(final_fake, axis=0)
    
    print(f"\nFinal Generator Statistics:")
    print(f"  Mean: {final_mean}")
    print(f"  Std:  {final_std}")
    print(f"\nTarget (Real) Statistics:")
    print(f"  Mean: {real_mean}")
    print(f"  Std:  {real_std}")


if __name__ == "__main__":
    main()
