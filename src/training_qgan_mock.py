"""
Mock Adversarial Training Loop für QuGAN - FAST VERSION:
- Zeigt Training-Loop Struktur funktioniert
- Verwendet schnelle Dummy Scores statt echte Quantum Circuits
"""

import os
import json
import numpy as np
from datetime import datetime
import csv

# Import config
from config import LEARNING_RATE, BATCH_SIZE, TRAINING_STEPS, SEED, N_CITIES

# Set seeds for reproducibility
np.random.seed(SEED)
rng = np.random.RandomState(SEED)


def load_real_edges():
    """Load cities from archive and compute all pairwise edge lengths"""
    import math
    archive_path = "archive/small.csv"
    
    cities = []
    if os.path.exists(archive_path):
        with open(archive_path) as f:
            for line in f:
                try:
                    x, y = map(float, line.strip().split(','))
                    cities.append((x, y))
                except ValueError:
                    pass
    
    if len(cities) < 2:
        return np.random.uniform(0.5, 5000, size=1000)
    
    edges = []
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            x1, y1 = cities[i]
            x2, y2 = cities[j]
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            edges.append(dist)
    
    return np.array(edges)


def create_batch_real(edges, batch_size, rng):
    """Create batch of real edges"""
    if len(edges) < 6:
        return rng.uniform(0, 1, size=(batch_size, 6))
    
    max_start = len(edges) - 6
    if max_start <= 0:
        return rng.uniform(0, 1, size=(batch_size, 6))
    
    start_indices = rng.choice(max_start, size=batch_size, replace=True)
    batch = np.array([edges[i:i+6] for i in start_indices])
    return batch / (np.max(batch) + 1e-10)  # Normalize to [0, 1]


def train_discriminator_step(disc_state, batch_real, batch_fake, rng):
    """Mock discriminator training step"""
    # Simulate disc learning from batches
    # Disc should give higher scores to real edges
    real_scores = rng.normal(0.6, 0.1, size=len(batch_real))
    fake_scores = rng.normal(0.4, 0.1, size=len(batch_fake))
    real_scores = np.clip(real_scores, 0, 1)
    fake_scores = np.clip(fake_scores, 0, 1)
    
    loss_real = np.mean((real_scores - 1.0) ** 2)
    loss_fake = np.mean((fake_scores - 0.0) ** 2)
    loss_total = loss_real + loss_fake
    
    # Update disc state parameter
    grad = rng.normal(0.05, 0.02)
    disc_state['param'] -= LEARNING_RATE * grad
    
    return loss_total, np.abs(grad), real_scores, fake_scores


def train_generator_step(disc_state, gen_state, batch_size, rng):
    """Mock generator training step"""
    # Generate fake samples
    batch_fake = rng.uniform(0, 1, size=(batch_size, 6))
    
    # Discriminator evaluates fakes
    # As gen trains, fake scores should move toward 1.0
    fake_scores = np.clip(rng.normal(gen_state['quality'], 0.15, size=batch_size), 0, 1)
    
    # Generator loss: wants disc to output 1.0 for fakes
    loss = np.mean((1.0 - fake_scores) ** 2)
    
    # Update gen state
    grad = rng.normal(0.03, 0.02)
    gen_state['param'] -= LEARNING_RATE * grad
    gen_state['quality'] += 0.01  # Improve quality over time
    gen_state['quality'] = min(gen_state['quality'], 0.7)  # Asymptote at 0.7
    
    return loss, np.abs(grad), batch_fake, fake_scores


def main():
    print("=" * 60)
    print("QuGAN Adversarial Training Loop (MOCK - FAST VERSION)")
    print("=" * 60)
    
    # Load real edges
    import math
    real_edges = load_real_edges()
    print(f"\nLoaded {len(real_edges)} real edges")
    
    # Initialize mock states
    disc_state = {'param': 0.0}
    gen_state = {'param': 0.0, 'quality': 0.45}
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/qgan_mock_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Save config
    config = {
        "MODE": "MOCK (Fast)",
        "LEARNING_RATE": LEARNING_RATE,
        "BATCH_SIZE": BATCH_SIZE,
        "TRAINING_STEPS": TRAINING_STEPS,
        "SEED": SEED,
        "N_CITIES": N_CITIES,
        "TIMESTAMP": timestamp
    }
    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # CSV logging
    csv_path = f"{log_dir}/metrics.csv"
    csv_header = [
        "step", 
        "disc_loss", "disc_grad_norm",
        "gen_loss", "gen_grad_norm",
        "real_score_mean", "real_score_std", "real_score_min", "real_score_max",
        "fake_score_mean_disc", "fake_score_std_disc", "fake_score_min_disc", "fake_score_max_disc",
        "fake_score_mean_gen", "fake_score_std_gen", "fake_score_min_gen", "fake_score_max_gen"
    ]
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
    
    print(f"\nTraining directory: {log_dir}")
    print(f"Training for {TRAINING_STEPS} steps\n")
    
    # Training loop
    for step in range(1, TRAINING_STEPS + 1):
        # Create batches
        batch_real = create_batch_real(real_edges, BATCH_SIZE, rng)
        noise_batch = rng.uniform(0, 1, size=(BATCH_SIZE, 6))
        batch_fake_gen = rng.uniform(0, 1, size=(BATCH_SIZE, 6))
        
        # Train Discriminator
        disc_loss_total, disc_grad_norm, real_scores, fake_scores_disc = train_discriminator_step(
            disc_state, batch_real, batch_fake_gen, rng
        )
        
        # Train Generator
        gen_loss_val, gen_grad_norm, batch_fake_final, fake_scores_gen = train_generator_step(
            disc_state, gen_state, BATCH_SIZE, rng
        )
        
        # Logging
        if step % 10 == 0 or step == 1:
            print(f"Step {step}:")
            print(f"  Disc Loss: {disc_loss_total:.6f}")
            print(f"  Gen Loss: {gen_loss_val:.6f}")
            print(f"  Real Scores: mean={real_scores.mean():.4f}, min={real_scores.min():.4f}, max={real_scores.max():.4f}")
            print(f"  Fake Scores (Disc): mean={fake_scores_disc.mean():.4f}, min={fake_scores_disc.min():.4f}, max={fake_scores_disc.max():.4f}")
            print(f"  Fake Scores (Gen): mean={fake_scores_gen.mean():.4f}, min={fake_scores_gen.min():.4f}, max={fake_scores_gen.max():.4f}")
            print(f"  Disc Grad Norm: {disc_grad_norm:.6f}, Gen Grad Norm: {gen_grad_norm:.6f}\n")
        
        # Save metrics to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                disc_loss_total, disc_grad_norm,
                gen_loss_val, gen_grad_norm,
                real_scores.mean(), real_scores.std(), real_scores.min(), real_scores.max(),
                fake_scores_disc.mean(), fake_scores_disc.std(), fake_scores_disc.min(), fake_scores_disc.max(),
                fake_scores_gen.mean(), fake_scores_gen.std(), fake_scores_gen.min(), fake_scores_gen.max()
            ])
    
    print(f"\nTraining complete!")
    print(f"Results saved to: {log_dir}")
    print(f"Metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()
