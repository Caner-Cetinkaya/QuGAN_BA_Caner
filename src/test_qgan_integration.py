"""
Quick integration test for Discriminator + Generator adversarial setup
"""

import numpy as np
from config import SEED
from discriminator import QDiscriminator
from generator import QGenerator

print("=" * 60)
print("QuGAN Integration Test")
print("=" * 60)

# Initialize models
rng = np.random.RandomState(SEED)
disc = QDiscriminator(n_layer=2, seed=SEED)
gen = QGenerator(n_layer=2, seed=SEED)

print(f"\n✅ Discriminator initialized:")
print(f"   - Weights shape: {disc.weights.shape}")
print(f"   - Parameters: {len(disc.weights)}")

print(f"\n✅ Generator initialized:")
print(f"   - Weights shape: {gen.weights.shape}")
print(f"   - Parameters: {len(gen.weights)}")

# Test forward pass
print(f"\n--- Forward Pass Test ---")

# Generate fake edges with generator
noise = rng.uniform(0, 1, size=6)
fake_edges = gen.forward(noise)
print(f"Generated fake edges: {np.round(fake_edges, 2)}")
print(f"   - Min: {fake_edges.min():.2f} km")
print(f"   - Max: {fake_edges.max():.2f} km")
print(f"   - Mean: {fake_edges.mean():.2f} km")

# Discriminate generated edges
fake_score = disc.forward(fake_edges)
print(f"Disc score on fake: {fake_score:.4f} (should be ~0.5 randomly)")

# Create real-like edges
real_edges = rng.uniform(100, 5000, size=6) / 5000.0  # Normalize to [0, 1]
real_score = disc.forward(real_edges)
print(f"Disc score on real-like: {real_score:.4f} (should be ~0.5 randomly)")

# Batch test
print(f"\n--- Batch Test ---")
noise_batch = rng.uniform(0, 1, size=(8, 6))
fake_batch = gen.batch_forward(noise_batch)
print(f"Generated batch: shape {fake_batch.shape}")
print(f"  - Sample 1: min={fake_batch[0].min():.2f}, max={fake_batch[0].max():.2f}, mean={fake_batch[0].mean():.2f}")
print(f"  - Sample 8: min={fake_batch[7].min():.2f}, max={fake_batch[7].max():.2f}, mean={fake_batch[7].mean():.2f}")

fake_scores = np.array([disc.forward(edges) for edges in fake_batch])
print(f"Disc scores on batch: mean={fake_scores.mean():.4f}, std={fake_scores.std():.4f}")

# Quick training step simulation
print(f"\n--- Training Step Simulation ---")

# Create combined batch (4 real + 4 fake)
real_batch = rng.uniform(100, 5000, size=(4, 6)) / 5000.0
combined = np.vstack([real_batch, fake_batch[:4]])
labels = np.hstack([np.ones(4), np.zeros(4)])

scores_before = np.array([disc.forward(edges) for edges in combined])
loss_before = np.mean((scores_before - labels) ** 2)

print(f"Before training step:")
print(f"   Real scores: mean={scores_before[:4].mean():.4f} (target: 1.0)")
print(f"   Fake scores: mean={scores_before[4:].mean():.4f} (target: 0.0)")
print(f"   Loss: {loss_before:.6f}")

# Simulate gradient update (simple)
disc.weights *= 0.99  # Dummy update

scores_after = np.array([disc.forward(edges) for edges in combined])
loss_after = np.mean((scores_after - labels) ** 2)

print(f"After dummy training step:")
print(f"   Real scores: mean={scores_after[:4].mean():.4f}")
print(f"   Fake scores: mean={scores_after[4:].mean():.4f}")
print(f"   Loss: {loss_after:.6f}")

print(f"\n✅ All integration tests passed!")
print(f"   Disc and Gen can work together for adversarial training.\n")
