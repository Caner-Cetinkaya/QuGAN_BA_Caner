#!/usr/bin/env python
"""Quick test of QGenerator"""

import numpy as np
from generator import QGenerator

print("=" * 80)
print("QGenerator Quick Test")
print("=" * 80 + "\n")

gen = QGenerator(n_layer=2, seed=42)

# Generiere 3 Samples aus Rausch
print("\n[Test] Generiere 3 Samples aus Rausch-Vektoren:\n")
for i in range(3):
    noise = np.random.rand(6)
    print(f"\nSample {i+1}:")
    print(f"  Input noise: {noise}")
    edges = gen.forward(noise)
    print(f"  Generated edges: {edges}")
    print(f"  Range: [{edges.min():.2f}, {edges.max():.2f}] km")
    print(f"  Mean: {edges.mean():.2f} km")

print("\n" + "=" * 80)
print("✅ QGenerator Test Complete!")
print("=" * 80)

# Test batch forward with broadcasting
print("\n[Test] Batch forward (Broadcasting):\n")
noise_batch = np.random.rand(5, 6)
print(f"Input batch shape: {noise_batch.shape}")
edges_batch = gen.batch_forward(noise_batch)
print(f"Output batch shape: {edges_batch.shape}")
print(f"Output range: [{edges_batch.min():.2f}, {edges_batch.max():.2f}]")
assert edges_batch.shape == (5, 6), "Batch output shape mismatch"
assert np.all((edges_batch >= 0) & (edges_batch <= 1)), "Edges out of normalized range [0,1]"
print("✅ Batch forward test passed!")

# Test circuit output range
print("\n[Test] Circuit output range (Z-expectations in [-1, 1]):\n")
noise = np.random.rand(6)
z_vals = gen.circuit(noise, gen.weights)
z_arr = np.array(z_vals)
print(f"Z-expectation values: {z_arr}")
print(f"Range: [{z_arr.min():.4f}, {z_arr.max():.4f}]")
assert np.all((z_arr >= -1.0) & (z_arr <= 1.0)), "Z-values out of expected range [-1, 1]"
print("✅ Circuit range test passed!")

# Test edge normalization
print("\n[Test] Edge normalization ([-1,1] → [0,1]):\n")
test_z = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
test_edges = gen._to_edge_length(test_z)
expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
print(f"Z-values: {test_z}")
print(f"Normalized edges: {test_edges}")
print(f"Expected: {expected}")
assert np.allclose(test_edges, expected), "Normalization formula incorrect"
print("✅ Edge normalization test passed!")
