#!/usr/bin/env python3
"""
Test-Skript für TSPDataset und QDiscriminator – demonstriert die neuen Methoden.
"""
import numpy as np
from tsp import TSPDataset
from discriminator import QDiscriminator

print("=" * 80)
print("Test 1: TSPDataset.check_triangle_inequality")
print("=" * 80)

# Test gültige Kanten (erfüllen Dreiecksungleichung)
valid_edges = np.array([1.0, 1.0, 1.0, 1.5])  # z.B. 3x1 + 1.5
is_valid = TSPDataset.check_triangle_inequality(valid_edges)
print(f"Kanten {valid_edges}: gültig? {is_valid}")
assert is_valid, "Sollte True sein!"

# Test ungültige Kanten
invalid_edges = np.array([0.1, 0.1, 0.1, 5.0])  # 0.1+0.1+0.1 < 5.0
is_valid = TSPDataset.check_triangle_inequality(invalid_edges)
print(f"Kanten {invalid_edges}: gültig? {is_valid}")
assert not is_valid, "Sollte False sein!"

print("\n" + "=" * 80)
print("Test 2: TSPDataset.sample_four_edges_flat")
print("=" * 80)

# Erstelle einen Mini-Datensatz (uniform random)
points = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [0.5, 0.5]
], dtype=np.float32)

ds = TSPDataset(zip_path="dummy", file_name="dummy.csv")
ds._xy = points  # manual setzen für Test

pts, edges = ds.sample_four_edges(seed=42)
print(f"Punkte form: {pts.shape}, Kanten form: {edges.shape}")
print(f"Punkte:\n{pts}")
print(f"Kanten: {edges}")

edges_flat = ds.sample_four_edges_flat(seed=42)
print(f"Flache Kanten: {edges_flat}")
assert np.allclose(edges, edges_flat), "Sollten identisch sein!"

print("\n" + "=" * 80)
print("Test 3: QDiscriminator mit sample_four_edges")
print("=" * 80)

disc = QDiscriminator(n_layer=2, seed=0)

# Forward pass mit gültigen Kanten
test_edges = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
try:
    score = disc.forward(test_edges)
    print(f"Score für Kanten {test_edges}: {score:.4f}")
    assert 0.0 <= score <= 1.0, f"Score sollte in [0,1] sein, ist aber {score}"
except Exception as e:
    print(f"Fehler: {e}")

print("\n" + "=" * 80)
print("Test 4: QDiscriminator._check_triangle_inequality")
print("=" * 80)

# Gültig
valid = disc._check_triangle_inequality(np.array([1.0, 1.0, 1.0, 1.5]))
print(f"[1.0, 1.0, 1.0, 1.5]: {valid}")
assert valid

# Ungültig
invalid = disc._check_triangle_inequality(np.array([0.1, 0.1, 0.1, 5.0]))
print(f"[0.1, 0.1, 0.1, 5.0]: {invalid}")
assert not invalid

print("\n" + "=" * 80)
print("Alle Tests erfolgreich! ✓")
print("=" * 80)
