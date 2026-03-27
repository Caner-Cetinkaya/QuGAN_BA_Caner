#!/usr/bin/env python3
"""
Detaillierter Beispiel-Durchlauf: Generator + Discriminator Training
Zeigt den vollständigen Variablenfluss mit realistischen Zahlenwerten.
"""

import numpy as np
from generator import QGen
from discriminator import QDiscriminator
from tsp import TSPDataset

print("=" * 100)
print("BEISPIEL-DURCHLAUF: QuGAN Training mit QGen + QDiscriminator")
print("=" * 100)

# ============================================================================
# TEIL 1: QGen Training (vereinfacht)
# ============================================================================
print("\n" + "=" * 100)
print("TEIL 1: QGen – Generator Training")
print("=" * 100)

print("\n[Step 0] Initialisierung:")
print("-" * 100)
gen = QGen(n_layer=2, seed=42)
print(f"  QGen erstellt mit:")
print(f"    - n_layer = 2")
print(f"    - num_qubit = 3")
print(f"    - weigths shape = {gen.weigths.shape}")
print(f"    - weigths (beispiel erste Zeile):\n{gen.weigths[0, 0, :]}")

z0 = np.array([0.3, -1.0, 0.7])
target = np.array([1/3, 1/3, 1/3])
print(f"\n  Input-Parameter:")
print(f"    - z0 (Rausch) = {z0}")
print(f"    - target-Verteilung = {target}")
print(f"    - temperature = 1.0")

print("\n[Step 1] Forward Pass (Quantum Circuit):")
print("-" * 100)
expvals = np.array(gen.quantum_circuit(z0, gen.weigths))
print(f"  gen.quantum_circuit(z0={z0}, weigths) ->")
print(f"    expvals (σZ Erwartungswerte ∈ [-1,1]) = {expvals}")

x01 = 0.5 * (expvals + 1.0)
print(f"\n  Map [-1, 1] -> [0, 1]: x01 = 0.5 * (expvals + 1.0)")
print(f"    x01 = {x01}")

probs = gen._softmax(x01, shape=1.0)
print(f"\n  Softmax(x01) ->")
print(f"    probs = {probs}")
print(f"    Summe = {probs.sum():.6f} (sollte ~1.0 sein)")

# Loss berechnen (vereinfacht: PCE)
loss = float(np.sum((probs - target)**2 / (target + 1e-9)))
print(f"\n  Loss (PCE) = Σ((probs - target)² / target)")
print(f"    = {loss:.6f}")

print("\n[Step 2] Gradient + Weight Update (Vereinfachte Demo):")
print("-" * 100)
# In echter training_qgen.py würde qml.grad verwendet, hier nur Demo
print(f"  In training_qgen.py:")
print(f"    grad_w = qml.grad(loss_inner)")
print(f"    g = grad_w(weigths)  # Gradienten shape {gen.weigths.shape}")
print(f"    weigths_new = weigths - lr * g")
print(f"    mit lr = 0.05")

# Demo: kleine Änderung
g_demo = 0.001 * np.random.randn(*gen.weigths.shape)
gen_weigths_new = gen.weigths - 0.05 * g_demo
print(f"\n  Gewichte-Änderung (Demo):")
print(f"    max(|weigths_new - weigths|) = {np.max(np.abs(gen_weigths_new - gen.weigths)):.6f}")

# ============================================================================
# TEIL 2: QDiscriminator Training
# ============================================================================
print("\n\n" + "=" * 100)
print("TEIL 2: QDiscriminator – Discriminator Training auf realen TSP-Kanten")
print("=" * 100)

print("\n[Step 0] Datensatz & Discriminator Initialisierung:")
print("-" * 100)

# Erstelle Mini-Datensatz
points = np.array([
    [0.1, 0.2],
    [0.9, 0.1],
    [0.8, 0.9],
    [0.2, 0.8],
    [0.5, 0.5],
], dtype=np.float32)
print(f"  Mini-Datensatz: {len(points)} Punkte")
print(f"    Punkte:\n{points}")

disc = QDiscriminator(n_layer=2, seed=0)
print(f"\n  QDiscriminator erstellt mit:")
print(f"    - n_layer = 2")
print(f"    - n_qubits = 4")
print(f"    - weigths shape = {disc.weigths.shape}")

print("\n[Step 1] Sample vier Kanten:")
print("-" * 100)
# Manueller Sample statt TSPDataset für Demo
idx = np.array([0, 1, 2, 3])
pts = points[idx]
rolled = np.roll(pts, -1, axis=0)
edges = np.linalg.norm(rolled - pts, axis=1)
print(f"  Zyklische Punkte: {idx}")
print(f"    p0={pts[0]}, p1={pts[1]}, p2={pts[2]}, p3={pts[3]}")
print(f"  Kanten (p0->p1, p1->p2, p2->p3, p3->p0):")
print(f"    edges = {edges}")

print(f"\n  Prüfe Dreiecksungleichung:")
is_valid = disc._check_triangle_inequality(edges)
print(f"    _check_triangle_inequality({edges}) = {is_valid}")

print("\n[Step 2] Forward Pass (Discriminator):")
print("-" * 100)
print(f"  disc.circuit(edges, weigths) wird aufgerufen:")
print(f"    1. AngleEmbedding(edges * π) -> 4-Qubit-State")
print(f"    2. StronglyEntanglingLayers(weigths) -> Verschränkung")
print(f"    3. Messung: expval(PauliZ(0)) ∈ [-1, 1]")

# Manueller Circuit-Aufruf (nur für Demo/Show)
z_exp = disc.circuit(edges, disc.weigths)
score = disc._to_prob(z_exp)
print(f"\n  Ergebnis:")
print(f"    z_exp = {z_exp:.6f} (Erwartungswert ∈ [-1,1])")
print(f"    score = 0.5*(z_exp + 1.0) = {score:.6f} (∈ [0,1])")

print("\n[Step 3] Loss & Weight Update:")
print("-" * 100)
target_score = 1.0  # Ziel: Discriminator soll für echte Kanten 1.0 geben
loss_disc = (score - target_score) ** 2
print(f"  target_label = {target_score}")
print(f"  Loss (MSE) = (score - target)² = ({score:.6f} - {target_score})²")
print(f"           = {loss_disc:.6f}")

print(f"\n  In training_qdis.py:")
print(f"    grad_w = qml.grad(loss_inner)")
print(f"    g = grad_w(weigths)  # shape {disc.weigths.shape}")
print(f"    weigths_new = weigths - lr * g  (lr=0.05)")

# ============================================================================
# TEIL 3: Zusammenfassung der Variable & Call-Chain
# ============================================================================
print("\n\n" + "=" * 100)
print("TEIL 3: Variablenfluss & Quer-Referenzen (Call-Chain)")
print("=" * 100)

print("""
main.py
  │
  └─> training_qgen.run(loss_type="pce", seed=42, steps=300)
       │
       ├─ z0 = [0.3, -1.0, 0.7]  (Default)
       ├─ target = [1/3, 1/3, 1/3]  (Default)
       │
       ├─> gen = QGen(n_layer=2, seed=42)
       │    └─ gen.weigths: Form (2, 3, 3) → wird im Loop aktualisiert
       │
       ├─ Trainingsloop (step = 1..300):
       │   ├─ expvals = gen.quantum_circuit(z0, gen.weigths)
       │   │  └─ circuit bekommt z0 (Noise) + weigths
       │   │  └─ gibt 3 Erwartungswerte zurück (∈ [-1,1])
       │   │
       │   ├─ x01 = 0.5 * (expvals + 1.0)  → [0,1]
       │   ├─ probs = softmax(x01, temperature)
       │   ├─ loss = pce_loss(probs, target)
       │   │
       │   ├─ g = grad_w(gen.weigths)  ← qml.grad
       │   └─ gen.weigths = gen.weigths - 0.05 * g
       │
       ├─ metrics.csv schreiben
       │  (step, loss, w1, w2, w3, entropy, var)
       │
       └─> run_dir (z.B. "logs/qgen_pce_20251203_120000")
            │
            └─> main.py nutzt run_dir für:
                ├─ plot_loss.py plots/metrics.csv → plot_loss.png
                ├─ plot_weights.py plots/metrics.csv → plot_weights.png
                └─ summarize_runs (summary.csv)


training_qdis.py
  │
  └─> run(file_name="tiny.csv", seed=42, steps=200, batch_size=8)
       │
       ├─> ds = TSPDataset("archive/").load()
       │    └─ ds.xy: (N, 2) Punkte normalisiert
       │
       ├─> disc = QDiscriminator(n_layer=2, seed=0)
       │    └─ disc.weigths: Form (2, 4, 3) → wird im Loop aktualisiert
       │
       ├─ Trainingsloop (step = 1..200):
       │   ├─ Sample Batch (batch_size=8):
       │   │   ├─ edges_batch = []
       │   │   ├─ while len < 8:
       │   │   │   ├─ pts, edges = ds.sample_four_edges(seed=...)
       │   │   │   ├─ if disc._check_triangle_inequality(edges):
       │   │   │   │    edges_batch.append(edges)
       │   │   │   └─ [Repeat bis 8 gültige Kanten]
       │   │
       │   ├─ Scores berechnen:
       │   │   ├─ for e in edges_batch:
       │   │   │   └─ s = disc.forward(e)
       │   │   │      └─ circuit(e, weigths) → score ∈ [0,1]
       │   │
       │   ├─ loss = pce_loss(scores, target_label=1.0)
       │   ├─ g = grad_w(disc.weigths)  ← qml.grad
       │   └─ disc.weigths = disc.weigths - 0.05 * g
       │
       ├─ metrics.csv schreiben
       │  (step, loss, score_mean, score_std)
       │
       └─> run_dir (z.B. "logs/qdis_20251203_120000")
            └─ plot_loss.py → plot_loss.png
""")

# ============================================================================
# TEIL 4: Beispiel-Metriken-Ausgabe
# ============================================================================
print("\n" + "=" * 100)
print("TEIL 4: Erwartete Ausgaben (Beispiel metrics.csv)")
print("=" * 100)

print("\ntraining_qgen.py Output (metrics.csv):")
print("""
step,loss,w1,w2,w3,entropy,var
1,0.012,0.300,0.390,0.310,1.063,0.0025
2,0.009,0.320,0.350,0.330,1.095,0.0018
3,0.007,0.333,0.333,0.334,1.099,0.0003
...
300,0.0001,0.333,0.333,0.334,1.099,0.0000
""")

print("\ntraining_qdis.py Output (metrics.csv):")
print("""
step,loss,score_mean,score_std
1,0.156,0.605,0.089
2,0.145,0.620,0.074
3,0.132,0.638,0.062
...
200,0.012,0.985,0.008
""")

print("\n" + "=" * 100)
print("Ende des Beispiel-Durchlaufs")
print("=" * 100)
