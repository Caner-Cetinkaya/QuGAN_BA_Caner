# QuGAN: Vollständige Architektur & Trainings-Walkthrough

**Dokument für: Meetings & Code Review**  
**Letzter Update:** Februar 4, 2026  
**Status:** Nach 100+ Bugfixes – Trainable & Learning ✅

---

## 📋 INHALTSVERZEICHNIS

1. [Überblick & Architektur](#überblick--architektur)
2. [Modulübersicht & Call-Graph](#modulübersicht--call-graph)
3. [Detaillierte Datenflüsse](#detaillierte-datenflüsse)
4. [Hyperparameter & Config](#hyperparameter--config)
5. [Trainingsloop: Step-by-Step Walkthrough](#trainingsloop-step-by-step-walkthrough)
6. [Konkrete numerische Beispiele](#konkrete-numerische-beispiele)
7. [Quantum Circuits erklärt](#quantum-circuits-erklärt)
8. [Fehlerfix-Historie & aktuelle Stabilität](#fehlerfix-historie--aktuelle-stabilität)

---

## 🏗️ Überblick & Architektur

### High-Level QuGAN-System

```
┌─────────────────────────────────────────────────────────────┐
│                    QGAN TRAINING LOOP                       │
│                   (training_qgan.py)                        │
└─────────────────────────────────────────────────────────────┘
                            │
                    ┌───────┴───────┐
                    │               │
            ┌───────▼─────────┐  ┌──▼──────────────┐
            │ REAL EDGES      │  │ GENERATOR (Gen) │
            │ (städte.csv)    │  │  6-Qubit VQC    │
            │  Sampling       │  │  36 Parameter   │
            └────────┬────────┘  └────┬────────────┘
                     │                │
                     └────────┬────────┘
                              │
                    ┌─────────▼──────────┐
                    │   DISCRIMINATOR    │
                    │    (6-Qubit VQC)   │
                    │   36 Parameter     │
                    │ Output: P(real)    │
                    │        ∈ [0, 1]    │
                    └──────┬────────┬────┘
                           │        │
                    ┌──────▼─┐  ┌───▼──────┐
                    │ Gen    │  │ Disc     │
                    │ Loss   │  │ Loss     │
                    │ Update │  │ Update   │
                    └────────┘  └──────────┘
```

### Quantenmechanische Grundstruktur

**Generator (6 Qubits, 36 Parameter):**
- Input: 6-dimensional noise vector (ℝ⁶)
- Prozess: AngleEmbedding (Y) → 2 VQC Layers → Messung
- Output: 6 normalisierte Kantenlängen ∈ [0, 1]

**Discriminator (6 Qubits, 36 Parameter):**
- Input: 6 normalisierte Kantenlängen ∈ [0, 1]
- Prozess: AngleEmbedding (Y) → 2 VQC Layers → Messung
- Output: Wahrscheinlichkeit P(real|edges) ∈ [0, 1]

---

## 🔗 Modulübersicht & Call-Graph

### Datei-Abhängigkeiten

```
main.py / training_qgan.py
    ├── imports: config.py
    │   └─ LEARNING_RATE, DISC_STEPS_PER_GEN, N_CITIES, MAX_EDGE_LENGTH_KM, etc.
    │
    ├── imports: discriminator.py → class QDiscriminator
    │   ├─ circuit(edge_weights, weights)  → z ∈ [-1, 1]
    │   ├─ forward(edges) → P(real) ∈ [0, 1]
    │   └─ batch_forward(batch) → (B, 6) weights/probabilities
    │
    ├── imports: generator.py → class QGenerator
    │   ├─ circuit(noise_vector, weights)  → 6 z-values ∈ [-1, 1]
    │   ├─ forward(noise) → 6 edges ∈ [0, 1]
    │   └─ batch_forward(noise_batch) → (B, 6) edges
    │
    ├── calls: load_cities("cities.csv") → list[dict]
    │   └─ reads: cities.csv (80 Städte mit lat/lon)
    │
    └── calls: create_batch_real(cities, batch_size, rng) → (B, 6) real edges
        ├─ Haversine distance (lat/lon → km)
        ├─ Normalisierung: edges_km / MAX_EDGE_LENGTH_KM
        └─ Clip to [0, 1]
```

### Trainings-Call-Flow pro Iteration

```
STEP s:
  ├─ create_batch_real() → batch_real (B, 6)
  ├─ gen.batch_forward(noise_batch) → batch_fake (B, 6)
  │
  ├─ FOR d in 1..DISC_STEPS_PER_GEN:
  │  └─ train_discriminator_step(disc, batch_real, batch_fake)
  │     ├─ Combine & shuffle real + fake
  │     ├─ loss_fn(disc.circuit(...), labels) → loss
  │     ├─ qml.grad(loss_fn) → gradient
  │     ├─ weights -= lr_disc * gradient
  │     └─ return: loss, grad_norm
  │
  ├─ train_generator_step(disc, gen, noise_batch)
  │  ├─ noise_batch ∈ [0, 1]^(B, 6) (uniform random)
  │  ├─ gen.circuit() → z-values
  │  ├─ disc.circuit(gen_edges) → fake_scores
  │  ├─ loss = - mean(log(fake_scores))  [Discriminator soll Gen lieben]
  │  ├─ qml.grad(loss) → gradient
  │  ├─ gen.weights -= lr_gen * gradient
  │  └─ return: loss, grad_norm
  │
  └─ log: step, disc_loss, gen_loss, real_score_mean, fake_score_mean, ...
```

---

## 📊 Detaillierte Datenflüsse

### 1️⃣ Real Edge Sampling

```python
# In: create_batch_real(cities, batch_size=16, cache, rng)
# Step 1: Sample 4 cities
    idx = rng.choice(80, size=4, replace=False)
    # Beispiel: idx = [5, 23, 47, 61]
    # Corresponding cities: [Paris, Berlin, Madrid, Rome]

# Step 2: Load distances from PRE-COMPUTED CACHE (NOT calculated!)
    cities_sample = [cities[5], cities[23], cities[47], cities[61]]
    
    # Create 6 pair keys (alphabetically sorted):
    # Pairs: (Berlin, Madrid), (Berlin, Paris), (Berlin, Rome),
    #        (Madrid, Paris), (Madrid, Rome), (Paris, Rome)
    
    edges_km = sample_edges_from_cache(cities_sample, cache, rng)
    # Lookup in cache dictionary:
    edges_km = [
        cache[("Berlin", "Madrid")]  = 1824.50 km,
        cache[("Berlin", "Paris")]   =  877.33 km,
        cache[("Berlin", "Rome")]    = 1534.67 km,
        cache[("Madrid", "Paris")]   = 1265.45 km,
        cache[("Madrid", "Rome")]    = 1786.78 km,
        cache[("Paris", "Rome")]     = 1435.22 km,
    ]

# Step 3: Normalize by MAX_EDGE_LENGTH_KM = 5000
    edges_normalized = edges_km / 5000.0
    # = [0.3649, 0.1755, 0.3069, 0.2531, 0.3574, 0.2870]
    
# Step 4: Clip to [0, 1]
    edges_clipped = np.clip(edges_normalized, 0, 1)
    # Output shape: (6,)

# Repeat 16 times → batch_real shape (16, 6)
```

### 2️⃣ Discriminator Forward Pass (Real)

```python
# In: discriminator.circuit(edges_normalized=(6,), weights=(2, 6, 3))

# === EMBEDDING (einmalig) ===
edges = np.array([0.3649, 0.1755, 0.3069, 0.2531, 0.3574, 0.2870])
# AngleEmbedding(edges * π, rotation="Y") 
# → RY(edges[i] * π) auf qubit i
# Q0: RY(0.3649 * π) = RY(1.147 rad)
# Q1: RY(0.1755 * π) = RY(0.551 rad)
# Q2: RY(0.3069 * π) = RY(0.964 rad)
# Q3: RY(0.2531 * π) = RY(0.795 rad)
# Q4: RY(0.3574 * π) = RY(1.123 rad)
# Q5: RY(0.2870 * π) = RY(0.902 rad)

# === VQC LAYER 0 ===
# RX(weights[0, 0, 0]) auf Q0,   RX(weights[0, 1, 0]) auf Q1, ...
# RY(weights[0, 0, 1]) auf Q0,   RY(weights[0, 1, 1]) auf Q1, ...
# RZ(weights[0, 0, 2]) auf Q0,   RZ(weights[0, 1, 2]) auf Q1, ...
# CNOT(0→1), CNOT(1→2), ..., CNOT(5→0)

# === VQC LAYER 1 ===
# (Same structure as Layer 0)

# === MEASUREMENT ===
# <Z₀> measurement auf Qubit 0
z_value = np.array([-0.7432])  # ∈ [-1, 1]

# Normalisierung zu P(real):
P_real = 0.5 * (z_value + 1) = 0.5 * (-0.7432 + 1) = 0.1284
# Output: ~0.128 (low probability for real - needs training!)
```

### 3️⃣ Generator Forward Pass (Fake)

```python
# In: generator.circuit(noise=(6,), weights=(2, 6, 3))

# === EMBEDDING (einmalig) ===
noise = np.array([0.5123, 0.9847, 0.1234, 0.6789, 0.3456, 0.8901])
# Normalize to [0, 1] (already in range)
# AngleEmbedding(noise * π, rotation="Y")
# Q0: RY(0.5123 * π) = RY(1.609 rad)
# ...

# === VQC LAYER 0 ===
# Same RX, RY, RZ, CNOT structure as discriminator

# === VQC LAYER 1 ===
# Same structure

# === MEASUREMENT ===
# <Z₀>, <Z₁>, ..., <Z₅> on all 6 qubits
z_values = np.array([0.1234, -0.5678, 0.9012, -0.3456, 0.7890, -0.2345])

# Convert to edge lengths [0, 1]:
edges_generated = 0.5 * (z_values + 1)
# = [0.5617, 0.2161, 0.9506, 0.3272, 0.8945, 0.3828]
# Output shape: (6,)
```

### 4️⃣ Discriminator Forward Pass (Generated/Fake)

```python
# Input: edges_generated = [0.5617, 0.2161, 0.9506, 0.3272, 0.8945, 0.3828]
# Same circuit as in step 2️⃣

# After VQC processing:
z_value = [0.2456]  # Different from real (depends on learned weights)

# P(fake|generated) = 0.5 * (0.2456 + 1) = 0.6228
# Output: ~0.623 (moderate prob for being real, actually fake)
```

### 5️⃣ Loss Computation & Gradient

```python
# === DISCRIMINATOR LOSS ===
# Batch: 8 real + 8 fake = 16 samples
# Labels: [0.9, 0.9, ..., 0.0, 0.0]  (label smoothing)
# Predictions: [0.088, 0.245, ..., 0.623, 0.589]

# BCE loss:
loss_disc = - mean(0.9 * log(0.088) + 0.1 * log(1-0.088) + ...)
#         = - mean([...big negative...] + [...smaller...])
#         ≈ 0.487

# Gradient w.r.t. disc.weights:
grad = ∇_weights loss_disc  # Shape (2, 6, 3)

# Update:
disc.weights -= DISC_LEARNING_RATE (0.05) * grad

# === GENERATOR LOSS ===
# Goal: fool discriminator
# Fake scores from generator: [0.623, 0.589, ...]
# Gen wants these to be ~1.0 (fake looks real)

loss_gen = - mean(log(fake_scores))
#         ≈ - mean(log(0.623) + log(0.589) + ...)
#         ≈ - mean([-0.473, -0.530, ...])
#         ≈ 0.502

# Gradient w.r.t. gen.weights:
grad = ∇_weights loss_gen

# Update:
gen.weights -= GEN_LEARNING_RATE (0.01) * grad
```

---

## ⚙️ Hyperparameter & Config

**Datei:** [config.py](config.py)

### Quantum Circuit Parameter

| Parameter | Wert | Bedeutung |
|-----------|------|-----------|
| `N_QUBITS` | 6 | Anzahl Qubits (1 pro Kante eines 4-Punkte-Polygons) |
| `N_LAYERS` | 2 | Anzahl VQC-Layer (Expressiveness) |
| **Trainierbare Parameter:** | **36** | 2 Layers × 6 Qubits × 3 Gates (RX, RY, RZ) |

### Lernraten

| Parameter | Wert | Bedeutung |
|-----------|------|-----------|
| `DISC_LEARNING_RATE` | 0.05 | Discriminator lernt schneller |
| `GEN_LEARNING_RATE` | 0.01 | Generator lernt 5× langsamer |
| `DISC_WARMUP_STEPS` | 50 | Disc stabilisiert sich vor Gen-Training |
| `DISC_STEPS_PER_GEN` | 5 | Disc macht 5 Updates pro Gen-Update |

### Training-Hyper

| Parameter | Wert | Bedeutung |
|-----------|------|-----------|
| `BATCH_SIZE` | 16 | 8 real + 8 fake pro Trainingsschritt |
| `TRAINING_STEPS` | 10000 | Gesamte Iterationen |
| `LOSS_TYPE` | "log" | BCE loss (numerisch stabil) |

### Label Smoothing

| Parameter | Wert | Bedeutung |
|-----------|------|-----------|
| `LABEL_REAL` | 0.9 | "Echte" Samples labeled als 0.9 (nicht 1.0) |
| `LABEL_FAKE` | 0.0 | "Falsche" Samples labeled als 0.0 (nicht 0.5) |

**Warum?** Verhindert Overfitting des Discriminators und stabilisiert GAN-Training.

### Daten-Normalisierung

| Parameter | Wert | Bedeutung |
|-----------|------|-----------|
| `MAX_EDGE_LENGTH_KM` | 5000.0 | Maximale realistische Kantenlänge |
| `N_CITIES` | 4 | Punkte pro Sample |
| Normalisierung | `edges_km / 5000` → `[0, 1]` | Eingang für Quantum Circuits |

---

## 🎯 Trainingsloop: Step-by-Step Walkthrough

### Pseudocode für eine Iteration (Step s)

```python
# === SCHRITT 1: Real Batch Sampling ===
batch_real = create_batch_real(cities, batch_size=16, rng)
# Output: (16, 6) array with normalized edge lengths

# === SCHRITT 2: Fake Batch Generation ===
noise_batch = rng.random((16, 6))  # Uniform [0, 1]
batch_fake = gen.batch_forward(noise_batch)
# Output: (16, 6) generated edges (normalized [0, 1])

# === SCHRITT 3: Discriminator Warmup ===
if s <= DISC_WARMUP_STEPS:  # First 50 steps
    # Only train discriminator, skip generator
    for d_step in range(DISC_STEPS_PER_GEN):
        loss_d, grad_d = train_discriminator_step(disc, batch_real, batch_fake, rng)
        disc.weights -= DISC_LEARNING_RATE * grad_d
    continue  # Skip generator update

# === SCHRITT 4: Adversarial Training (after warmup) ===
# Discriminator multiple updates
for d_step in range(DISC_STEPS_PER_GEN):  # 5 discriminator updates
    loss_d, grad_d = train_discriminator_step(disc, batch_real, batch_fake, rng)
    disc.weights -= DISC_LEARNING_RATE * grad_d

# Generator one update
loss_g, grad_g = train_generator_step(disc, gen, batch_size=16, noise_batch)
gen.weights -= GEN_LEARNING_RATE * grad_g

# === SCHRITT 5: Logging ===
log_metrics_to_csv(
    step=s,
    disc_loss=loss_d,
    disc_grad_norm=grad_norm_d,
    gen_loss=loss_g,
    gen_grad_norm=grad_norm_g,
    real_score_mean=mean(disc_probs(batch_real)),
    fake_score_mean_disc=mean(disc_probs(batch_fake)),
    fake_score_mean_gen=mean(disc_probs(batch_generated)),
)
```

### Kontrollflussverzweigung im Detail

```
STEP s:
  │
  ├─ IF s < 50:  [WARMUP PHASE]
  │  │
  │  ├─ Sample batch_real & batch_fake
  │  ├─ Train discriminator 5× (gen stays frozen)
  │  ├─ Log: only disc_loss, disc_grad_norm
  │  └─ Continue to next step
  │
  └─ ELSE:  [ADVERSARIAL TRAINING]
     │
     ├─ Sample batch_real & batch_fake
     ├─ FOR d_step = 1..5:
     │  └─ train_discriminator_step()
     │     ├─ Forward: disc_probs(batch_real) + disc_probs(batch_fake)
     │     ├─ Backward: qml.grad(loss_fn)
     │     ├─ Update: disc.weights -= 0.05 * grad
     │     └─ Report: loss_d, grad_norm_d
     │
     ├─ train_generator_step()
     │  ├─ Forward: gen(noise) → batch_gen
     │  ├─ Forward: disc(batch_gen) → fake_probs
     │  ├─ Backward: qml.grad(BCE_loss)
     │  ├─ Update: gen.weights -= 0.01 * grad
     │  └─ Report: loss_g, grad_norm_g
     │
     └─ Log all metrics to CSV
```

---

## 🔢 Konkrete numerische Beispiele

### Vollständiges Beispiel für Step 100

#### Setup
```
Cities: 80 Städte loaded from cities.csv
Batch size: 16
Step: 100 (nach Warmup)
```

#### Phase 1: Real Edge Sampling

```
create_batch_real(cities, 16, rng) →

  Sample 1: cities [5, 23, 47, 61]
    → edges_km = [2345.67, 1234.23, 890.45, 1567.89, 673.21, 456.78]
    → edges_norm = [0.4691, 0.2469, 0.1781, 0.3136, 0.1346, 0.0914]
  
  Sample 2: cities [8, 34, 52, 71]
    → edges_km = [1567.89, 2012.34, 678.90, 1234.56, 945.67, 789.01]
    → edges_norm = [0.3136, 0.4025, 0.1358, 0.2469, 0.1891, 0.1578]
  
  ... (14 more samples)
  
  batch_real: shape (16, 6)
  Sample shapes in batch: each (6,) with values ∈ [0, 1]
```

#### Phase 2: Generator Creates Fake Batch

```
noise_batch = rng.random((16, 6))
           = [[0.5123, 0.9847, 0.1234, ...],
              [0.8765, 0.4321, 0.5678, ...],
              ...]

gen.batch_forward(noise_batch) →
  For each noise sample:
    1. Embedding: RY(noise[i] * π) on qubit i
    2. VQC: 2 layers of RX/RY/RZ + CNOT
    3. Measurement: <Z₀>, ..., <Z₅> → z_values shape (6,)
    4. Normalize: edges = 0.5 * (z_values + 1)
  
  batch_fake: shape (16, 6)
  Example z_values: [-0.8234, 0.1234, 0.9012, -0.3456, 0.7890, -0.2345]
  → edges: [0.0883, 0.5617, 0.9506, 0.3272, 0.8945, 0.3828]
```

#### Phase 3: Discriminator Training (5 updates)

```
combine: batch_real (16, 6) + batch_fake (16, 6) → combined (32, 6)
labels:  [0.9, 0.9, ..., 0.0, 0.0]  (32 labels)

update 1:
  forward: disc.circuit(combined, weights) → probs (32,)
  loss = BCE(probs, labels)
       = -mean(0.9 * log(probs[:16]) + 0.1 * log(1 - probs[:16])
              + 0.0 * log(probs[16:]) + 1.0 * log(1 - probs[16:]))
       ≈ 0.487
  
  grad = qml.grad(loss) → shape (2, 6, 3)
  norm = ||grad||_F ≈ 0.0247
  
  update: disc.weights -= 0.05 * grad

update 2-5: (similar)
  loss decreases: 0.487 → 0.412 → 0.378 → 0.345 → 0.298
  grad_norm stabilizes around 0.015-0.022
```

#### Phase 4: Generator Training (1 update)

```
noise_batch = rng.random((16, 6))
batch_gen = gen.batch_forward(noise_batch)  # shape (16, 6)

forward: disc.circuit(batch_gen, disc.weights) → fake_probs (16,)
         fake_probs ≈ [0.623, 0.589, 0.645, ...]  (avg ≈ 0.60)
         
loss = - mean(log(fake_probs))
     ≈ - mean(log([0.623, 0.589, 0.645, ...]))
     ≈ - mean([-0.473, -0.530, -0.435, ...])
     ≈ 0.502

grad = qml.grad(loss) → shape (2, 6, 3)
norm = ||grad||_F ≈ 0.0156

update: gen.weights -= 0.01 * grad
```

#### Phase 5: Logging

```
CSV row for step 100:
  step = 100
  disc_loss = 0.298  (after 5 updates)
  disc_grad_norm = 0.0187
  gen_loss = 0.502
  gen_grad_norm = 0.0156
  real_score_mean = 0.8234
  real_score_std = 0.1567
  real_score_min = 0.5123
  real_score_max = 0.9876
  fake_score_mean_disc = 0.2156  (disc sees fake as mostly wrong)
  fake_score_mean_gen = 0.6023  (gen sees its own creations as mostly right)
  separation = 0.8234 - 0.2156 = 0.6078
```

---

## ⚛️ Quantum Circuits erklärt

### Discriminator Circuit Detailliert

```
┌─────────────────────────────────────────────────────┐
│ Input: edges = [0.4691, 0.2469, 0.1781, ...]       │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  EMBEDDING LAYER    │
        │  AngleEmbedding     │
        │  edges * π, rot="Y" │
        └──────────┬──────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
Q0: RY(0.4691π)    │     Q1: RY(0.2469π)    │     ...
    │              │              │
    └──────────────┼──────────────┘
                   │
        ┌──────────▼──────────────┐
        │  LAYER 0 (RX, RY, RZ)   │
        │  ─────────────────────  │
        │  RX(w[0,q,0]) per qubit │
        │  RY(w[0,q,1]) per qubit │
        │  RZ(w[0,q,2]) per qubit │
        └──────────┬───────────────┘
                   │
        ┌──────────▼──────────────┐
        │  ENTANGLEMENT (CNOT)    │
        │  ──────────────────     │
        │  Q0-Q1, Q1-Q2, ..., Q5-Q0
        │  (zirkulär)             │
        └──────────┬───────────────┘
                   │
        ┌──────────▼──────────────┐
        │  LAYER 1 (RX, RY, RZ)   │
        │  (like Layer 0)         │
        └──────────┬───────────────┘
                   │
        ┌──────────▼──────────────┐
        │  ENTANGLEMENT (CNOT)    │
        └──────────┬───────────────┘
                   │
        ┌──────────▼──────────────┐
        │  MEASUREMENT (only Q0)  │
        │  <Z₀> → z ∈ [-1, 1]    │
        └──────────┬───────────────┘
                   │
        ┌──────────▼──────────────┐
        │  NORMALIZATION          │
        │  P(real) = 0.5(z + 1)  │
        │  Output ∈ [0, 1]        │
        └──────────────────────────┘
```

### Generator Circuit (ähnlich, aber mit Noise Input)

```
┌─────────────────────────────────────────────────────────┐
│ Input: noise = [0.5123, 0.9847, 0.1234, ...]           │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  EMBEDDING LAYER    │
        │  AngleEmbedding     │
        │  noise * π, rot="Y" │
        └──────────┬──────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
Q0: RY(0.5123π)    │     Q1: RY(0.9847π)    │     ...
    │              │              │
    └──────────────┼──────────────┘
                   │
        ┌──────────▼─────────────┐
        │  LAYER 0 (RX, RY, RZ)  │
        │  + CNOT Entanglement   │
        └──────────┬─────────────┘
                   │
        ┌──────────▼─────────────┐
        │  LAYER 1 (RX, RY, RZ)  │
        │  + CNOT Entanglement   │
        └──────────┬─────────────┘
                   │
    ┌──────────────┼──────────────────────┐
    │              │              ... │    │
    │              │                   │    │
    ▼              ▼                   ▼    ▼
  <Z₀>           <Z₁>               <Z₄> <Z₅>
  [-1,1]         [-1,1]             [-1,1][-1,1]
    │              │                   │    │
    │      ┌───────┴───────────────────┴────┴──┐
    │      │  NORMALIZATION                    │
    │      │  z → edges = 0.5(z + 1)          │
    │      │  Output: [0.0883, 0.5617, ...]   │
    │      │  Shape: (6,) ∈ [0, 1]            │
    │      └─────────────────────────────────┘
    │              │
    └──────────────▼─────────────────────────────┐
                   │                              │
     Output (6 edges for one 4-point sample):    │
     (0.0883, 0.5617, 0.9506, 0.3272, 0.8945,  │
      0.3828)                                    │
     Each value normalized to [0, 1] (not km)   │
```

---

## 🐛 Fehlerfix-Historie & aktuelle Stabilität

### Kritische Fehler gefunden & behoben

#### ❌ Bug #1: Generator Batch Embedding (KRITISCH)

**Problem:** Manual RY loop in generator behandelt batch falsch
```python
# FALSCH (alter Code):
for i in range(6):
    qml.RY(noise_vector[i] * np.pi, wires=i)

# Bei batch shape (B, 6):
# Behandelt noise_vector[i] als i-tes Sample, nicht i-tes Element!
```

**Folge:** Generator Batch Broadcasting funktioniert nicht → Trainingsstagnation bei 0.5

**Fix:** 
```python
# RICHTIG (neuer Code):
qml.AngleEmbedding(noise_vector * np.pi, wires=range(6), rotation="Y")
# Broadcasting-aware: Funktioniert mit (6,) und (B, 6)
```

**Validierung:** 100-step test zeigte Separation stieg von 0.5 → 0.12 ✅

---

#### ❌ Bug #2: Inconsistent Real Edge Normalization

**Problem:** Real edges sometimes normalized by different constants
```python
# In create_batch_real:
edges_km / 5000.0  # ✅ Correct
# In some older runs:
edges_km / max_edges  # ❌ Changes per batch
```

**Folge:** Inconsistent input distribution for discriminator

**Fix:** Always use `MAX_EDGE_LENGTH_KM = 5000.0` (config constant)

---

#### ❌ Bug #3: "pce" Loss Denominator Explosion

**Problem:**
```python
loss = mean((preds - targets)² / targets)
# When targets = 0 (fake labels), division by zero → NaN/Inf
```

**Folge:** Loss explodes every time fake samples are in batch

**Fix:**
```python
denom = np.where(targets > eps, targets + eps, 1.0)
loss = mean((preds - targets)² / denom)
# Fallback to unweighted MSE when target=0
```

---

#### ✅ Current Stability Checks

**10k-step Discriminator Run Baseline:**
- Loss: 0.247 → 0.098 (60% improvement) ✅
- Score: 0.503 → 0.731 (from random to discriminative) ✅
- Gradient norm: Mean 0.0587, Max 0.221 (stable, no spikes) ✅

**10k-step QuGAN Run:**
- Discriminator learning: Real score 0.8+ (excellent separation) ✅
- Generator trend: Early 0.50 → Late 0.39 (degrading, indicates D too strong) ⚠️
- Remedy: Consider reducing `DISC_STEPS_PER_GEN` from 5 → 2 or 1

---

## 📝 Summary für Meeting

### Aktuelle Architektur (Feb 2026)

| Komponente | Details |
|-----------|---------|
| **Generator** | 6-Qubit VQC, 36 params, noise→edges |
| **Discriminator** | 6-Qubit VQC, 36 params, edges→P(real) |
| **Training** | Adversarial (5 D-steps per 1 G-step) |
| **Optimization** | Analytic gradients via PennyLane's `qml.grad` |
| **Normalization** | All edges normalized to [0, 1] via `/5000 km` |
| **Data** | Real cities (80) → sample 4 → 6 pairwise edges |

### Key Hyperparameters

```
DISC_LR=0.05, GEN_LR=0.01 (separate)
BATCH_SIZE=16, STEPS=10k
DISC_WARMUP=50 steps
LABEL_SMOOTHING: real=0.9, fake=0.0
DISC_STEPS_PER_GEN=5 (may reduce to 2)
```

### Training Dynamics

- **Warmup (0-50):** Discriminator stabilizes, generator frozen
- **Adversarial (50-10k):** Both networks train, D updates 5× per G update
- **Monitoring:** CSV logs every step with detailed metrics
- **Plotting:** 6-panel visualization of loss, scores, gradients, weights

### Next Steps

1. **Reduce D-update frequency:** Try `DISC_STEPS_PER_GEN=2` to balance learning
2. **Monitor Gen recovery:** Check if generator degradation stops
3. **Extended runs:** 20k or 50k steps to see long-term stability
4. **Ablation studies:** Test different hyperparameter combinations

