# QUERVERZEICHNIS: Call Graph & Data Flow

Dieses Dokument zeigt **WER WAS AUFRUFT** und **WOHER WERTE KOMMEN** für das komplette QuGAN-System.

---

## 📍 ENTRY POINTS (Wo Training startet)

### 1. `python main.py` (Nicht aktiv für QuGAN)
- Legacy-Skript für Generator-only training
- Calls: `training_qgen.run()`
- Deprecated für QuGAN

### 2. `python training_qgan.py` (AKTIV - Haupttraining)

```
training_qgan.py
├─ main()
│  ├─ Parse args (10000 steps, etc.)
│  ├─ Set seed (SEED=42)
│  ├─ Init disc = QDiscriminator(2 layers)
│  ├─ Init gen = QGenerator(2 layers)
│  ├─ Load cities = load_cities("cities.csv")
│  ├─ Load all_edges_km = compute_edge_lengths(cities)
│  │
│  └─ FOR step in 0..10000:
│     ├─ create_batch_real(cities, 16, rng) → (16, 6) array
│     ├─ gen.batch_forward(noise) → (16, 6) fake edges
│     ├─ IF step < 50: [WARMUP]
│     │  └─ FOR d in 0..5:
│     │     └─ train_discriminator_step(disc, real, fake)
│     │
│     ├─ ELSE: [ADVERSARIAL]
│     │  ├─ FOR d in 0..5:
│     │  │  └─ train_discriminator_step(...)
│     │  │
│     │  └─ train_generator_step(disc, gen, noise)
│     │
│     └─ log_metrics(step, losses, gradients, scores)
```

---

## 🔗 FUNKTIONS-CALL-GRAPH

### Kanal A: Real Data Loading

```
load_cities("cities.csv")
│
├─ Liest CSV-Datei mit 80 Städten
├─ Columns: city, country, lat, lon
├─ Encoding-Handling: utf-8, cp1252, latin-1
├─ Rückgabe: list[dict] with 80 entries
│  └─ Beispiel: {"name": "Paris", "lat": 48.8566, "lon": 2.3522}
│
└─ Verbrauch: create_batch_real(cities, ...)


load_distance_cache("distance_cache.csv")
│
├─ Reads PRE-COMPUTED Haversine distances
├─ C(80,2) = 3160 city pairs with distances
├─ Format: Dict[Tuple[str, str], float]
│  └─ Key: ("city1", "city2") alphabetically sorted
│  └─ Value: distance in km
├─ Rückgabe: Dict with all 3160 pairs
│  └─ Example: cache[("Berlin", "Paris")] = 877 km
│
└─ Verbrauch: create_batch_real() uses this once per epoch


sample_edges_from_cache(sampled_cities, cache, rng)
│
├─ For 4 sampled cities, create 6 pair keys
├─ Lookup each pair in pre-computed cache (fast!)
│  ├─ Use _pair_key() to ensure alphabetical consistency
│  ├─ cache[("Berlin", "Madrid")] → 1824 km
│  ├─ cache[("Berlin", "Rom")] → 1534 km
│  ├─ ...etc for all 6 pairs
│
├─ Rückgabe: np.array (6,) with distances in km
│  └─ For 4 cities: C(4,2) = 6 pairs
│
└─ Verbrauch: create_batch_real() calls this once per sample
```

### Kanal B: Batch Creation (Real Edges from Cache)

```
create_batch_real(cities, batch_size=16, cache, rng) [CALLED EVERY STEP]
│
├─ FOR i in 1..16:
│  │
│  ├─ rng.choice(80, size=4, replace=False)
│  │  └─ Selects 4 random city indices from 80
│  │
│  ├─ sampled_cities = [cities[idx] for idx in sampled_indices]
│  │  └─ Extract 4 city dicts
│  │
│  ├─ sample_edges_from_cache(sampled_cities, cache, rng)
│  │  ├─ Creates 6 pair keys for the 4 cities
│  │  ├─ Looks up each pair in PRE-COMPUTED cache dictionary
│  │  └─ Returns (6,) array of distances in km
│  │     └─ 6 pairs = C(4,2)
│  │
│  ├─ edges_km / MAX_EDGE_LENGTH_KM (5000.0)
│  │  └─ Normalizes to [0, 1]
│  │
│  └─ np.clip(edges_normalized, 0, 1)
│     └─ Clipping (safety)
│
└─ Stack all 16 samples → (16, 6) array
   └─ Values in [0, 1], ready for QDiscriminator
```

**Example for one sample (Step 100, Sample 1):**
```
sampled cities: [idx=5, idx=23, idx=47, idx=61]
               = [Paris, Berlin, Madrid, Rome]

Pair keys created:
  1. ("Berlin", "Madrid")  → cache lookup → 1824 km
  2. ("Berlin", "Paris")   → cache lookup → 877 km
  3. ("Berlin", "Rome")    → cache lookup → 1534 km
  4. ("Madrid", "Paris")   → cache lookup → 1265 km
  5. ("Madrid", "Rome")    → cache lookup → 1786 km
  6. ("Paris", "Rome")     → cache lookup → 1435 km

edges_km = [1824, 877, 1534, 1265, 1786, 1435]  (from cache, not computed!)
edges_norm = [0.3648, 0.1754, 0.3068, 0.2530, 0.3572, 0.2870]  (÷5000)
```

---

### Kanal C: Quantum Circuits

#### C1: QDiscriminator Circuit

```
QDiscriminator.__init__(n_layer=2, seed=0)
│
├─ self.n_qubits = 6
├─ self.n_layer = 2
├─ self.dev = qml.device("default.qubit", wires=6, shots=None)
├─ self.weights = rng.normal(0, 0.1, size=(2, 6, 3))
│  └─ Shape: (layers=2, qubits=6, gates=3)
│  └─ Initialized random, small std=0.1
│
└─ @qml.qnode: circuit(edge_weights, weights)
   │
   ├─ INPUT: edge_weights shape (6,) ∈ [0, 1]
   │         weights shape (2, 6, 3)
   │
   ├─ qml.AngleEmbedding(edge_weights * π, wires=[0..5], rotation="Y")
   │  └─ Applies RY(edge_weights[i] * π) on qubit i
   │
   ├─ FOR layer in 0..1:
   │  │
   │  ├─ FOR qubit in 0..5:
   │  │  ├─ qml.RX(weights[layer, qubit, 0], wires=qubit)
   │  │  ├─ qml.RY(weights[layer, qubit, 1], wires=qubit)
   │  │  └─ qml.RZ(weights[layer, qubit, 2], wires=qubit)
   │  │
   │  └─ FOR qubit in 0..5:
   │     └─ qml.CNOT(wires=[qubit, (qubit+1) % 6])
   │
   └─ MEASURE: <Z₀> → returns z ∈ [-1, 1]


disc.forward(edges, verbose=False)
│
├─ Calls: circuit(edges, self.weights)
│         → returns z ∈ [-1, 1]
│
├─ Converts: P(real) = 0.5 * (z + 1) → [0, 1]
│
└─ Returns: float ∈ [0, 1]
   └─ Probability that edges are from real sample


disc.batch_forward(batch, use_numpy=False)
│
├─ Input: batch shape (16, 6)
│
├─ Calls: circuit(batch, self.weights) with broadcasting
│         → PennyLane handles (B, 6) → (B,) output
│
├─ Converts: all z values to probabilities
│
└─ Returns: (16,) array of probabilities
```

#### C2: QGenerator Circuit

```
QGenerator.__init__(n_layer=2, seed=0)
│
├─ self.n_qubits = 6
├─ self.n_layer = 2
├─ self.dev = qml.device("default.qubit", wires=6, shots=None)
├─ self.weights = rng.normal(0, 0.1, size=(2, 6, 3))
│
└─ @qml.qnode: circuit(noise_vector, weights)
   │
   ├─ INPUT: noise_vector shape (6,) ∈ [0, 1]
   │         weights shape (2, 6, 3)
   │
   ├─ qml.AngleEmbedding(noise_vector * π, wires=[0..5], rotation="Y")
   │  └─ Applies RY(noise_vector[i] * π) on qubit i
   │
   ├─ FOR layer in 0..1:
   │  │
   │  ├─ FOR qubit in 0..5:
   │  │  ├─ qml.RX(weights[layer, qubit, 0], wires=qubit)
   │  │  ├─ qml.RY(weights[layer, qubit, 1], wires=qubit)
   │  │  └─ qml.RZ(weights[layer, qubit, 2], wires=qubit)
   │  │
   │  └─ FOR qubit in 0..5:
   │     └─ qml.CNOT(wires=[qubit, (qubit+1) % 6])
   │
   └─ MEASURE: <Z₀>, <Z₁>, ..., <Z₅> → returns [z₀, ..., z₅]
              each zᵢ ∈ [-1, 1]


gen.forward(noise_vector)
│
├─ Input: noise_vector shape (6,)
│
├─ Calls: circuit(noise_vector, self.weights)
│         → returns 6 z-values ∈ [-1, 1]
│
├─ Converts: edges = 0.5 * (z + 1) → [0, 1]
│
└─ Returns: (6,) array ∈ [0, 1]
   └─ 6 generated edge lengths


gen.batch_forward(noise_batch)
│
├─ Input: noise_batch shape (16, 6)
│        Each row is independent noise sample
│
├─ Calls: circuit(noise_batch, self.weights) with broadcasting
│         → PennyLane evaluates (16, 6) → (16, 6) z-values
│
├─ Converts: all z values to edge lengths [0, 1]
│
└─ Returns: (16, 6) array
   └─ 16 generated 6-edge samples
```

---

### Kanal D: Training Steps

#### D1: Discriminator Training

```
train_discriminator_step(disc, batch_real, batch_fake, rng)
│  [CALLED: 5×/step after warmup, 5×/step during warmup]
│
├─ INPUT:
│  ├─ batch_real: (16, 6) float
│  ├─ batch_fake: (16, 6) float (from generator)
│  └─ rng: random number generator
│
├─ combined_batch = vstack([batch_real, batch_fake]) → (32, 6)
│
├─ combined_labels = [0.9, ..., 0.9, 0.0, ..., 0.0]
│                  → (32,) with label smoothing
│
├─ Shuffle: random permutation
│
├─ Define loss function:
│  │
│  └─ loss_inner(weights):
│     │
│     ├─ preds = _disc_probs_from_edges(disc, weights, combined_batch)
│     │  │
│     │  ├─ Calls: disc.circuit(combined_batch, weights)
│     │  │         with broadcasting
│     │  │
│     │  └─ Returns: (32,) probabilities
│     │
│     ├─ Returns: _loss_fn(preds, labels, loss_type="log")
│     │  │
│     │  └─ loss = - mean(
│     │           labels * log(preds) +
│     │           (1 - labels) * log(1 - preds)
│     │         )
│     │    (Binary Cross-Entropy with label smoothing)
│     │
│     └─ Value: ~0.3-0.5 (training)
│
├─ Compute gradient:
│  │
│  ├─ grad_fn = qml.grad(loss_inner)
│  │  └─ Analytic gradient via automatic differentiation
│  │
│  ├─ grad = grad_fn(disc.weights)
│  │  └─ Shape: (2, 6, 3) same as weights
│  │
│  └─ grad_norm = ||grad||_F
│     └─ Frobenius norm, ~0.01-0.05
│
├─ Update weights:
│  │
│  └─ disc.weights -= DISC_LEARNING_RATE (0.05) * grad
│
└─ RETURNS: (loss: float, grad_norm: float)
   └─ Logged to CSV
```

#### D2: Generator Training

```
train_generator_step(disc, gen, batch_size, noise_batch)
│  [CALLED: 1×/step after warmup]
│
├─ INPUT:
│  ├─ disc: trained discriminator
│  ├─ gen: generator to update
│  ├─ batch_size: 16
│  └─ noise_batch: (16, 6) uniform random ∈ [0, 1]
│
├─ Define loss function:
│  │
│  └─ loss_inner(gen_weights):
│     │
│     ├─ batch_gen = _gen_edges_from_noise(gen, gen_weights, noise_batch)
│     │  │
│     │  ├─ Calls: gen.circuit(noise_batch, gen_weights)
│     │  │         with broadcasting
│     │  │
│     │  ├─ z_values = (16, 6) array ∈ [-1, 1]
│     │  │
│     │  ├─ edges = 0.5 * (z_values + 1)
│     │  │
│     │  └─ Returns: (16, 6) array ∈ [0, 1]
│     │
│     ├─ fake_probs = _disc_probs_from_edges(disc, disc.weights, batch_gen)
│     │  │
│     │  ├─ Calls: disc.circuit(batch_gen, disc.weights)
│     │  │
│     │  └─ Returns: (16,) probabilities
│     │           [discriminator's assessment of gen samples]
│     │
│     ├─ Goal: Make fake_probs → 1.0 (fool discriminator)
│     │
│     ├─ loss = - mean(log(fake_probs))
│     │  │       [standard BCE for generator: wants disc to output 1]
│     │  │
│     │  └─ If fake_probs=[0.3, 0.4, 0.5, ...]
│     │     loss = - mean(log(...)) ≈ 0.8-1.0
│     │
│     └─ Value: ~0.5-1.0 (training)
│
├─ Compute gradient:
│  │
│  ├─ grad_fn = qml.grad(loss_inner)
│  │  └─ Chain rule through disc.circuit!
│  │
│  ├─ grad = grad_fn(gen.weights)
│  │  └─ Shape: (2, 6, 3)
│  │
│  └─ grad_norm = ||grad||_F
│     └─ ~0.005-0.02
│
├─ Update weights:
│  │
│  └─ gen.weights -= GEN_LEARNING_RATE (0.01) * grad
│
└─ RETURNS: (loss: float, grad_norm: float)
   └─ Logged to CSV
```

---

## 🎯 DATEN-FLOW MATRICES

### Pro Trainingsschritt (Step s = 100)

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 100 (Post-Warmup)                                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ 1. REAL BATCH CREATION                                             │
│    ├─ Input:  cities (80)                                         │
│    ├─ Sample: 16 × [4 cities sampled]                             │
│    ├─ Compute: 16 × [6 distances (km)]                           │
│    ├─ Normalize: ÷ 5000 → [0, 1]                                  │
│    └─ Output: batch_real (16, 6)                                  │
│              Values: [0.234, 0.567, 0.123, ...]                  │
│                                                                     │
│ 2. FAKE BATCH GENERATION (from Generator)                         │
│    ├─ Input: noise_batch = rng.random((16, 6)) ∈ [0,1]^(16,6)   │
│    ├─ Circuit: 6-qubit, 2 layers, 36 params                      │
│    ├─ Output: 16 × 6 z-values ∈ [-1, 1]                          │
│    ├─ Convert: (z + 1) / 2 → [0, 1]                              │
│    └─ Output: batch_fake (16, 6)                                  │
│              Values: [0.456, 0.789, 0.234, ...]                  │
│                                                                     │
│ 3. DISCRIMINATOR UPDATES (5 iterations)                           │
│    ├─ Combine: [batch_real, batch_fake] → (32, 6)                │
│    ├─ Labels: [0.9×16, 0.0×16] → (32,)                           │
│    ├─ For d=1..5:                                                 │
│    │  ├─ Forward: disc.circuit(combined) → (32,) ∈ [0,1]        │
│    │  ├─ Loss: BCE(preds, labels) ≈ 0.30-0.50                   │
│    │  ├─ Backward: ∇_w loss → (2,6,3) gradients                 │
│    │  └─ Update: w -= 0.05 × grad                                │
│    │                                                               │
│    └─ Output: disc_loss, disc_grad_norm                           │
│              (Final values after 5 updates)                       │
│                                                                     │
│ 4. GENERATOR UPDATE (1 iteration)                                 │
│    ├─ Forward: gen.circuit(noise_batch) → (16, 6) z-values       │
│    ├─ Forward: disc.circuit(gen_edges) → (16,) probs             │
│    ├─ Loss: -mean(log(fake_probs)) ≈ 0.50-1.00                  │
│    ├─ Backward: ∇_w loss → (2,6,3) gradients                    │
│    ├─ Update: w -= 0.01 × grad                                   │
│    │                                                               │
│    └─ Output: gen_loss, gen_grad_norm                            │
│                                                                     │
│ 5. LOGGING                                                         │
│    ├─ step = 100                                                  │
│    ├─ disc_loss = 0.298                                          │
│    ├─ disc_grad_norm = 0.0187                                    │
│    ├─ gen_loss = 0.502                                           │
│    ├─ gen_grad_norm = 0.0156                                     │
│    ├─ real_score_mean = 0.8234                                   │
│    ├─ fake_score_mean_disc = 0.2156                              │
│    ├─ fake_score_mean_gen = 0.6023                               │
│    └─ CSV row written to metrics.csv                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 WERT-HERKUNFTS-TABELLE

| Wert | Berechnung | Quelle | Bereich |
|------|-----------|--------|---------|
| **edges_km** | Haversine(lat1,lon1, lat2,lon2) | cities.csv | 0-20000 km |
| **edges_norm** | edges_km / 5000 | config | [0, 1] |
| **noise** | rng.random((16,6)) | numpy | [0, 1] |
| **z_values** | PennyLane <Z_i> | Quantum circuit | [-1, 1] |
| **edges_gen** | 0.5 * (z + 1) | generator.circuit | [0, 1] |
| **disc_prob** | 0.5 * (z + 1) | discriminator.circuit | [0, 1] |
| **label_real** | Config constant | config.py | 0.9 |
| **label_fake** | Config constant | config.py | 0.0 |
| **disc_loss** | BCE(preds, labels) | loss_fn | [0, 1] |
| **gen_loss** | -mean(log(probs)) | loss_fn | [0, 2] |
| **disc_grad** | ∇_w disc_loss | qml.grad | ℝ^36 |
| **gen_grad** | ∇_w gen_loss | qml.grad | ℝ^36 |
| **disc_weights** | weights - 0.05 * grad | update | ℝ^36 |
| **gen_weights** | weights - 0.01 * grad | update | ℝ^36 |

---

## 📍 Import-Abhängigkeiten

```
training_qgan.py (main execution)
│
├─ from config import:
│  ├─ LEARNING_RATE (legacy)
│  ├─ DISC_LEARNING_RATE (0.05)
│  ├─ GEN_LEARNING_RATE (0.01)
│  ├─ BATCH_SIZE (16)
│  ├─ TRAINING_STEPS (10000)
│  ├─ DISC_STEPS_PER_GEN (5)
│  ├─ DISC_WARMUP_STEPS (50)
│  ├─ LABEL_REAL (0.9)
│  ├─ LABEL_FAKE (0.0)
│  ├─ MAX_EDGE_LENGTH_KM (5000)
│  ├─ N_CITIES (4)
│  └─ LOSS_TYPE ("log")
│
├─ from discriminator import QDiscriminator
│  └─ disc = QDiscriminator(n_layer=2, seed=SEED)
│
├─ from generator import QGenerator
│  └─ gen = QGenerator(n_layer=2, seed=SEED)
│
└─ Built-in imports:
   ├─ numpy (for arrays, random sampling)
   ├─ pennylane (for quantum circuits, qml.grad)
   ├─ csv (for logging metrics)
   └─ datetime (for run timestamps)
```

---

## 🎬 Execution Flow (Pseudocode)

```python
# training_qgan.py :: main()

# === INITIALIZATION ===
rng = np.random.default_rng(SEED=42)
disc = QDiscriminator(n_layer=2, seed=SEED)  # Init 36 random weights
gen = QGenerator(n_layer=2, seed=SEED)        # Init 36 random weights
cities = load_cities("cities.csv")            # Load 80 cities

# === TRAINING LOOP ===
FOR step in range(TRAINING_STEPS):  # 0..9999
    
    # Create batch of real samples
    batch_real = create_batch_real(cities, batch_size=16, rng)
    #  Shape: (16, 6), values ∈ [0, 1]
    
    # Create batch of fake samples
    noise_batch = rng.random((16, 6))
    batch_fake = gen.batch_forward(noise_batch)
    #  Shape: (16, 6), values ∈ [0, 1]
    
    # === WARMUP PHASE (step < 50) ===
    IF step < DISC_WARMUP_STEPS:
        FOR d_step in range(DISC_STEPS_PER_GEN):
            loss_d, grad_norm_d = train_discriminator_step(
                disc, batch_real, batch_fake, rng
            )
        log_metrics(step, disc_loss=loss_d, disc_grad_norm=grad_norm_d)
        CONTINUE
    
    # === ADVERSARIAL PHASE (step >= 50) ===
    
    # Train discriminator (5 updates)
    FOR d_step in range(DISC_STEPS_PER_GEN):  # 0..4
        loss_d, grad_norm_d = train_discriminator_step(
            disc, batch_real, batch_fake, rng
        )
    
    # Train generator (1 update)
    loss_g, grad_norm_g = train_generator_step(
        disc, gen, batch_size=16, noise_batch
    )
    
    # Compute scores for metrics
    real_probs = [disc.forward(edge) for edge in batch_real]
    fake_probs_disc = [disc.forward(edge) for edge in batch_fake]
    fake_probs_gen = [disc.forward(edge) for edge in gen.batch_forward(noise_batch)]
    
    # Log metrics
    log_metrics(
        step=step,
        disc_loss=loss_d,
        disc_grad_norm=grad_norm_d,
        gen_loss=loss_g,
        gen_grad_norm=grad_norm_g,
        real_score_mean=np.mean(real_probs),
        real_score_std=np.std(real_probs),
        # ... more metrics ...
    )

print("Training complete! Metrics saved to logs/qgan_*/metrics.csv")
```

---

## ✅ Call-Graph Validation Checklist

- [x] All functions have clear inputs/outputs
- [x] All loops and conditionals documented
- [x] All data shapes verified
- [x] All hyperparameters traced to config.py
- [x] All losses properly implemented
- [x] All gradients computed via PennyLane
- [x] All weights properly updated
- [x] All metrics logged correctly

