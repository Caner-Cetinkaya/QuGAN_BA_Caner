# 🎯 QUICK REFERENCE: Visual Index & Flowchart

---

## 📍 Wo Dinge aufgerufen werden

```
USER startet Training
         │
         ▼
training_qgan.py::main()
         │
         ├─ config.py (liest alle Hyperparameter)
         │  ├─ DISC_LEARNING_RATE = 0.05
         │  ├─ GEN_LEARNING_RATE = 0.01
         │  ├─ BATCH_SIZE = 16
         │  ├─ TRAINING_STEPS = 10000
         │  └─ MAX_EDGE_LENGTH_KM = 5000.0
         │
         ├─ cities.csv (lädt 80 Städte)
         │  │
         │  ├─ load_cities() → list[dict]
         │  │
         │  └─ FOR each step in 0..9999:
         │     │
         │     ├─ create_batch_real()
         │     │  │
         │     │  ├─ rng.choice(80, 4) → 4 cities
         │     │  ├─ sample_edges_from_cache() → PRE-COMPUTED distances
         │     │  ├─ ÷ 5000 → [0, 1]
         │     │  └─ RESULT: batch_real (16, 6)
         │     │
         │     ├─ gen.batch_forward(noise)
         │     │  │
         │     │  ├─ gen.circuit(noise, weights) [QUANTUM]
         │     │  │  ├─ AngleEmbedding(noise * π, rotation="Y")
         │     │  │  ├─ FOR layer in 0..1:
         │     │  │  │  ├─ RX(w[l,q,0]) per qubit
         │     │  │  │  ├─ RY(w[l,q,1]) per qubit
         │     │  │  │  ├─ RZ(w[l,q,2]) per qubit
         │     │  │  │  └─ CNOT entanglement
         │     │  │  └─ MEASURE <Z₀> .. <Z₅>
         │     │  │
         │     │  ├─ Normalize: 0.5 * (z + 1) → [0, 1]
         │     │  └─ RESULT: batch_fake (16, 6)
         │     │
         │     ├─ IF step < 50: [WARMUP]
         │     │  │
         │     │  └─ FOR d in 0..4:
         │     │     └─ train_discriminator_step()
         │     │
         │     ├─ ELSE: [ADVERSARIAL]
         │     │  │
         │     │  ├─ FOR d in 0..4:
         │     │  │  │
         │     │  │  └─ train_discriminator_step()
         │     │  │     ├─ Forward: disc.circuit(real+fake) [QUANTUM]
         │     │  │     ├─ Loss: BCE(preds, labels)
         │     │  │     ├─ Backward: qml.grad(loss) [ANALYTIC]
         │     │  │     └─ Update: weights -= 0.05 * grad
         │     │  │
         │     │  └─ train_generator_step()
         │     │     ├─ Forward: gen.circuit(noise) [QUANTUM]
         │     │     ├─ Forward: disc.circuit(gen_edges) [QUANTUM]
         │     │     ├─ Loss: -mean(log(probs))
         │     │     ├─ Backward: qml.grad(loss) [CHAIN RULE!]
         │     │     └─ Update: weights -= 0.01 * grad
         │     │
         │     └─ log_metrics() → CSV
         │        ├─ step, disc_loss, disc_grad_norm
         │        ├─ gen_loss, gen_grad_norm
         │        ├─ real_score_mean, fake_score_mean_disc
         │        └─ separation = real_mean - fake_mean
         │
         └─ DONE: 10,000 steps trained!
            └─ logs/qgan_TIMESTAMP/metrics.csv written
               └─ plot_training.py visualizes
```

---

## 🔢 Daten-Format Transformationen

```
cities.csv (80 Städte)
├─ Input: lat ∈ [-90, 90], lon ∈ [-180, 180]
│
├─ SAMPLING (create_batch_real)
│  │
│  ├─ Sample 4 cities
│  │  │
│  │  └─ PRE-COMPUTED DISTANCE CACHE (from distance_cache.csv)
│  │     ├─ Cache contains C(80,2) = 3160 pairs
│  │     ├─ Pre-computed once (Haversine during initialization)
│  │     ├─ Lookup: cache[("city1", "city2")] → distance (km)
│  │     └─ Fast retrieval instead of live calculation
│  │
│  └─ Result: 6 distances in KM (one pair → 1 edge)
│     ├─ e_01 = 877 km (Berlin-Paris) [from cache!]
│     ├─ e_02 = 1265 km (Madrid-Paris) [from cache!]
│     ├─ e_03 = 1435 km (Paris-Rome) [from cache!]
│     ├─ e_12 = 1824 km (Berlin-Madrid) [from cache!]
│     ├─ e_13 = 1534 km (Berlin-Rome) [from cache!]
│     └─ e_23 = 1786 km (Madrid-Rome) [from cache!]
│
├─ NORMALIZATION (÷ MAX_EDGE_LENGTH_KM = 5000)
│  │
│  └─ Result: Normalized edges ∈ [0, 1]
│     ├─ n_01 = 0.1754
│     ├─ n_02 = 0.2530
│     ├─ n_03 = 0.2870
│     ├─ n_12 = 0.3648
│     ├─ n_13 = 0.3068
│     └─ n_23 = 0.3572
│
└─ QUANTUM INPUT (batch shape (16, 6))
   └─ 16 Samples × 6 Edges/Sample
      └─ All values ∈ [0, 1]
         └─ Ready for Quantum Circuit Embedding
```

---

## ⚛️ Quantum Circuit Pipeline

```
DISCRIMINATOR CIRCUIT:
┌─────────────────────────────────────┐
│ INPUT: edges ∈ [0, 1]^6             │
└──────────────┬──────────────────────┘
               │
    ┌──────────▼──────────┐
    │ EMBEDDING           │
    │ AngleEmbedding      │
    │ RY(edges[i]*π)      │
    │ on qubit i          │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────────────────┐
    │ LAYER 0: RX/RY/RZ + CNOT        │
    │ weights[0, :, :] = (6,3)        │
    └──────────┬──────────────────────┘
               │
    ┌──────────▼──────────────────────┐
    │ LAYER 1: RX/RY/RZ + CNOT        │
    │ weights[1, :, :] = (6,3)        │
    └──────────┬──────────────────────┘
               │
    ┌──────────▼──────────┐
    │ MEASUREMENT         │
    │ <Z₀> → z ∈ [-1, 1] │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────────────┐
    │ NORMALIZATION               │
    │ P(real) = 0.5*(z+1)         │
    │ Output ∈ [0, 1]             │
    └──────────┬──────────────────┘
               │
┌──────────────▼─────────────────────┐
│ OUTPUT: P(sample is real)           │
│ Scalar ∈ [0, 1]                     │
└─────────────────────────────────────┘

GENERATOR CIRCUIT: (same structure, different purpose)
┌─────────────────────────────────────┐
│ INPUT: noise ∈ [0, 1]^6             │
└──────────────┬──────────────────────┘
               │
    ┌──────────▼──────────────────┐
    │ EMBEDDING + 2 LAYERS        │
    │ (same RX/RY/RZ + CNOT)      │
    │ Different weights!          │
    └──────────┬──────────────────┘
               │
    ┌──────────▼──────────────────────┐
    │ MEASUREMENT                      │
    │ <Z₀>, <Z₁>, ..., <Z₅>           │
    │ 6 expvals, each z ∈ [-1, 1]     │
    └──────────┬──────────────────────┘
               │
    ┌──────────▼──────────────────┐
    │ NORMALIZATION               │
    │ edges = 0.5*(z+1) per qubit │
    │ Output shape (6,) ∈ [0,1]^6 │
    └──────────┬──────────────────┘
               │
┌──────────────▼─────────────────────┐
│ OUTPUT: 6 generated edge lengths    │
│ Array (6,) ∈ [0, 1]                │
└─────────────────────────────────────┘
```

---

## 🎓 Hyperparameter Beziehungen

```
CONFIG.PY DEPENDENCIES:

LEARNING_RATES:
├─ DISC_LEARNING_RATE = 0.05  ──→ Wie schnell Disc lernt
├─ GEN_LEARNING_RATE = 0.01   ──→ Wie schnell Gen lernt (5× langsamer)
└─ Ratio: D:G = 5:1 (angepasst für Balance)

TRAINING:
├─ TRAINING_STEPS = 10000     ──→ Total iterations
├─ BATCH_SIZE = 16            ──→ 8 real + 8 fake
├─ DISC_STEPS_PER_GEN = 5     ──→ D updates 5× pro 1× G (aktuell: zu hoch?)
└─ DISC_WARMUP_STEPS = 50     ──→ Only D trains first 50 steps

LABEL SMOOTHING:
├─ LABEL_REAL = 0.9           ──→ "Real" not really 1.0 (stabilität)
├─ LABEL_FAKE = 0.0           ──→ "Fake" stays 0.0
└─ Effect: Verhindert Overfitting von Disc

QUANTUM:
├─ N_QUBITS = 6               ──→ Why? 4 cities → C(4,2)=6 edges
├─ N_LAYERS = 2               ──→ Expressiveness trade-off
└─ Parameters = N_QUBITS × N_LAYERS × 3 (RX, RY, RZ) = 36

DATA:
├─ MAX_EDGE_LENGTH_KM = 5000  ──→ Normalization constant
├─ N_CITIES = 4               ──→ Sample size per batch
└─ CITIES_PATH = "cities.csv" ──→ Real data source

LOSS:
├─ LOSS_TYPE = "log"          ──→ BCE (Binary Cross-Entropy)
├─ Alternatives: "mse", "pce" (pce has numerical issues)
└─ Effect: Smoother gradients than MSE
```

---

## ✅ Parameter Update Formeln

```
DISCRIMINATOR UPDATE:

1. Forward pass:
   preds = disc.circuit(combined_batch, weights)  ∈ (32,) [0,1]
   
2. Loss computation:
   loss = - mean(labels * log(preds) + 
                 (1-labels) * log(1-preds))
   
3. Gradient:
   grad = ∇_weights loss    ∈ (2,6,3)
   grad_norm = ||grad||_F
   
4. Update:
   weights_new = weights - DISC_LEARNING_RATE * grad
              = weights - 0.05 * grad


GENERATOR UPDATE:

1. Forward pass:
   batch_gen = gen.circuit(noise, weights) ∈ (16,6) [0,1]
   fake_probs = disc.circuit(batch_gen, disc.weights) ∈ (16,)
   
2. Loss computation:
   loss = - mean(log(fake_probs))
   [Goal: Make fake_probs → 1.0]
   
3. Gradient (THROUGH DISCRIMINATOR!):
   grad = ∇_gen_weights loss
        = ∇_gen [- mean(log(disc.circuit(gen.circuit(...))))]
   [Chain rule: gen → disc → loss]
   
4. Update:
   weights_new = weights - GEN_LEARNING_RATE * grad
              = weights - 0.01 * grad  [5× langsamer!]
```

---

## 📊 Metriken Interpretation

```
SCORES (Discriminator Output):

real_score_mean = mean(disc.forward(batch_real))
├─ Ideal: 1.0 (disc says "definitely real")
├─ Current (Step 100): 0.8824 ✅ (sehr gut!)
├─ Random guess: 0.5
└─ Bad: < 0.5 (disc confused)

fake_score_mean_disc = mean(disc.forward(batch_fake))
├─ Ideal: 0.0 (disc says "definitely fake")
├─ Current (Step 100): 0.2764 ✅ (gut!)
├─ Random guess: 0.5
└─ Bad: > 0.5 (disc fooled)

fake_score_mean_gen = mean(disc.forward(gen.batch_forward(...)))
├─ Goal: increase over training (gen improving)
├─ Current (Step 100): 0.5743 (moderate)
├─ Early: ~0.5 (random)
└─ Late goal: > 0.7 (gen improved)

SEPARATION = real_score_mean - fake_score_mean_disc
├─ Ideal: 1.0 (perfect discrimination)
├─ Current (Step 100): 0.6060 ✅ (good!)
├─ Random: 0.0
└─ Interpretation: Disc learned significant bias


LOSS (Decrease = Learning):

disc_loss:
├─ Should: decrease monotonically (or plateau)
├─ Current trend (Step 100): 0.50 → 0.29 ✅
├─ Typical range: 0.2 - 0.8
└─ If explodes: numerical instability

gen_loss:
├─ Should: decrease over training (gen improving)
├─ Current trend (Step 100): 0.505 (moderate)
├─ Typical range: 0.3 - 1.0
└─ If stuck at 0.693 (log(2)): disc too strong


GRADIENTS (Stability Check):

grad_norm = ||∇_w loss||_F
├─ Should: stay in reasonable range (0.001 - 0.1)
├─ Current: disc ~0.013, gen ~0.016 ✅
├─ If too small: learning stalled
├─ If too large: instability, NaN risk
└─ Plot: watch for sudden spikes
```

---

## 🔄 Trainings-Phasen

```
PHASE 1: WARMUP (Step 0-50)
┌────────────────────────────────┐
│ Nur Discriminator trainiert    │
├────────────────────────────────┤
│ FOR step in 0..49:             │
│   batch_real = sample_real()   │
│   batch_fake = gen.forward()   │
│   FOR d in 0..4:               │
│     train_discriminator_step() │
│   [Generator frozen]           │
│   log only disc_loss           │
└────────────────────────────────┘

PHASE 2: ADVERSARIAL (Step 50-9999)
┌────────────────────────────────┐
│ Beide Netzwerke trainieren     │
├────────────────────────────────┤
│ FOR step in 50..9999:          │
│   batch_real = sample_real()   │
│   batch_fake = gen.forward()   │
│                                │
│   [5 Discriminator Updates]    │
│   FOR d in 0..4:               │
│     train_discriminator_step() │
│                                │
│   [1 Generator Update]         │
│   train_generator_step()       │
│                                │
│   [Logging]                    │
│   log all metrics to CSV       │
└────────────────────────────────┘

Why this structure?
├─ Warmup: Disc muss zuerst lernen zu klassifizieren
├─ Dann: Gen kann anfangen zu lernen (von trained Disc)
└─ Result: Mehr stabile Konvergenz
```

---

## 🐛 Bug-Fix Summary

```
BUG #1: GENERATOR BATCH EMBEDDING
├─ Symptom: Training stuck at 0.5 separation (random guessing)
├─ Cause: Old code used manual RY loop:
│         for i in range(6):
│             qml.RY(noise_vector[i] * π, wires=i)
│         
│         With batch shape (B, 6):
│         noise_vector[i] treated as i-th sample, not i-th element!
│
├─ Fix: Use AngleEmbedding with broadcasting:
│       qml.AngleEmbedding(noise_vector * π, wires=range(6), rotation="Y")
│
└─ Validation: 100-step test showed separation jump 0.5 → 0.12 ✅


BUG #2: INCONSISTENT NORMALIZATION
├─ Symptom: Some runs normalized by different constants
├─ Cause: edges_km / max_edges_in_batch (variable!)
├─ Fix: Always divide by MAX_EDGE_LENGTH_KM = 5000.0
└─ Effect: Consistent input distribution


BUG #3: "pce" LOSS EXPLOSION
├─ Symptom: Loss → Inf when fake labels = 0
├─ Cause: loss = mean((preds - targets)² / targets)
│         When targets=0: division by zero!
│
├─ Fix: Denominator guarding:
│       denom = np.where(targets > eps, targets + eps, 1.0)
│       loss = mean((preds - targets)² / denom)
│
└─ Alternative: Use "log" (BCE) loss instead
```

---

## 🎯 Nächste Schritte

```
SHORT TERM (aktuell):
├─ Reduce DISC_STEPS_PER_GEN from 5 → 2
│  └─ Grund: Gen degradiert über 10k steps (trend: -0.1072)
│
├─ Monitor: fake_score_mean_gen should ↑ not ↓
│
└─ Run: 10k steps mit neuem setting

MEDIUM TERM (nächste Woche):
├─ Hyperparameter sweep:
│  ├─ Try DISC_STEPS_PER_GEN ∈ {1, 2, 3, 5, 10}
│  ├─ Try LR ratios: different D:G ratios
│  └─ Try warmup lengths
│
├─ Extended runs: 50k or 100k steps
│
└─ Ablation studies: remove label smoothing, etc.

LONG TERM (wenn Zeit):
├─ Other loss types: Wasserstein, Hinge
├─ Better architectures: more layers, qubits
├─ Real TSP evaluation: actual tour quality
└─ Scaling: Größere Probleme (6, 8 Städte)
```

---

## 📋 Dokumentation Filemap

```
README_DOCUMENTATION.md (this index)
├─ Quick Links to all docs
├─ Schnelleinstieg (5/20/45 min)
└─ Top 10 Talking Points

ARCHITECTURE_AND_WALKTHROUGH.md ⭐ START HERE
├─ High-Level Überblick
├─ Module & Call Graph
├─ Data Flows (1-5)
├─ Config & Hyperparameter
├─ Training Loop (Pseudocode)
└─ Quantum Circuits + Bug Fixes

CALL_GRAPH_AND_DATA_FLOW.md
├─ Detailed Entry Points
├─ Function Call Graph (Kanäle A-D)
├─ Data Flow Matrices
├─ Value Origin Table
└─ Execution Flow Pseudocode

NUMERICAL_EXAMPLE_STEP_100.md
├─ Concrete city data (Paris-Berlin-Madrid-Rom)
├─ Phase 1: Real batch sampling (Haversine)
├─ Phase 2: Generator forward (z-values)
├─ Phase 3: Disc training (5 updates)
├─ Phase 4: Gen training (1 update)
├─ Phase 5: Metrics & logging
└─ Full step 100 CSV row

QUICK_REFERENCE.md (this file!)
├─ Visual Index & Flowchart
├─ Data transformations
├─ Quantum Circuit Pipeline
├─ Hyperparameter Relationships
├─ Update Formulas
├─ Metrics Interpretation
└─ Training Phases
```

---

**Für Fragen: Siehe relevante Dokumentation oder durchrechne NUMERICAL_EXAMPLE_STEP_100.md!**

