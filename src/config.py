"""
Zentrale Konfiguration für QuGAN Training
Alle Hyperparameter hier definieren
"""

# ============================================================================
# QUANTUM CIRCUIT HYPERPARAMETER
# ============================================================================

N_QUBITS = 6          # 6 Qubits (1 pro Kante: e_ab, e_bc, e_cd, e_da, e_ac, e_bd)
N_LAYERS = 4          # Anzahl der VQC-Layer (Hyperparameter)

# ============================================================================
# TRAINING HYPERPARAMETER
# ============================================================================

LEARNING_RATE = 0.01      # Legacy default (falls getrennte LRs nicht genutzt werden)

# Separate Lernraten (typisch: Discriminator etwas schneller als Generator)
DISC_LEARNING_RATE = 0.005
GEN_LEARNING_RATE = 0.005

# Warmup: erst D stabilisieren, dann G aktivieren
DISC_WARMUP_STEPS = 0

# Adversariales Training: wie oft D pro G-Step trainieren
DISC_STEPS_PER_GEN = 2.0  # 1 D-Step pro G-Step (1:1 Verhältnis)

# Optional: Label smoothing für stabileres GAN-Training
LABEL_REAL = 1.0
LABEL_FAKE = 0.0

BATCH_SIZE = 16           # Mini-Batch für schnellsten Test
TRAINING_STEPS = 1000
LOSS_TYPE = "log"         # "pce", "mse", "log" / bce für cGan aktuell
TARGET_LABEL = 1.0        # Label für echte Kanten (1.0 = real, 0.0 = fake)
SEED = 42                 # Reproduzierbarkeit
DEVICE_NAME = "default.qubit"  # PennyLane Device

# ============================================================================
# DATEN PFADE
# ============================================================================

CITIES_PATH = "cities.csv"
DISTANCE_CACHE_PATH = "distance_cache.csv"
LOGS_DIR = "logs"
N_CITIES = 4  # Anzahl der Städte im TSP

# ============================================================================
# NORMALISIERUNG
# ============================================================================

MAX_EDGE_LENGTH_KM = 5000.0  # Maximale Kantenlänge für Normalisierung (Embedding)

# ============================================================================
# PLOTTING & VISUALISIERUNG
# ============================================================================

ENABLE_PLOTTING = True
PLOT_EVERY_N_STEPS = 100  # Zwischenplots alle N Steps (0 = nur am Ende)
