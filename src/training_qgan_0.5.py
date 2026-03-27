"""
Adversarial Training Loop für QuGAN:
- Discriminator trainiert auf Real vs Generated Samples
- Generator trainiert um Discriminator zu täuschen
"""

import os
import json
import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from datetime import datetime
import csv
import math
from pathlib import Path

# Import config and models
from config import (
    LEARNING_RATE,
    DISC_LEARNING_RATE,
    GEN_LEARNING_RATE,
    BATCH_SIZE,
    TRAINING_STEPS,
    DEVICE_NAME,
    N_LAYERS,
    SEED,
    N_CITIES,
    MAX_EDGE_LENGTH_KM,
    LOSS_TYPE,
    DISC_STEPS_PER_GEN,
    DISC_WARMUP_STEPS,
    LABEL_REAL,
    LABEL_FAKE,
)
from discriminator import QDiscriminator
from generator import QGenerator

# Set seeds for reproducibility
rng = np.random.default_rng(SEED)

# Patch-only knobs for this variant
GEN_INIT_STD = 0.3
DISC_HEAD_INIT_STD = 1.0


def _loss_fn(preds, targets, loss_type: str):
    """Differentiable loss used for both discriminator and generator steps."""
    "Input: Vorhersagen, Zielweret und der Loss Type"
    "Output: Loss (skalar)"
    if loss_type == "mse":
        return pnp.mean((preds - targets) ** 2)

    if loss_type == "pce":
        # NOTE: This project historically defined "pce" as a weighted MSE:
        #   (pred-target)^2 / (target + eps)
        # That works only when targets are strictly > 0.
        # In adversarial (real=1, fake=0) training, fake targets include 0,
        # which would blow up the loss by ~1/eps.
        eps = 1e-9
        denom = pnp.where(targets > eps, targets + eps, 1.0)
        return pnp.mean((preds - targets) ** 2 / denom)

    if loss_type == "log":
        eps = 1e-12
        preds = pnp.clip(preds, eps, 1.0 - eps)
        return -pnp.mean(targets * pnp.log(preds) + (1.0 - targets) * pnp.log(1.0 - preds))

    raise ValueError(f"Unknown LOSS_TYPE={loss_type!r}. Use 'mse', 'pce', or 'log'.")


def load_cities(path: str = "cities.csv"):
    """Load cities from cities.csv"""
    "Lädt Städte aus csv und gibt Liste von :{name, country, lat, lon} zurück"
    "Ist aktuell robuster als es sein müsste, da ursprünglich andere CVS genutzt wurde jetzt aber die eigen Erstellte"
    import pandas as pd
    
    read_kwargs = dict(sep=';', decimal=',', dtype=str)
    df = None
    
    candidates = [Path(path)]
    script_dir = Path(__file__).parent
    if not Path(path).is_absolute():
        candidates.append(script_dir / path)
        candidates.append(script_dir / 'archive' / Path(path).name)
    
    for candidate in candidates:
        if not candidate.exists():
            continue
        for enc in (None, 'utf-8-sig', 'cp1252', 'latin-1'):
            try:
                if enc is None:
                    df = pd.read_csv(candidate, **read_kwargs)
                else:
                    df = pd.read_csv(candidate, encoding=enc, **read_kwargs)
                break
            except:
                continue
        if df is not None:
            break
    
    if df is None:
        raise FileNotFoundError(f"cities.csv not found")
    
    # Process columns
    df.columns = [c.strip().lower() for c in df.columns]
    if 'latitude' in df.columns:
        df = df.rename(columns={'latitude': 'lat'})
    if 'longitude' in df.columns:
        df = df.rename(columns={'longitude': 'lon'})
    
    for col in ('lat', 'lon'):
        df[col] = df[col].astype(str).str.replace(r'\s+', '', regex=True).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    cities = []
    for _, r in df.iterrows():
        cities.append({
            'name': str(r["city"]).strip(),
            'country': str(r["country"]).strip(),
            'lat': float(r["lat"]),
            'lon': float(r["lon"])
        })
    
    return cities


def _pair_key(a, b):
    """Stabiler Cache-Key für Stadtpaar (alphabetically sorted)"""
    "Gibt das Städtepaar aus wobei Reihenfolge egal"
    # Format: "CityName|Country"
    ka = f"{a['name']}|{a['country']}"
    kb = f"{b['name']}|{b['country']}"
    return tuple(sorted((ka, kb)))


def load_distance_cache(cache_path="distance_cache.csv"):
    """Lädt Distance-Cache aus CSV."""
    "Lädt die Distanz zwichen key:(k1,k2) aus distance_cache.csv"
    cache = {}
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache nicht gefunden: {cache_path}")
    with open(cache_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = (row["k1"], row["k2"])
            cache[k] = float(row["distance_km"])
    return cache


def sample_edges_from_cache(cities, cache, rng):
    """
    Sampelt 4 zufällige Städte bildet die 6 Distanzen und holt die 6 Kanten aus dem Cache.
    Returns: (success, edges) wobei success=True wenn alle 6 Kanten im Cache vorhanden sind.
    """
    idx = rng.choice(len(cities), size=4, replace=False)
    a, b, c, d = [cities[i] for i in idx]
    
    pairs = [
        _pair_key(a, b),  # e_ab
        _pair_key(b, c),  # e_bc
        _pair_key(c, d),  # e_cd
        _pair_key(d, a),  # e_da
        _pair_key(a, c),  # e_ac
        _pair_key(b, d),  # e_bd
    ]
    
    # Prüfe ob alle Kanten im Cache vorhanden sind
    missing = [i for i, pair in enumerate(pairs) if pair not in cache]
    if missing:
        return False, None
    
    edges = np.array([cache[pair] for pair in pairs], dtype=float)
    return True, edges


def create_batch_real(cities, batch_size, cache, rng):
    """Create batch of REAL edges: Sample 4 cities, lookup 6 pairwise edges from cache"""
    "Erstellt einen Batch aus den 4 Städten und der Output ist (B,6) also sprich die 6 normalisiertenKanten"
    batch = []
    max_retries = batch_size * 10  # Prevent infinite loops
    attempt = 0
    
    while len(batch) < batch_size and attempt < max_retries:
        success, edges_km = sample_edges_from_cache(cities, cache, rng)
        attempt += 1
        
        if not success:
            # Retry if edges not in cache
            continue

        # CONSISTENT normalization: always divide by MAX_EDGE_LENGTH_KM
        edges_normalized = np.clip(edges_km / MAX_EDGE_LENGTH_KM, 0.0, 1.0)
        batch.append(edges_normalized.astype(float))
    
    if len(batch) < batch_size:
        raise RuntimeError(f"Could not create full batch: got {len(batch)}/{batch_size} samples after {max_retries} attempts")
    
    return np.array(batch)


def _sigmoid(x):
    return 1.0 / (1.0 + pnp.exp(-x))


def _ensure_disc_head(disc):
    if not hasattr(disc, "head_w"):
        init = DISC_HEAD_INIT_STD * disc.rng.standard_normal(disc.n_qubits)
        disc.head_w = pnp.ones(disc.n_qubits, dtype=float)
    if not hasattr(disc, "head_b"):
        disc.head_b = pnp.array(0.0, dtype=float)


def _patch_discriminator_multireadout(disc: QDiscriminator):
    """Replace the original single-qubit readout with 6 readouts for training."""
    @qml.qnode(disc.dev, interface="autograd", diff_method="backprop")
    def circuit(edge_weights, weights):
        qml.AngleEmbedding(edge_weights * np.pi, wires=range(disc.n_qubits), rotation="Y")
        qml.AngleEmbedding(edge_weights * np.pi, wires=range(disc.n_qubits), rotation="Z")

        for layer in range(disc.n_layer):
            for qubit in range(disc.n_qubits):
                qml.RX(weights[layer, qubit, 0], wires=qubit)
            for qubit in range(disc.n_qubits):
                qml.RY(weights[layer, qubit, 1], wires=qubit)
            for qubit in range(disc.n_qubits):
                qml.RZ(weights[layer, qubit, 2], wires=qubit)
            for qubit in range(disc.n_qubits):
                target = (qubit + 1) % disc.n_qubits
                qml.CNOT(wires=[qubit, target])
            for qubit in range(disc.n_qubits):
                target = (qubit + 2) % disc.n_qubits
                qml.CNOT(wires=[qubit, target])

        return [qml.expval(qml.PauliZ(i)) for i in range(disc.n_qubits)]

    disc.circuit = circuit


def _reinit_generator_weights(gen: QGenerator, std: float = GEN_INIT_STD):
    gen.weights = gen.rng.normal(0, std, size=(gen.n_layer, gen.n_qubits, 3))


#def _disc_probs_from_edges(disc: QDiscriminator, weights, edges_batch_raw):
def _disc_probs_from_edges(disc: QDiscriminator, q_weights, edges_batch, head_w=None, head_b=None):
    """Differentiable discriminator probabilities for a batch of 6-edge vectors."""
    z = disc.circuit(edges_batch, q_weights)

    if isinstance(z, (list, tuple)):
        z = pnp.stack(z, axis=-1)

    if getattr(z, "ndim", None) == 1:
        # single sample readout -> (1, n_qubits)
        z = z[pnp.newaxis, :]

    batch_size = getattr(edges_batch, "shape", [None])[0]
    if getattr(z, "ndim", None) == 2 and z.shape[0] == disc.n_qubits and z.shape[1] == batch_size:
        z = pnp.transpose(z)

    if head_w is None or head_b is None:
        _ensure_disc_head(disc)
        head_w = disc.head_w if head_w is None else head_w
        head_b = disc.head_b if head_b is None else head_b

    logits = pnp.dot(z, head_w) + head_b
    return _sigmoid(logits)


def _gen_edges_from_noise(gen: QGenerator, gen_weights, noise_batch_raw):
    """Differentiable generator outputs (normalized edges in [0,1]) for a noise batch."""
    "Erstellt ein Batch aus Noise Output: (B,6) aber diesmal Fake im Bereich [0,1]"
    noise_batch = pnp.array(noise_batch_raw, dtype=float)
    #noise_batch = pnp.clip(noise_batch, 0.0, 1.0)

    # Broadcasting: gen.circuit returns 6 expectation streams for the whole batch.
    z_values = gen.circuit(noise_batch, gen_weights)
    batch_size = noise_batch.shape[0]

    if isinstance(z_values, (list, tuple)):
        # List[expval] with broadcasting: each element has shape (B,)
        z_arr = pnp.stack(z_values, axis=1)  # (B, 6)
    else:
        #z_arr = pnp.asarray(z_values)
        z_arr = z_values
        # Expected shapes:
        # - (6, B)
        # - (B, 6)
        if z_arr.ndim == 2 and z_arr.shape[0] == 6 and z_arr.shape[1] == batch_size:
            z_arr = pnp.transpose(z_arr)

    edges = 0.5 * (z_arr + 1.0)
    #return pnp.clip(edges, 0.0, 1.0)
    return edges

"""
# Legacy (ohne analytische Gradienten, nur zum Vergleich):
def disc_loss_real(disc, batch_real):
    # Discriminator loss on real samples (should output ~1.0)
    scores = np.array([disc.forward(edges, verbose=False) for edges in batch_real])
    loss = np.mean((scores - 1.0) ** 2)
    return loss, scores


def disc_loss_fake(disc, batch_fake):
    # Discriminator loss on fake samples (should output ~0.0)
    scores = np.array([disc.forward(edges, verbose=False) for edges in batch_fake])
    loss = np.mean((scores - 0.0) ** 2)
    return loss, scores """

def train_discriminator_step(disc, batch_real, batch_fake):
    """Single discriminator step with separated real/fake losses and a trainable classical head."""

    batch_real_p = pnp.array(batch_real, dtype=float)
    batch_fake_p = pnp.array(batch_fake, dtype=float)

    targets_real = pnp.full(len(batch_real_p), LABEL_REAL, dtype=float)
    targets_fake = pnp.full(len(batch_fake_p), LABEL_FAKE, dtype=float)

    _ensure_disc_head(disc)

    disc.weights = pnp.array(disc.weights, dtype=float, requires_grad=True)
    disc.head_w = pnp.array(disc.head_w, dtype=float, requires_grad=True)
    disc.head_b = pnp.array(disc.head_b, dtype=float, requires_grad=True)

    def loss_inner(q_weights, head_w, head_b):
        real_preds = _disc_probs_from_edges(disc, q_weights, batch_real_p, head_w, head_b)
        fake_preds = _disc_probs_from_edges(disc, q_weights, batch_fake_p, head_w, head_b)

        loss_real = _loss_fn(real_preds, targets_real, LOSS_TYPE)
        loss_fake = _loss_fn(fake_preds, targets_fake, LOSS_TYPE)
        return 0.5 * (loss_real + loss_fake)

    grad_fn = qml.grad(loss_inner, argnum=[0, 1, 2])
    grad_q, grad_hw, grad_hb = grad_fn(disc.weights, disc.head_w, disc.head_b)

    lr = DISC_LEARNING_RATE if DISC_LEARNING_RATE is not None else LEARNING_RATE
    disc.weights = disc.weights - lr * grad_q
    disc.head_w = disc.head_w - lr * grad_hw
    disc.head_b = disc.head_b - lr * grad_hb

    grad_norm = float(
        pnp.sqrt(
            pnp.sum(grad_q ** 2) +
            pnp.sum(grad_hw ** 2) +
            grad_hb ** 2
        )
    )

    real_preds_after = _disc_probs_from_edges(disc, disc.weights, batch_real_p, disc.head_w, disc.head_b)
    fake_preds_after = _disc_probs_from_edges(disc, disc.weights, batch_fake_p, disc.head_w, disc.head_b)

    loss_real_after = float(_loss_fn(real_preds_after, targets_real, LOSS_TYPE))
    loss_fake_after = float(_loss_fn(fake_preds_after, targets_fake, LOSS_TYPE))
    loss_after = 0.5 * (loss_real_after + loss_fake_after)

    return {
        "disc_loss": loss_after,
        "disc_loss_real": loss_real_after,
        "disc_loss_fake": loss_fake_after,
        "disc_grad_norm": grad_norm,
    }
"""
def train_discriminator_step(disc, batch_real, batch_fake, rng):
    Single discriminator training step using analytic gradients via qml.grad.
    1. Kombiniert real + fake zu einem großen Batch (vstack)
    2. Labels: real=1, fake=0
    3. Shufflen
    4. disc.weights auf pnp.array(..., requires_grad=True)
    5. Definiert loss_inner(weights):
        preds = _disc_probs_from_edges(...)
        loss = _loss_fn(preds, labels)
    6. grad_fn = qml.grad(loss_inner)
    7. Update: disc.weights = disc.weights - lr * grad
       
    D lernt real→1 und fake→0; Update passiert hier; G bleibt konstant.
    --> Gen Eingefroren
    
    # Combine batches
    combined_batch = np.vstack([batch_real, batch_fake])
    combined_labels = np.hstack([
        np.full(len(batch_real), LABEL_REAL, dtype=float),
        np.full(len(batch_fake), LABEL_FAKE, dtype=float),
    ])
    
    # Shuffle
    idx = rng.permutation(len(combined_batch))
    combined_batch = combined_batch[idx]
    combined_labels = combined_labels[idx]
    
    # Compute loss
    combined_labels_p = pnp.array(combined_labels, dtype=float)

    # Ensure weights are differentiable
    disc.weights = pnp.array(disc.weights, dtype=float, requires_grad=True)

    def loss_inner(weights):
        preds = _disc_probs_from_edges(disc, weights, combined_batch)
        return _loss_fn(preds, combined_labels_p, LOSS_TYPE)

    grad_fn = qml.grad(loss_inner)
    loss = float(loss_inner(disc.weights))
    grad = grad_fn(disc.weights)

    # Update all parameters
    lr = DISC_LEARNING_RATE if DISC_LEARNING_RATE is not None else LEARNING_RATE
    disc.weights = disc.weights - lr * grad
    grad_norm = float(pnp.linalg.norm(grad))

    return loss, grad_norm

"""
def train_generator_step(disc, gen, noise_batch):
    """Single generator training step using the patched discriminator readout + head."""

    _ensure_disc_head(disc)

    gen.weights = pnp.array(gen.weights, dtype=float, requires_grad=True)
    disc.weights = pnp.array(disc.weights, dtype=float)
    disc.head_w = pnp.array(disc.head_w, dtype=float)
    disc.head_b = pnp.array(disc.head_b, dtype=float)

    def loss_inner(gen_weights):
        batch_fake = _gen_edges_from_noise(gen, gen_weights, noise_batch)
        scores = _disc_probs_from_edges(disc, disc.weights, batch_fake, disc.head_w, disc.head_b)
        targets = pnp.full(scores.shape, LABEL_REAL, dtype=float)
        return _loss_fn(scores, targets, LOSS_TYPE)

    grad_fn = qml.grad(loss_inner)
    loss = float(loss_inner(gen.weights))
    grad = grad_fn(gen.weights)

    lr = GEN_LEARNING_RATE if GEN_LEARNING_RATE is not None else LEARNING_RATE
    gen.weights = gen.weights - lr * grad
    grad_norm = float(pnp.linalg.norm(grad))

    batch_fake_np = np.array(_gen_edges_from_noise(gen, gen.weights, noise_batch), dtype=float)
    return loss, grad_norm, batch_fake_np


def main():
    print("=" * 60)
    print("QuGAN Adversarial Training Loop")
    print("=" * 60)
    
    # Initialize models
    disc = QDiscriminator(n_layer=N_LAYERS, seed=SEED)
    gen = QGenerator(n_layer=N_LAYERS, seed=SEED)

    # Patch discriminator to use all 6 readouts + trainable classical head
    _patch_discriminator_multireadout(disc)
    _ensure_disc_head(disc)

    # Optional: stronger generator init so fake samples do not start too close to 0.5 everywhere
    _reinit_generator_weights(gen, std=GEN_INIT_STD)
    
    # Load cities and distance cache
    cities = load_cities("cities.csv")
    cache = load_distance_cache("distance_cache.csv")
    print(f"Loaded {len(cities)} cities")
    print(f"Cache contains {len(cache)} pairwise distances")
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/qgan_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Save config
    config = {
        "LEARNING_RATE": LEARNING_RATE,
        "DISC_LEARNING_RATE": DISC_LEARNING_RATE,
        "GEN_LEARNING_RATE": GEN_LEARNING_RATE,
        "BATCH_SIZE": BATCH_SIZE,
        "TRAINING_STEPS": TRAINING_STEPS,
        "LOSS_TYPE": LOSS_TYPE,
        "DISC_STEPS_PER_GEN": DISC_STEPS_PER_GEN,
        "DISC_WARMUP_STEPS": DISC_WARMUP_STEPS,
        "LABEL_REAL": LABEL_REAL,
        "LABEL_FAKE": LABEL_FAKE,
        "GEN_INIT_STD": GEN_INIT_STD,
        "DISC_HEAD_INIT_STD": DISC_HEAD_INIT_STD,
        "DEVICE": DEVICE_NAME,
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
        "disc_loss", "disc_loss_real", "disc_loss_fake", "disc_grad_norm",
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
        # Train Discriminator multiple times per Generator step
        disc_losses = []
        disc_grad_norms = []

        batch_real = None
        batch_fake_gen = None
        last_d_metrics = None
        #Mögllichkeit Diskrimantor öfter zu tarinieren pro Generator Step, typischerweise 1:1 oder 2:1 Verhältnis
        for _ in range(int(DISC_STEPS_PER_GEN)):
            
            # REAL: Sample 4 cities, lookup 6 edges from cache
            batch_real = create_batch_real(cities, BATCH_SIZE, cache, rng)

            # FAKE: Generate edges via generator from fresh noise
            noise_batch_disc = rng.uniform(0, 1, size=(BATCH_SIZE, 6))
            batch_fake_gen = gen.batch_forward(noise_batch_disc)

            d_metrics = train_discriminator_step(disc, batch_real, batch_fake_gen)
            last_d_metrics = d_metrics
            disc_losses.append(d_metrics["disc_loss"])
            disc_grad_norms.append(d_metrics["disc_grad_norm"])

        disc_loss_total = float(np.mean(disc_losses))
        disc_grad_norm = float(np.mean(disc_grad_norms))

        # Train Generator once (to fool discriminator), after warmup
        gen_loss_val = float("nan")
        gen_grad_norm = float("nan")
        batch_fake_final = batch_fake_gen
        #Diskriminator Warmup möglich sprich x Steps Disc lernen ohne Generator
        if step > int(DISC_WARMUP_STEPS):
            noise_batch = rng.uniform(0, 1, size=(BATCH_SIZE, 6))
            gen_loss_val, gen_grad_norm, batch_fake_final = train_generator_step(disc, gen, noise_batch)
        
        # Compute scores for logging
        real_scores = np.array(_disc_probs_from_edges(disc, disc.weights, batch_real, disc.head_w, disc.head_b), dtype=float)
        fake_scores_disc = np.array(_disc_probs_from_edges(disc, disc.weights, batch_fake_gen, disc.head_w, disc.head_b), dtype=float)
        fake_scores_gen = np.array(_disc_probs_from_edges(disc, disc.weights, batch_fake_final, disc.head_w, disc.head_b), dtype=float)
        
        # Logging
        if step % 50 == 0 or step == 1:
            print(f"Step {step}:")
            print(f"  Disc Loss: {disc_loss_total:.6f}")
            if last_d_metrics is not None:
                print(f"  Disc Loss Real: {last_d_metrics['disc_loss_real']:.6f}")
                print(f"  Disc Loss Fake: {last_d_metrics['disc_loss_fake']:.6f}")
            if np.isfinite(gen_loss_val):
                print(f"  Gen Loss: {gen_loss_val:.6f}")
            else:
                print(f"  Gen Loss: (warmup)")
            print(f"  Real Scores: mean={real_scores.mean():.4f}, min={real_scores.min():.4f}, max={real_scores.max():.4f}")
            print(f"  Fake Scores (Disc): mean={fake_scores_disc.mean():.4f}, min={fake_scores_disc.min():.4f}, max={fake_scores_disc.max():.4f}")
            print(f"  Fake Scores (Gen): mean={fake_scores_gen.mean():.4f}, min={fake_scores_gen.min():.4f}, max={fake_scores_gen.max():.4f}")
            if np.isfinite(gen_grad_norm):
                print(f"  Disc Grad Norm: {disc_grad_norm:.6f}, Gen Grad Norm: {gen_grad_norm:.6f}\n")
            else:
                print(f"  Disc Grad Norm: {disc_grad_norm:.6f}, Gen Grad Norm: (warmup)\n")
        
        # Save metrics to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                disc_loss_total,
                last_d_metrics["disc_loss_real"] if last_d_metrics is not None else float("nan"),
                last_d_metrics["disc_loss_fake"] if last_d_metrics is not None else float("nan"), 
                disc_grad_norm,
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
