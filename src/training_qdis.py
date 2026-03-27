import os
import time
import json
import random
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pennylane as qml
import pennylane.numpy as pnp
from discriminator import QDiscriminator
from config import (
    N_QUBITS, N_LAYERS, LEARNING_RATE, BATCH_SIZE, TRAINING_STEPS,
    LOSS_TYPE, TARGET_LABEL, RANDOM_SEED, CITIES_PATH, DISTANCE_CACHE_PATH,
    LOGS_DIR, MAX_EDGE_LENGTH_KM, ENABLE_PLOTTING
)


@dataclass(frozen=True)
class City:
    name: str
    country: str
    lat: float
    lon: float


def load_cities(path: str = "cities.csv") -> List[City]:
    """Lädt Städte aus cities.csv (robust mit Encoding-Fallbacks)."""
    read_kwargs = dict(sep=';', decimal=',', dtype=str)
    df = None
    source_path = None

    path_obj = Path(path)
    script_dir = Path(__file__).parent
    candidates = [Path(path)]
    if not path_obj.is_absolute():
        candidates.append(script_dir / path)
        candidates.append(script_dir / 'archive' / path_obj.name)

    seen = set()
    candidates = [p for p in candidates if not (str(p) in seen or seen.add(str(p)))]

    for candidate in candidates:
        if not candidate.exists():
            continue
        for enc in (None, 'utf-8-sig', 'cp1252', 'latin-1'):
            try:
                if enc is None:
                    df = pd.read_csv(candidate, **read_kwargs)
                else:
                    df = pd.read_csv(candidate, encoding=enc, **read_kwargs)
                source_path = candidate
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        if df is not None:
            break

    if df is None:
        raise FileNotFoundError(f"cities.csv nicht gefunden")

    df.columns = [c.strip().lower() for c in df.columns]
    if 'latitude' in df.columns and 'lat' not in df.columns:
        df = df.rename(columns={'latitude': 'lat'})
    if 'longitude' in df.columns and 'lon' not in df.columns:
        df = df.rename(columns={'longitude': 'lon'})

    for col in ('lat', 'lon'):
        df[col] = df[col].astype(str).str.replace(r'\s+', '', regex=True).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    cities: List[City] = []
    for _, r in df.iterrows():
        cities.append(City(str(r["city"]).strip(), str(r["country"]).strip(), float(r["lat"]), float(r["lon"])))

    return cities


def _pair_key(a: City, b: City) -> Tuple[str, str]:
    """Stabiler Cache-Key für Stadtpaar."""
    ka = f"{a.name}|{a.country}"
    kb = f"{b.name}|{b.country}"
    return tuple(sorted((ka, kb)))


def load_distance_cache(cache_path: str = "distance_cache.csv") -> Dict[Tuple[str, str], float]:
    """Lädt Distance-Cache aus CSV."""
    cache: Dict[Tuple[str, str], float] = {}
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache nicht gefunden: {cache_path}")
    with open(cache_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = (row["k1"], row["k2"])
            cache[k] = float(row["distance_km"])
    return cache


def sample_edges_from_cache(
    cities: List[City],
    cache: Dict[Tuple[str, str], float],
    rng: np.random.Generator
) -> Tuple[bool, np.ndarray]:
    """
    Sampelt 4 zufällige Städte und holt 6 Kanten aus dem Cache.
    Returns: (success, edges) wobei success=True wenn alle 6 Kanten im Cache vorhanden sind.
    """
    # FIX: Nutze konsequent rng statt random.sample für bessere Reproduzierbarkeit
    idx = rng.choice(len(cities), size=4, replace=False)
    a, b, c, d = [cities[i] for i in idx]
    
    print(f"    [sample_edges] Sampled Cities: {a.name} → {b.name} → {c.name} → {d.name}")
    
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
        print(f"    [sample_edges] Missing: Kanten {missing} nicht im Cache → verworfen")
        return False, None
    
    edges = np.array([cache[pair] for pair in pairs], dtype=float)
    edge_names = ["e_ab", "e_bc", "e_cd", "e_da", "e_ac", "e_bd"]
    print(f"    [sample_edges] 6 Kanten erfolgreich geladen:")
    for name, val in zip(edge_names, edges):
        print(f"      {name}: {val:.2f} km")
    
    return True, edges


def mse_loss(probs, target):
    return float(np.mean((probs - target) ** 2))


def pce_loss(probs, target, eps=1e-9):
    return float(np.mean((probs - target) ** 2 / (target + eps)))


def log_loss(probs, target, eps=1e-12):
    probs = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(target * np.log(probs) + (1 - target) * np.log(1 - probs)))


def run(
    cities_path: str = CITIES_PATH,
    cache_path: str = DISTANCE_CACHE_PATH,
    seed: int = RANDOM_SEED,
    steps: int = TRAINING_STEPS,
    batch_size: int = BATCH_SIZE,
    n_layers: int = N_LAYERS,
    lr: float = LEARNING_RATE,
    target_label: float = TARGET_LABEL,
    loss_type: str = LOSS_TYPE,
):
    """
    Trainiert den QDiscriminator auf echten Kanten aus cities.csv/distance_cache.csv.
    - Sampelt on-the-fly 4 zufällige Städte
    - Holt 6 Distanzen aus dem Cache
    - Trainiert mit Label=1 (real)
    """
    print("\n" + "="*80)
    print("QUANTUM DISCRIMINATOR TRAINING")
    print("="*80 + "\n")
    
    disc = QDiscriminator(n_layer=n_layers, seed=seed)
    
    # Visualisiere den Circuit
    print("[Circuit Struktur]")
    try:
        # Zeige den abstrakten Circuit
        circuit_info = f"""
        Quantum Circuit mit {disc.n_qubits} Qubits und {n_layers} Entangling-Lagen:
        
        Eingabe (6 Kanten):
          Q0 ← e_ab  |  Q1 ← e_bc  |  Q2 ← e_cd
          Q3 ← e_da  |  Q4 ← e_ac  |  Q5 ← e_bd
        
        Layer 0-{n_layers-1}:
          - AngleEmbedding: Y-Rotation auf allen 6 Qubits
          - StronglyEntanglingLayers: RX, RY, RZ + CX-Verschränkung
        
        Messung:
          Z-Erwartungswert(Q0) → [-1,1] → normalisieren zu [0,1]
        
        Trainierbare Parameter: {disc.n_layer} Lagen × {disc.n_qubits} Qubits × 3 (RX,RY,RZ) = {disc.n_layer * disc.n_qubits * 3} Parameter
        """
        print(circuit_info)
    except Exception as e:
        print(f"  Fehler: {e}")
    print()

    run_dir = os.path.join(LOGS_DIR, f"qdis_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(
            {
                "cities_path": cities_path,
                "cache_path": cache_path,
                "seed": seed,
                "steps": steps,
                "batch_size": batch_size,
                "n_layers": n_layers,
                "lr": lr,
                "target_label": target_label,
                "loss_type": loss_type,
            },
            f,
            indent=2,
        )

    target_p = float(target_label)

    # Lade Städte und Cache
    print(f"Lade Städte aus {cities_path}...")
    cities = load_cities(cities_path)
    print(f"✓ {len(cities)} Städte geladen\n")
    
    print(f"Lade Cache aus {cache_path}...")
    cache = load_distance_cache(cache_path)
    print(f"✓ {len(cache)} Distanzen im Cache\n")

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"Starte Training mit folgenden Hyperparametern:")
    print(f"  - Discriminator: {n_layers} Lagen, {disc.n_qubits} Qubits")
    print(f"  - Learning Rate: {lr}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Loss Type: {loss_type}")
    print(f"  - Target Label (für echte Daten): {target_label}\n")

    def _forward_normalized(edges_batch_raw, weights):
        """
        Hilfs-Funktion: Berechnet Diskriminator-Ausgaben auf normalisierten Inputs.
        WICHTIG: Gleicher Pfad wie loss_inner() für konsistentes Training & Logging!
        """
        edges_batch = pnp.array(edges_batch_raw, dtype=float)
        edges_batch = edges_batch / MAX_EDGE_LENGTH_KM
        edges_batch = pnp.clip(edges_batch, 0.0, 1.0)
        
        preds = []
        for e in edges_batch:
            z = disc.circuit(e, weights)
            p = 0.5 * (z + 1.0)
            preds.append(p)
        return pnp.stack(preds)

    def loss_inner(weights, edges_batch):
        # FIX BUG #1: Normalisiere edges GENAU WIE in forward()
        # Das ist KRITISCH - sonst trainieren wir auf anderer Verteilung als wir evaluieren!
        preds = _forward_normalized(edges_batch, weights)
        
        if loss_type == "pce":
            return pnp.mean((preds - target_p) ** 2 / (target_p + 1e-9))
        elif loss_type == "log":
            preds_clipped = pnp.clip(preds, 1e-12, 1.0 - 1e-12)
            return -pnp.mean(target_p * pnp.log(preds_clipped) + (1 - target_p) * pnp.log(1 - preds_clipped))
        else:
            return pnp.mean((preds - target_p) ** 2)

    grad_w = qml.grad(lambda w, eb: loss_inner(w, eb))

    metrics_path = os.path.join(run_dir, "metrics.csv")
    with open(metrics_path, "w", encoding="utf-8") as f:
        # Header: step, loss, score_mean, score_std, score_min, score_max, 
        #         grad_norm, edge_min, edge_max, dann 36 Gewichte
        # Gewichte Shape: (n_layer, n_qubits, 3) für RX, RY, RZ pro Qubit pro Layer
        header = ["step", "loss", "score_mean", "score_std", "score_min", "score_max", 
                  "grad_norm", "edge_norm_min", "edge_norm_max"]
        for layer in range(n_layers):
            for qubit in range(6):  # 6 Qubits
                for gate_idx, gate_name in enumerate(["RX", "RY", "RZ"]):
                    header.append(f"w_l{layer}_q{qubit}_{gate_name}")
        f.write(",".join(header) + "\n")

    for step in range(1, steps + 1):
        # Sampelt Batch von 4er-Stadtgruppen
        edges_batch = []
        attempts = 0
        max_attempts = max(1000, batch_size * 100)
        
        print(f"[Step {step}/{steps}] Generiere Batch ({batch_size} Samples):")
        
        while len(edges_batch) < batch_size and attempts < max_attempts:
            attempts += 1
            success, edges = sample_edges_from_cache(cities, cache, rng)
            if success:
                edges_batch.append(edges)
        
        if len(edges_batch) == 0:
            print(f"[WARN] step {step}: Keine gültigen Samples gefunden - überspringe Step\n")
            continue
        
        if len(edges_batch) < batch_size:
            print(f"[WARN] step {step}: nur {len(edges_batch)}/{batch_size} gültige Samples\n")
        
        edges_batch = pnp.array(edges_batch, dtype=float)

        # Vorwärts-Score (nutze den gleichen normalisierten Pfad wie loss_inner!)
        print(f"  [forward pass] Klassifiziere {len(edges_batch)} Samples:")
        
        # WICHTIG: Berechne Scores mit der gleichen Normalisierung wie im Loss
        scores_tensor = _forward_normalized(edges_batch, pnp.array(disc.weights))
        scores = np.array(scores_tensor, dtype=float)
        
        # Debug-Logging
        edge_names = ["e_ab", "e_bc", "e_cd", "e_da", "e_ac", "e_bd"]
        for idx, (e_raw, s) in enumerate(zip(edges_batch, scores)):
            print(f"    Sample {idx+1}/{len(edges_batch)}:")
            e_norm = e_raw / MAX_EDGE_LENGTH_KM
            e_norm = np.clip(e_norm, 0.0, 1.0)
            print(f"  [forward] Input Kanten (werden auf Q0..Q5 abgebildet):")
            for i, (name, val_raw, val_norm) in enumerate(zip(edge_names, e_raw, e_norm)):
                print(f"    Q{i} <- {name}: {val_raw:.2f} km -> normalized: {val_norm:.4f}")
            print(f"  [forward] Output: P(real)={s:.4f}\n")

        # Verlust für Logging (aus den gleichen normalisierten Scores berechnet!)
        if loss_type == "pce":
            loss_val = pce_loss(scores, target_p)
        elif loss_type == "log":
            loss_val = log_loss(scores, target_p)
        else:
            loss_val = mse_loss(scores, target_p)

        # Gradienten-Update
        print(f"  [gradient update] Compute gradients für {n_layers} Lagen × {disc.n_qubits} Qubits")
        # FIX BUG #4: qml.grad() gibt ein Tuple zurück: (gradient_array, predictions)
        g_result = grad_w(pnp.array(disc.weights), edges_batch)
        
        # Extrahiere nur den Gradienten aus dem Tuple
        if isinstance(g_result, tuple):
            g_w = g_result[0]
        else:
            g_w = g_result
        
        g_np = np.asarray(g_w, dtype=float)
        
        # Berechne Gradient Norm
        grad_norm = float(np.linalg.norm(g_np))
        print(f"    Gradient Norm: {grad_norm:.6f}")
        
        disc.weights = disc.weights - lr * g_np  # Standard: Gradienten-Abstieg
        print(f"  [gradient update] Gewichte aktualisiert (lr={lr})\n")

        # Berechne erweiterte Score-Statistiken
        score_mean = float(scores.mean())
        score_std = float(scores.std())
        score_min = float(scores.min())
        score_max = float(scores.max())
        
        # Berechne Statistiken der normalisierten edges
        edges_normalized = edges_batch / MAX_EDGE_LENGTH_KM
        edges_normalized = np.clip(edges_normalized, 0.0, 1.0)
        edge_norm_min = float(np.min(edges_normalized))
        edge_norm_max = float(np.max(edges_normalized))
        
        # Berechne Weight Statistiken (für Console-Output)
        weight_mean = float(np.mean(disc.weights))
        weight_std = float(np.std(disc.weights))
        
        # Berechne Entropy (Shannon Entropy der normalisierten Gewichte)
        weights_abs = np.abs(disc.weights).flatten()
        weights_norm = weights_abs / (np.sum(weights_abs) + 1e-10)
        entropy = float(-np.sum(weights_norm * np.log(weights_norm + 1e-10)))
        
        # Debug-Ausgabe (nur alle 100 Steps, um Spam zu sparen)
        if step % 100 == 0 or step <= 10:
            print(f"  [METRIC CHECKPOINT Step {step}]")
            print(f"    Score:  mean={score_mean:.4f} | std={score_std:.4f} | min={score_min:.4f} | max={score_max:.4f}")
            print(f"    Gradient Norm: {grad_norm:.6f}")
            print(f"    Normalized Edges: min={edge_norm_min:.4f} | max={edge_norm_max:.4f}")
            print(f"    Loss: {loss_val:.6f}")
            print(f"    Weights: mean={weight_mean:.6f} | std={weight_std:.6f} | entropy={entropy:.4f}\n")
        
        print(f"  [weights] Current Gewichte Shape: {disc.weights.shape}")
        print(f"    Gesamt: {n_layers * 6 * 3} Parameter ({n_layers} Lagen × 6 Qubits × 3 Gates)")
        print(f"    Gewichte (pro Layer, pro Qubit: RX, RY, RZ):")
        qubit_names = ["e_ab", "e_bc", "e_cd", "e_da", "e_ac", "e_bd"]
        for layer in range(n_layers):
            print(f"      Layer {layer}:")
            for qubit in range(6):
                rx_val = disc.weights[layer, qubit, 0]
                ry_val = disc.weights[layer, qubit, 1]
                rz_val = disc.weights[layer, qubit, 2]
                print(f"        Q{qubit} ({qubit_names[qubit]}): RX={rx_val:7.4f} | RY={ry_val:7.4f} | RZ={rz_val:7.4f}")
        print()

        # Schreibe Metrics: step, loss, score_mean, score_std, score_min, score_max, grad_norm, edge_min, edge_max, dann alle 36 Gewichte
        with open(metrics_path, "a", encoding="utf-8") as f:
            row_data = [str(step), str(loss_val), str(score_mean), str(score_std), 
                       str(score_min), str(score_max), str(grad_norm), str(edge_norm_min), str(edge_norm_max)]
            # Füge alle 36 Gewichte ein (flatt aus (n_layer, n_qubits, 3) Shape)
            for w in disc.weights.flatten():
                row_data.append(str(w))
            f.write(",".join(row_data) + "\n")

        if step % 20 == 0 or step == 1:
            print(f"[SUMMARY Step {step}] loss={loss_val:.6f} | score_mean={score_mean:.4f} | score_min={score_min:.4f} | score_max={score_max:.4f} | grad_norm={grad_norm:.6f}\n")

    print("Fertig. Logs unter:", run_dir)
    return run_dir


if __name__ == "__main__":
    run(
        cities_path=CITIES_PATH,
        cache_path=DISTANCE_CACHE_PATH,
        seed=RANDOM_SEED,
        steps=TRAINING_STEPS,
        batch_size=BATCH_SIZE,
        n_layers=N_LAYERS,
        lr=LEARNING_RATE,
        target_label=TARGET_LABEL,
        loss_type=LOSS_TYPE,
    )
