"""
REPRODUZIERBARKEIT & DETERMINISMUS

Dieses Dokument erklärt, wie die QuGAN Basis-Komponenten
für reproduzierbare Ergebnisse konfiguriert sind.

========================================================================================================
1. SEED-HANDLING
========================================================================================================

Generator (QGen):
─────────────────
- Seed wird bei Initialisierung übergeben: QGen(n_layer=2, seed=42)
- Der Seed steuert:
  1. Numpy RNG für Gewichte-Initialisierung (self.rng)
  2. Permanente Gewichte (self.weigths) sind damit deterministisch

In training_qgen.py:
  ├─ np.random.seed(seed)        # Globaler Numpy-Seed
  ├─ gen = QGen(..., seed=seed)  # Generator mit Seed
  └─ Trainingsloop ist dann deterministisch für gegebene Parameter


Discriminator (QDiscriminator):
───────────────────────────────
- Seed bei Initialisierung: QDiscriminator(n_layer=2, seed=0)
- Der Seed steuert:
  1. Numpy RNG für Gewichte-Initialisierung
  2. Permanente Gewichte sind damit deterministisch

In training_qdis.py:
  ├─ np.random.seed(seed)           # Globaler Numpy-Seed
  ├─ disc = QDiscriminator(..., seed=seed)
  └─ Deterministische Datensatz-Sampling via:
     seed_per_sample = seed + step * 1000 + len(edges_batch)
     ds.sample_four_edges(seed=seed_per_sample)


TSPDataset Sampling:
────────────────────
- sample_four_edges(seed=None) nutzt np.random.default_rng(seed)
- Mit gleichen Seed → identische Punkte & Kanten
- Jeder Step erhält einen eindeutigen Seed:
  seed_per_sample = base_seed + step_offset + batch_offset

========================================================================================================
2. PENNYLANE AUTOGRAD vs TORCH
========================================================================================================

GEWÄHLT: pennylane.numpy (pnp) + qml.grad()
───────────────────────────────────────────

Warum NICHT torch:
  ❌ PyTorch besitzt in diesem Fall Overkill (einfache Gradient-Berechnung)
  ❌ Quantum Circuits sind besser mit qml.grad() kompatibel (automatische Differenziation)
  ❌ torch.device würde komplexere Implementierung erfordern

Warum qml.grad():
  ✓ Native Integration mit PennyLane
  ✓ Funktioniert mit @qml.qnode (Quantum Circuits)
  ✓ Automatische Differenziation durch Autograd (stable numerisch)
  ✓ Keine zusätzlichen Dependencies
  ✓ Performance für kleine bis mittlere Netzwerke adäquat

DOKUMENTATION:
  - Interface in QNode wird explizit gesetzt: interface="autograd"
  - In training_*.py wird qml.grad(loss_inner) verwendet
  - Gradienten werden zu NumPy konvertiert bevor SGD-Update: np.asarray(g, dtype=float)


Torch Imports (aktuell ungenutzt):
──────────────────────────────────
In discriminator.py:
  import torch
  import torch.nn
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Status: Vorhanden aber ungenutzt (für zukünftige GPU-Beschleunigung)
Fix: Könnten entfernt werden, falls nicht geplant

========================================================================================================
3. REPRODUZIERBARKEIT CHECKLIST
========================================================================================================

Reproduzierbarkeitsfaktoren:
  ✓ Seed wird in allen Run-Funktionen akzeptiert (default=42)
  ✓ Numpy global seed gesetzt: np.random.seed(seed)
  ✓ QGen/QDiscriminator erhalten explizite Seeds
  ✓ TSPDataset Sampling verwendet deterministischen Seed pro Step
  ✓ Keine globalen Zufallszahlengeneratoren ohne Seed
  ✓ Debug-Prints entfernt (keine nicht-deterministischen Ausgaben)
  ✓ Gradient-Berechnung ist deterministic (qml.grad)
  ✓ SGD-Update ist deterministisch (weigths -= lr * gradient)

Config wird gespeichert:
  - config.json speichert alle Hyperparameter
  - Inklusive z0, target, seed, temperature, etc.
  - Reproduzierung möglich durch config.json Laden + gleiche Seed

========================================================================================================
4. TESTING: REPRODUZIERBARKEIT VERIFIZIEREN
========================================================================================================

Test 1: Gleichzeitige Runs mit gleichen Seeds
──────────────────────────────────────────────

from training_qgen import run
import numpy as np

# Run 1
metrics1 = run(loss_type="pce", seed=42, steps=10)
df1 = pd.read_csv(os.path.join(metrics1, "metrics.csv"))

# Run 2 (gleiches Setup)
metrics2 = run(loss_type="pce", seed=42, steps=10)
df2 = pd.read_csv(os.path.join(metrics2, "metrics.csv"))

# Überprüfung
assert np.allclose(df1['loss'].values, df2['loss'].values, atol=1e-10), "Loss unterschiedlich!"
print("✓ Reproduzierbar: Beide Runs haben identische Loss-Kurve")


Test 2: Unterschiedliche Seeds → unterschiedliche Ergebnisse
─────────────────────────────────────────────────────────────

from training_qgen import run
import pandas as pd

# Run mit seed=42
metrics_42 = run(loss_type="pce", seed=42, steps=100)
df_42 = pd.read_csv(os.path.join(metrics_42, "metrics.csv"))

# Run mit seed=43
metrics_43 = run(loss_type="pce", seed=43, steps=100)
df_43 = pd.read_csv(os.path.join(metrics_43, "metrics.csv"))

# Check: Sollten unterschiedlich sein (mit hoher Wahrscheinlichkeit)
are_different = not np.allclose(df_42['loss'].values[:10], df_43['loss'].values[:10], atol=1e-10)
assert are_different, "Unterschiedliche Seeds sollten unterschiedliche Ergebnisse geben!"
print("✓ Deterministisch: Verschiedene Seeds → verschiedene Kurven")

========================================================================================================
5. VERSION-INFORMATION FÜR REPRODUZIERBARKEIT
========================================================================================================

Wichtige Dependencies (sollten dokumentiert sein):
  ├─ numpy: >= 1.20 (für np.random.default_rng)
  ├─ pennylane: >= 0.30 (für stabile autograd)
  ├─ pandas: >= 1.0 (für CSV-Handling)
  ├─ PyTorch: >= 1.8 (optional, derzeit ungenutzt)
  └─ Python: >= 3.10 (für type hints: int | None)

Falls anders benötigt:
  pip show numpy pennylane pandas torch
  # → Versionsnummern notieren


========================================================================================================
6. POTENTIAL ISSUES & NOTES
========================================================================================================

⚠️  Pennylane-Version Kompatibilität:
    - Neuere qml-Versionen könnten autograd-Interface ändern
    - Test: Kleine Run mit steps=5 um schnell zu verifizieren

⚠️  Numerische Stabilität:
    - Softmax benutzt numerisch stabilen Trick (subtract max)
    - Aber bei extremen Learning Rates könnte NaN auftreten
    - Default lr=0.05 ist konservativ

⚠️  RNG Unterschiede zwischen Plattformen:
    - np.random.default_rng sollte plattformunabhängig sein
    - Aber getestet wurde hauptsächlich auf Windows
    - Linux/Mac: bitte verifizieren mit Test 1 oben

⚠️  Torch Device (ungenutzt):
    - device = torch.device(...) wird nicht verwendet
    - Kann entfernt oder für GPU-Unterstützung aktiviert werden

========================================================================================================
7. BEST PRACTICES
========================================================================================================

Für reproduzierbare Runs verwende:
  
  # In Python/Jupyter
  from training_qgen import run
  
  # Mit explizitem Seed
  run_dir = run(
      loss_type="pce",
      seed=42,  # ← WICHTIG: Immer Seed setzen
      steps=300,
      n_layers=2,
      lr=0.05,
      temperature=1.0
  )
  
  # Konfiguration wird gespeichert in:
  # run_dir/config.json
  # → Nachträglich nachvollziehbar!


Für Batch-Experimente (multiple Seeds):
  
  seeds = [0, 1, 2, 3, 4]
  run_dirs = []
  
  for s in seeds:
      rd = run(loss_type="pce", seed=s, steps=300)
      run_dirs.append(rd)
      # Jeder Run hat identische initialisierung & Parameter
      # nur unterschiedliche Gewichte durch unterschiedliche Seeds
  
  # Vergleiche dann die Loss-Kurven in den metrics.csv

========================================================================================================
FAZIT
========================================================================================================

Dein System ist JETZT reproduzierbar:

✓ Seeds werden konsistent gehandhabt
✓ Keine nicht-deterministischen Operationen im kritischen Pfad
✓ Konfigurationen werden gespeichert
✓ qml.grad für stabile Autograd-Berechnung
✓ Debug-Outputs entfernt

Das bedeutet: Mit `seed=42` erhälst du IMMER die gleiche Loss-Kurve,
wenn die Hardware/Software-Umgebung identisch ist.

"""

print(__doc__)
