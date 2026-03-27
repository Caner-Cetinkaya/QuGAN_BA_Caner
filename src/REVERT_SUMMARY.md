# Circuit Revert Summary (Feb 2, 2026 - Evening)

## Reason for Revert
Nach mehreren Experiments und Tests war klar, dass der Circuit zurück auf die original **Explizite Layer-Struktur** (RX/RY/RZ Loop) gebracht werden sollte:
- **XYZ-Triple Embedding** (experimentiert): Performte schlechter (Score 0.504→0.490)
- **Flattened Weights (24 params)**: Verwirrte die Struktur
- **Original (36 params)**: Simpel, transparent, und konzeptionell klarer

## Was wurde reverted

### discriminator.py
```
BEFORE (XYZ-Triple):
- Gewichte Shape: (24,) - flattened
- Embedding: XYZ-Triple (2×6 RX/RY nur)
- Keine RZ-Rotationen

AFTER (Explicit RX/RY/RZ):
- Gewichte Shape: (2, 6, 3) = 36 Parameter
- Embedding: AngleEmbedding Y-Rotation (einmalig)
- Layer Loop: RX → RY → RZ → CNOT (pro Layer)
```

Circuit Architektur (aktuell - FINAL):
```
1. EMBEDDING (einmalig):
   - AngleEmbedding mit Y-Rotationen
   - Kodiert 6 Kantenlängen auf 6 Qubits

2. VQC LAYER LOOP (2 Lagen):
   - RX-Rotationen (trainierbar)
   - RY-Rotationen (trainierbar)
   - RZ-Rotationen (trainierbar)
   - CNOT-Verschränkung (zirkulär)

3. MESSUNG:
   - Z-Erwartungswert auf Qubit 0
   - Output ∈ [-1, 1] → normalisiert zu P(real) ∈ [0, 1]
```

### training_qdis.py
```
BEFORE (24 params):
- Header: 4 + 24 Gewichte-Spalten
- Weight indexing: idx-based flattening

AFTER (36 params):
- Header: 4 + 36 Gewichte-Spalten
- Weight indexing: (layer, qubit, gate) tuple access
- Flattening: .flatten() für CSV-Output
```

## Validierung

✅ **discriminator.py**: Keine Syntax-Fehler
✅ **training_qdis.py**: Keine Syntax-Fehler
✅ **Forward Pass Test**: P(real)=0.5432 ∈ [0,1] ✓
✅ **Weight Shape Test**: (2, 6, 3) = 36 Parameter ✓

## Nächste Schritte

1. **Optional**: Kurzer Test mit 50 Steps um sicherzustellen dass Training jetzt wieder funktioniert
2. **TODO**: Supervisor-Konsultation notwendig für korrekte Ansatz-Design
   - Circuit gibt immer noch P(real)≈0.5 aus
   - Z-Werte Bereich zu klein (-0.13 bis +0.27)
   - Keine Diskriminationskraft erkennbar
3. **Blocker**: Nur positive Beispiele (target=1.0) verfügbar → benötige Generator für Fake-Samples

## Known Issues (ungelöst)

- ⚠️ P(real) ≈ 0.5 unabhängig vom Input (keine Lernfähigkeit)
- ⚠️ Z-Messungs-Bereich zu eng (nur 0.4 Differenz statt potentiell 2.0)
- ⚠️ Keine negativen Trainings-Beispiele (alle target=1.0)
- ⚠️ Circuit responsiveness fragwürdig

**Hypothesis**: Circuit-Design ist fundamentales Problem, nicht Optimierungs-Problem.
