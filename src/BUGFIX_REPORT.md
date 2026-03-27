# Bug-Fix Report – QuGAN Code Review
**Datum:** 25. März 2026  
**Geprüfte Dateien:** `generator.py`, `discriminator.py`, `training_qgan.py`  
**Testdatei:** `test_qgan_bugs.py` (66 Tests, alle bestanden)

---

## Übersicht der gefundenen und behobenen Bugs

| # | Datei | Schwere | Status |
|---|---|---|---|
| BUG-1 | `discriminator.py` | Mittel | ✅ Behoben |
| BUG-2 | `generator.py` | Mittel | ✅ Behoben |
| BUG-3 | `training_qgan.py` | Gering | ✅ Behoben |
| BUG-4 | `training_qgan.py` | Info | Dokumentiert |
| BUG-5 | `training_qgan.py` | Info | Dokumentiert |

---

## BUG-1 – `discriminator.py`: `_check_triangle_inequality` lehnte Batch-Input ab

### Problem
Die Methode `_check_triangle_inequality` hatte laut Docstring folgende Signatur:
```
Args:
    edges: Array shape (4,) oder (n, 4)
Returns:
    True wenn Bedingung erfüllt, False sonst
```
Der tatsächliche Code warf jedoch bei einem `(n, 4)`-Array einen `ValueError`:
```python
if edges.shape != (4,):
    raise ValueError(f"edges muss shape (4,) haben, bekam {edges.shape}")
```
Das heißt: Docstring und Implementierung stimmten nicht überein. Batch-Auswertung
mehrerer Kanten auf einmal war damit unmöglich.

### Ursache
Die `(n, 4)`-Behandlung wurde im Docstring versprochen, aber nie implementiert.

### Fix
```python
# Vorher
if edges.shape != (4,):
    raise ValueError(f"edges muss shape (4,) haben, bekam {edges.shape}")
sorted_edges = np.sort(edges)
return bool(np.sum(sorted_edges[:3]) >= sorted_edges[3])

# Nachher
# BUG-1 FIX: (n, 4) wird jetzt korrekt als Batch behandelt
if edges.ndim == 2 and edges.shape[1] == 4:
    sorted_edges = np.sort(edges, axis=1)  # (n, 4)
    return (np.sum(sorted_edges[:, :3], axis=1) >= sorted_edges[:, 3])
if edges.shape != (4,):
    raise ValueError(f"edges muss shape (4,) oder (n,4) haben, bekam {edges.shape}")
sorted_edges = np.sort(edges)
return bool(np.sum(sorted_edges[:3]) >= sorted_edges[3])
```

### Betroffene Datei und Zeile
- `discriminator.py`, Methode `_check_triangle_inequality`

---

## BUG-2 – `generator.py`: `batch_forward` lieferte falsche Shape bei 1D-Input

### Problem
Der Docstring von `batch_forward` garantiert:
```
Returns:
    Array mit Shape (N, 6) Kantenlaengen
```
Bei einem 1D-Input mit Shape `(6,)` (ein einzelner Noise-Vektor) gab die Funktion
jedoch Shape `(6,)` zurück statt `(1, 6)`. Der interne Code behandelte diesen Fall
mit einem frühzeitigen `return` ohne den Output in die garantierte Batch-Form zu bringen:

```python
# Single sample (alter Code)
if noise_batch.ndim == 1:
    noise_normalized = pnp.clip(noise_batch, 0.0, 1.0)
    z_values = self.circuit(noise_normalized, weights)
    z_values = pnp.array(z_values, dtype=float)
    edges_normalized = self._to_edge_length(z_values)
    return edges_normalized  # Shape (6,) – verletzt Docstring-Garantie!
```

### Ursache
Der Sonderfall für 1D-Input wurde separat behandelt und gibt direkt zurück,
ohne den Output zum versprochenen `(N, 6)`-Format zu expandieren.

### Fix
Der 1D-Input wird nun vor der Verarbeitung zu `(1, 6)` aufgespannt. Der allgemeine
Batch-Pfad übernimmt dann die Verarbeitung und liefert korrekt `(1, 6)`:

```python
# BUG-2 FIX: 1D-Input (6,) zu (1,6) umformen, damit Output-Shape (1,6) ist
squeezed = False
if noise_batch.ndim == 1:
    noise_batch = noise_batch[pnp.newaxis, :]  # (1, 6)
    squeezed = True
# … danach läuft der normale Batch-Pfad durch und liefert (1, 6)
```

### Betroffene Datei und Zeile
- `generator.py`, Methode `batch_forward`

---

## BUG-3 – `training_qgan.py`: Ungenutzter `rng`-Parameter in `train_discriminator_step`

### Problem
Die Funktion `train_discriminator_step` akzeptierte einen `rng`-Parameter:
```python
def train_discriminator_step(disc, batch_real, batch_fake, rng):
```
Dieser Parameter wurde jedoch **nirgends im Funktionsrumpf verwendet**. Das führte zu:
- Irreführender API (Caller denkt, der RNG beeinflusst den Schritt)
- Unnötigem Parameter, der an allen Aufrufstellen mitgegeben werden musste
- Potenziellem Missverständnis über Reproduzierbarkeit

### Ursache
Der Parameter war vermutlich ein Überbleibsel einer früheren Version, in der
innerhalb des Schritts noch ein Shuffle/Sampling mit dem `rng` stattfand.
Nach dem Refactoring (getrennte real/fake Batches) wurde er nicht entfernt.

### Fix
Parameter aus der Signatur entfernt:
```python
# Vorher
def train_discriminator_step(disc, batch_real, batch_fake, rng):

# Nachher
def train_discriminator_step(disc, batch_real, batch_fake):
```
Alle Aufrufstellen wurden entsprechend angepasst:
- `training_qgan.py` → `main()`-Loop: `train_discriminator_step(disc, batch_real, batch_fake_gen)`
- `sanity_check_qgan.py` → `train_discriminator_step(disc, batch_real, batch_fake)`

### Betroffene Dateien und Zeilen
- `training_qgan.py`, Funktionssignatur `train_discriminator_step` + Aufruf in `main()`
- `sanity_check_qgan.py`, Aufruf in `main()`

---

## BUG-4 – `training_qgan.py`: Toter Code in `_disc_probs_from_edges` (Info)

### Problem
In `_disc_probs_from_edges` ist ein größerer Block auskommentiert, der das Broadcasting
über mehrere Qubits implementieren sollte. Der aktive Code darunter macht etwas anderes
(nur Qubit 0 lesen). Dies ist kein direkter Fehler, aber der auskommentierte Code ist
irreführend und sollte bei nächster Gelegenheit bereinigt werden.

```python
# Auskommentierter Block – nie aktiv:
"""
z = disc.circuit(edges_batch, weights)
if isinstance(z, (list, tuple)):
    z = pnp.stack(z, axis=-1)
z_mean = pnp.mean(z, axis=-1)
"""
# Aktiver Code:
z = disc.circuit(edges_batch, weights)
return 0.5 * (z + 1.0)
```

### Empfehlung
Den auskommentierten Block vollständig entfernen, wenn die Entscheidung
(nur Qubit 0 vs. alle Qubits mitteln) endgültig getroffen ist.

**Status:** Nicht behoben – nur dokumentiert (kein aktiver Bug, nur Code-Hygiene).

---

## BUG-5 – `training_qgan.py`: Modul-Level `rng` als globaler Zustand (Info)

### Problem
Auf Modul-Ebene wird direkt beim Import ein globaler RNG instanziiert:
```python
rng = np.random.default_rng(SEED)
```
Dieser globale RNG wird im `main()`-Loop genutzt (`batch_real`, `noise_batch` etc.).
Das hat folgende Konsequenzen:
- Beim Import des Moduls (z.B. in Tests) wird der Seed sofort verbraucht
- Jeder Aufruf von Utility-Funktionen, die intern sampling machen, verschiebt
  den globalen RNG-State unkontrolliert
- Reproduzierbarkeit hängt davon ab, dass niemand den Modul-RNG zwischen den
  Trainingsschritten verändert

### Empfehlung
Den RNG lokal in `main()` instanziieren statt auf Modul-Ebene:
```python
# Empfehlung:
def main():
    rng = np.random.default_rng(SEED)
    ...
```

**Status:** Nicht behoben – nur dokumentiert (Refactoring würde alle Funktionsaufrufe
betreffen; kein akuter Fehler im Normalbetrieb).

---

## Zusammenfassung der Änderungen

### Geänderte Dateien

| Datei | Art der Änderung |
|---|---|
| `discriminator.py` | Batch-Input `(n,4)` in `_check_triangle_inequality` unterstützt (BUG-1) |
| `generator.py` | `batch_forward` gibt korrekt `(1,6)` bei 1D-Input zurück (BUG-2) |
| `training_qgan.py` | `rng`-Parameter aus `train_discriminator_step` entfernt (BUG-3) |
| `sanity_check_qgan.py` | Aufruf von `train_discriminator_step` ohne `rng` angepasst (BUG-3 Folgeänderung) |

### Neue Datei

| Datei | Inhalt |
|---|---|
| `test_qgan_bugs.py` | 66 eigenständige pytest-Tests für alle drei Quelldateien |

---

## Testergebnisse

```
============================= test session starts =============================
collected 66 items

test_qgan_bugs.py  ..................................................................
..............                                                    [100%]

============================= 66 passed in 6.00s ==============================
```
