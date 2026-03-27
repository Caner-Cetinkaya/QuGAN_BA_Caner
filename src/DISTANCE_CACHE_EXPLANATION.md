# ⚡ Distance Cache Explanation (WICHTIG!)

**Dieses Dokument korrigiert einen wichtigen Fehler in den anderen Docs!**

---

## 🚨 KORREKTUR: Real Edge Sampling verwendet distance_cache.csv!

### Was ich FALSCH dokumentiert habe:
❌ "Haversine-Berechnung On-The-Fly zwischen 4 Städten"  
❌ "compute_edge_lengths() wird aufgerufen"

### Was WIRKLICH passiert:
✅ **distance_cache.csv wird geladen** (Pre-computed Haversine-Distanzen)  
✅ **`sample_edges_from_cache()` wird aufgerufen**  
✅ **Alle C(80,2)=3160 paarweisen Distanzen sind bereits berechnet**

---

## 📁 Distance Cache Struktur

### `distance_cache.csv`
- **Erstellt mit:** `build_distance_dataset.py`
- **Format:** `k1,k2,distance_km` (City-Paar → Haversine in km)
- **Größe:** 3160 Zeilen (C(80,2) = 80×79/2)
- **Beispiel:**
  ```
  k1,k2,distance_km
  Paris,Berlin,877.23
  Berlin,Madrid,1823.89
  Madrid,Rom,1786.45
  Rom,Paris,1435.67
  Paris,Madrid,1265.32
  Berlin,Rom,1534.78
  ...
  ```

### Loading Process

```python
# In training_qdis.py und training_qgan.py:

cache = load_distance_cache("distance_cache.csv")
# Returns: Dict[Tuple[str, str], float]
# Example: {("Paris", "Berlin"): 877.23, ...}

# Dann in create_batch_real():
success, edges = sample_edges_from_cache(cities, cache, rng)
# Returns: (bool, np.array) 
#   success=True if all 6 edges found in cache
#   edges=(6,) with distances in km
```

---

## 🔄 Logik: `sample_edges_from_cache()`

```python
def sample_edges_from_cache(cities, cache, rng):
    """
    Sample 4 zufällige Städte.
    Hole 6 Kanten aus PRE-COMPUTED distance_cache.
    """
    
    # Step 1: Sample 4 cities
    idx = rng.choice(len(cities), size=4, replace=False)
    a, b, c, d = [cities[i] for i in idx]
    
    # Step 2: Definiere die 6 Paare (zyklisch für TSP)
    pairs = [
        _pair_key(a, b),  # e_ab
        _pair_key(b, c),  # e_bc
        _pair_key(c, d),  # e_cd
        _pair_key(d, a),  # e_da
        _pair_key(a, c),  # e_ac (diagonal)
        _pair_key(b, d),  # e_bd (diagonal)
    ]
    
    # Step 3: Lookup in cache
    edges = []
    for pair in pairs:
        if pair not in cache:
            return (False, None)  # Missing edge!
        edges.append(cache[pair])
    
    # Step 4: Return as numpy array (km)
    return (True, np.array(edges))
```

### Was ist `_pair_key()`?

```python
def _pair_key(city1, city2):
    """
    Erstellt einen konsistenten Schlüssel für zwei Städte.
    Wichtig: Ordnung muss gleich sein (Paris,Berlin) vs (Berlin,Paris)
    """
    # Alphabetisch sortiert für Konsistenz:
    names = tuple(sorted([city1.name, city2.name]))
    return names
    
# Beispiel:
_pair_key(Paris, Berlin)  → ("Berlin", "Paris")
_pair_key(Berlin, Paris)  → ("Berlin", "Paris")  [same!]
```

---

## ✅ Die echte Real Edge Sampling Logik (Step 100 Beispiel)

```
PHASE 1: Create Batch Real

FOR sample in 1..16:
    │
    ├─ Sample 4 cities: [5, 23, 47, 61] → [Paris, Berlin, Madrid, Rom]
    │
    ├─ Create pair keys:
    │  ├─ _pair_key(Paris, Berlin) → ("Berlin", "Paris")
    │  ├─ _pair_key(Berlin, Madrid) → ("Berlin", "Madrid")
    │  ├─ _pair_key(Madrid, Rom) → ("Madrid", "Rom")
    │  ├─ _pair_key(Rom, Paris) → ("Paris", "Rom")
    │  ├─ _pair_key(Paris, Madrid) → ("Madrid", "Paris")
    │  └─ _pair_key(Berlin, Rom) → ("Berlin", "Rom")
    │
    ├─ Lookup in cache:
    │  ├─ cache[("Berlin", "Paris")] → 877.23 km
    │  ├─ cache[("Berlin", "Madrid")] → 1823.89 km
    │  ├─ cache[("Madrid", "Rom")] → 1786.45 km
    │  ├─ cache[("Paris", "Rom")] → 1435.67 km
    │  ├─ cache[("Madrid", "Paris")] → 1265.32 km
    │  └─ cache[("Berlin", "Rom")] → 1534.78 km
    │
    ├─ edges_km = [877.23, 1823.89, 1786.45, 1435.67, 1265.32, 1534.78]
    │
    ├─ Normalize: ÷ 5000
    │  └─ edges_norm = [0.1754, 0.3648, 0.3573, 0.2871, 0.2531, 0.3070]
    │
    └─ Append to batch

RESULT: batch_real shape (16, 6)
        All values ∈ [0, 1]
```

---

## 🎯 Warum Distance Cache?

### Vorteile:
1. **Performance:** Keine Haversine-Berechnung jeden Step (O(1) Lookup statt O(1) calc)
2. **Konsistenz:** Gleiche Abstände über alle Läufe hinweg
3. **Caching:** Kann paralleles Training unterstützen
4. **Debugging:** Reproduzierbar mit gespeicherten Werten

### Haversine Berechnung passiert NUR EINMAL:
```
Initial:  build_distance_dataset.py
  └─ Liest 80 Städte aus cities.csv
  └─ Berechnet alle C(80,2)=3160 Paare mit Haversine
  └─ Speichert in distance_cache.csv

Training:  training_qdis.py / training_qgan.py
  └─ Lädt distance_cache.csv einmal
  └─ Nutzt nur Lookups (kein Rechnen mehr!)
```

---

## 🔗 Wo wird Cache geladen?

### In `training_qdis.py` (Zeile ~230):
```python
cache = load_distance_cache(DISTANCE_CACHE_PATH)
print(f"✓ {len(cache)} Distanzen im Cache")

# Dann in der Trainingsschleife:
success, edges = sample_edges_from_cache(cities, cache, rng)
```

### In `training_qgan.py`:
```python
# Falls dein Code Cache nutzt:
cache = load_distance_cache(DISTANCE_CACHE_PATH)

# Oder: Falls nicht implementiert, bitte nachholten!
```

---

## 📋 Checkliste

- [ ] `distance_cache.csv` existiert
- [ ] `training_qgan.py` lädt Cache in `load_real_edges()`
- [ ] `create_batch_real()` nutzt `sample_edges_from_cache()`
- [ ] Alle 16 Samples pro Batch kommen aus Cache
- [ ] Keine Haversine-Berechnung während Training

---

## 🚀 Falls Cache nicht vorhanden:

```bash
# Rebuild cache:
python build_distance_dataset.py
# → Erstellt neue distance_cache.csv
```

---

**MERKSATZ:** Training sampelt aus 80 Städten mit einem Pre-Computed Cache.  
Das ist ~1000× schneller als Haversine On-The-Fly zu berechnen!

