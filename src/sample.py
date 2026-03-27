"""
sample.py

Generiere Trainingsdaten-Samples aus dem vorberechneten Distance-Cache.
- Lädt cities.csv und distance_cache.csv
- Sampelt 4 zufällige Städte
- Prüft: sind alle 6 Distanzen im Cache vorhanden?
- Falls ja: schreibe Row mit Label "real"
- Falls nein: sample verwerfen

KEINE API-Calls mehr nötig!
"""

import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# =========================
# Config
# =========================
CITIES_CSV = "cities.csv"
OUT_CSV = "edge_length_dataset.csv"
CACHE_CSV = "distance_cache.csv"

# Sampling-Regeln
SAMPLES_TARGET = 100  # Anzahl gültiger Samples


@dataclass(frozen=True)
class City:
    name: str
    country: str
    lat: float
    lon: float


# =========================
# 1) load_cities()
# =========================
def load_cities(path: str = CITIES_CSV) -> List[City]:
    """
    Lädt die Städte-CSV robust:
    - versucht Semikolon als Separator und Komma als Dezimaltrennzeichen
    - probiert mehrere Encodings (utf-8-sig, cp1252, latin-1)
    - normalisiert Spaltennamen und konvertiert lat/lon zu float
    """
    read_kwargs = dict(sep=';', decimal=',', dtype=str)
    df = None
    source_path = None

    path_obj = Path(path)
    script_dir = Path(__file__).parent
    candidates = [Path(path)]
    if not path_obj.is_absolute():
        candidates.append(script_dir / path)
        candidates.append(script_dir / 'archive' / path_obj.name)

    # Deduplicate
    seen = set()
    candidates = [p for p in candidates if not (str(p) in seen or seen.add(str(p)))]

    tried = []
    for candidate in candidates:
        if not candidate.exists():
            tried.append(str(candidate))
            continue
        for enc in (None, 'utf-8-sig', 'cp1252', 'latin-1'):
            try:
                if enc is None:
                    df = pd.read_csv(candidate, **read_kwargs)
                else:
                    df = pd.read_csv(candidate, encoding=enc, **read_kwargs)
                print(f"✓ Städte geladen: {candidate}")
                source_path = candidate
                break
            except UnicodeDecodeError:
                continue
            except pd.errors.ParserError:
                try:
                    if enc is None:
                        df = pd.read_csv(candidate, sep=',', decimal='.', dtype=str)
                    else:
                        df = pd.read_csv(candidate, encoding=enc, sep=',', decimal='.', dtype=str)
                    print(f"✓ Städte geladen: {candidate}")
                    source_path = candidate
                    break
                except Exception:
                    continue
        if df is not None:
            break
        else:
            tried.append(str(candidate))

    if df is None:
        raise FileNotFoundError(f"Keine passende Eingabedatei gefunden. Versuchte Pfade: {tried}")

    # Normalisiere Spaltennamen
    df.columns = [c.strip().lower() for c in df.columns]

    if 'latitude' in df.columns and 'lat' not in df.columns:
        df = df.rename(columns={'latitude': 'lat'})
    if 'longitude' in df.columns and 'lon' not in df.columns:
        df = df.rename(columns={'longitude': 'lon'})

    required = {"city", "country", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source_path} braucht Spalten: {sorted(required)}. Gefunden: {list(df.columns)}")

    # Konvertiere lat/lon zu Float
    for col in ('lat', 'lon'):
        df[col] = df[col].astype(str).str.replace(r'\s+', '', regex=True).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if df[['lat', 'lon']].isna().any().any():
        bad = df[df[['lat', 'lon']].isna().any(axis=1)]
        raise ValueError(f"Ungültige Koordinaten in {path}:\n{bad}")

    cities: List[City] = []
    for _, r in df.iterrows():
        cities.append(City(str(r["city"]).strip(), str(r["country"]).strip(), float(r["lat"]), float(r["lon"])))

    if len(cities) < 4:
        raise ValueError("Du brauchst mindestens 4 Städte in cities.csv.")
    return cities


# =========================
# 2) Cache-Helpers
# =========================
def _pair_key(a: City, b: City) -> Tuple[str, str]:
    """Stabiler, symmetrischer Cache-Key."""
    ka = f"{a.name}|{a.country}"
    kb = f"{b.name}|{b.country}"
    return tuple(sorted((ka, kb)))


def _load_cache(path: str = CACHE_CSV) -> Dict[Tuple[str, str], float]:
    """Lade den kompletten Distance-Cache aus CSV."""
    cache: Dict[Tuple[str, str], float] = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache nicht gefunden: {path}. Starte erst build_distance_dataset.py!")
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = (row["k1"], row["k2"])
            cache[k] = float(row["distance_km"])
    print(f"✓ Cache geladen: {len(cache)} Einträge\n")
    return cache


# =========================
# 3) sample_four_cities()
# =========================
def sample_four_cities(cities: List[City]) -> Tuple[City, City, City, City]:
    """Wähle 4 zufällige unterschiedliche Städte."""
    return tuple(random.sample(cities, 4))


# =========================
# 4) Prüfe ob alle 6 Distanzen im Cache vorhanden sind
# =========================
def all_distances_cached(
    a: City,
    b: City,
    c: City,
    d: City,
    cache: Dict[Tuple[str, str], float]
) -> bool:
    """
    Prüfe: sind alle 6 Kanten eines 4er-Samples im Cache vorhanden?
    Returns: True falls alle da sind, sonst False
    """
    pairs = [
        _pair_key(a, b),  # e_ab
        _pair_key(b, c),  # e_bc
        _pair_key(c, d),  # e_cd
        _pair_key(d, a),  # e_da
        _pair_key(a, c),  # e_ac
        _pair_key(b, d),  # e_bd
    ]
    return all(pair in cache for pair in pairs)


# =========================
# 5) Baue Sample aus Cache
# =========================
def build_sample(
    a: City,
    b: City,
    c: City,
    d: City,
    cache: Dict[Tuple[str, str], float]
) -> Dict[str, float]:
    """
    Lese alle 6 Distanzen aus dem Cache.
    (Annahme: all_distances_cached() wurde bereits geprüft!)
    """
    e_ab = cache[_pair_key(a, b)]
    e_bc = cache[_pair_key(b, c)]
    e_cd = cache[_pair_key(c, d)]
    e_da = cache[_pair_key(d, a)]
    e_ac = cache[_pair_key(a, c)]
    e_bd = cache[_pair_key(b, d)]

    return {
        "e_ab": e_ab,
        "e_bc": e_bc,
        "e_cd": e_cd,
        "e_da": e_da,
        "e_ac": e_ac,
        "e_bd": e_bd,
    }


# =========================
# 6) Schreibe Output
# =========================
def append_to_csv(row: Dict[str, float], out_path: str = OUT_CSV) -> None:
    """Append eine Row zum Output-CSV."""
    exists = os.path.exists(out_path)
    
    base_fields = ["e_ab", "e_bc", "e_cd", "e_da", "e_ac", "e_bd"]
    city_fields = [f"city_{x}" for x in ("a", "b", "c", "d") if f"city_{x}" in row]
    pair_fields = [f"pair_{p}" for p in ("ab", "bc", "cd", "da", "ac", "bd") if f"pair_{p}" in row]
    label_fields = ["label"] if "label" in row else []
    
    # Reihenfolge: Cities, Pairs, Label, dann Distanzen
    fieldnames = city_fields + pair_fields + label_fields + base_fields
    
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        out = {k: row.get(k, "") for k in fieldnames}
        writer.writerow(out)


# =========================
# Main: Generate Dataset
# =========================
def main():
    print("=" * 70)
    print("Sample Generator (Cache-basiert)")
    print("=" * 70 + "\n")
    
    random.seed(42)
    
    # Lade Städte und Cache
    cities = load_cities(CITIES_CSV)
    print(f"✓ {len(cities)} Städte geladen\n")
    
    cache = _load_cache(CACHE_CSV)
    
    # Statistik
    made = 0
    tried = 0
    skipped = 0
    
    print(f"Ziel: {SAMPLES_TARGET} gültige Samples\n")
    
    # Samplen bis wir SAMPLES_TARGET gültige Samples haben
    while made < SAMPLES_TARGET:
        tried += 1
        
        a, b, c, d = sample_four_cities(cities)
        
        # Prüfe: sind alle 6 Distanzen im Cache?
        if not all_distances_cached(a, b, c, d, cache):
            skipped += 1
            continue
        
        # Baue Sample aus Cache
        row = build_sample(a, b, c, d, cache)
        
        # Füge Metadaten hinzu
        row.update({
            "city_a": a.name,
            "city_b": b.name,
            "city_c": c.name,
            "city_d": d.name,
            "pair_ab": f"{a.name}|{b.name}",
            "pair_bc": f"{b.name}|{c.name}",
            "pair_cd": f"{c.name}|{d.name}",
            "pair_da": f"{d.name}|{a.name}",
            "pair_ac": f"{a.name}|{c.name}",
            "pair_bd": f"{b.name}|{d.name}",
            "label": "real",  # Alle Samples aus Cache sind "echt"
        })
        
        # Schreibe Output
        append_to_csv(row)
        
        made += 1
        
        if made % 10 == 0:
            print(f"[{made:3d}/{SAMPLES_TARGET}] {a.name} - {b.name} - {c.name} - {d.name}")
    
    print("\n" + "=" * 70)
    print("Zusammenfassung:")
    print("=" * 70)
    print(f"Gültige Samples: {made}/{SAMPLES_TARGET}")
    print(f"Insgesamt versucht: {tried}")
    print(f"Übersprungen (unvollständig): {skipped}")
    print(f"Erfolgsrate: {100*made/tried:.1f}%")
    print(f"\nDataset gespeichert: {OUT_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
