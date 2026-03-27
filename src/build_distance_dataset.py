"""
build_distance_dataset.py

Systematisch alle paarweisen Distanzen zwischen Städten berechnen und cachen.
- Lädt die Städteliste aus cities.csv
- Iteriert: für jede Stadt S → compute distance to all other cities
- Nutzt Caching konsequent (prüft vor jedem API-Call)
- Bei Fehler (404, rate limit, netzwerk) → markieren und loggen
- Speichert alle berechneten Distanzen in distance_cache.csv
- Output: strukturierte Logdatei mit success/fail-Status pro Paar
"""

import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests

# =========================
# Config
# =========================
CITIES_CSV = "cities.csv"
CACHE_CSV = "distance_cache.csv"
BUILD_LOG = "distance_build_log.csv"

# Routing: OpenRouteService (ORS)
USE_ORS = True
ORS_API_KEY = os.getenv("ORS_API_KEY", "")
ORS_URL = "https://api.openrouteservice.org/v2/directions/driving-car"

# Wartezeit zwischen ORS-Anfragen (Sekunden)
ORS_SLEEP_S = 2.0

# Daily Limit: max. 2000 API-Anfragen pro Tag
# (zählt erfolgreiche Requests + Rate-Limit Retries, excludes cached)
MAX_API_REQUESTS_PER_DAY = 2000


@dataclass(frozen=True)
class City:
    name: str
    country: str
    lat: float
    lon: float


# =========================
# Helper: load_cities()
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
                print(f"Geladene Datei: {candidate} (encoding={enc or 'default'}, sep=';', decimal=',')")
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
                    print(f"Geladene Datei: {candidate} (encoding={enc or 'default'}, sep=',')")
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

    if len(cities) < 2:
        raise ValueError("Du brauchst mindestens 2 Städte in cities.csv.")
    return cities


# =========================
# Cache-Helpers
# =========================
def _pair_key(a: City, b: City) -> Tuple[str, str]:
    """Stabiler, symmetrischer Cache-Key."""
    ka = f"{a.name}|{a.country}"
    kb = f"{b.name}|{b.country}"
    return tuple(sorted((ka, kb)))


def _load_cache(path: str = CACHE_CSV) -> Dict[Tuple[str, str], float]:
    """Laden des bestehenden Caches."""
    cache: Dict[Tuple[str, str], float] = {}
    if not os.path.exists(path):
        return cache
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = (row["k1"], row["k2"])
            cache[k] = float(row["distance_km"])
    return cache


def _append_cache_row(path: str, k1: str, k2: str, dist_km: float) -> None:
    """Append eine neue Distanz zum Cache."""
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["k1", "k2", "distance_km"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({"k1": k1, "k2": k2, "distance_km": dist_km})


def _append_log(
    path: str,
    city_a: str,
    city_b: str,
    status: str,
    distance_km: float = None,
    error_msg: str = None
) -> None:
    """Schreibt einen Log-Eintrag für die Distanzberechnung."""
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["city_a", "city_b", "status", "distance_km", "error_msg"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "city_a": city_a,
            "city_b": city_b,
            "status": status,
            "distance_km": distance_km or "",
            "error_msg": error_msg or ""
        })


# =========================
# Distance Berechnung via ORS
# =========================
def get_route_distance(
    a: City,
    b: City,
    cache: Dict[Tuple[str, str], float],
    use_ors: bool = USE_ORS,
    ors_api_key: str = ORS_API_KEY,
    sleep_s: float = ORS_SLEEP_S
) -> Tuple[float, str, str]:
    """
    Gibt (distance_km, status, error_msg) zurück.
    - Status: "cached", "success", "error_404", "error_429", "error_network", "error_unexpected"
    - Wenn status == "success", wird die Distanz auch zum Cache hinzugefügt.
    """
    k1, k2 = _pair_key(a, b)
    
    if (k1, k2) in cache:
        return cache[(k1, k2)], "cached", ""

    if not use_ors:
        # Demo: Haversine
        R = 6371.0
        phi1, phi2 = math.radians(a.lat), math.radians(b.lat)
        dphi = math.radians(b.lat - a.lat)
        dlmb = math.radians(b.lon - a.lon)
        a_val = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a_val), math.sqrt(1 - a_val))
        dist = R * c
        cache[(k1, k2)] = dist
        _append_cache_row(CACHE_CSV, k1, k2, dist)
        return dist, "success", ""

    if not ors_api_key:
        return None, "error_api_key", "ORS_API_KEY nicht gesetzt"

    headers = {
        "Authorization": ors_api_key,
        "Content-Type": "application/json",
    }
    body = {
        "coordinates": [[a.lon, a.lat], [b.lon, b.lat]]
    }

    max_retries = 6
    attempt = 0
    last_exc = None
    r = None

    while attempt < max_retries:
        try:
            r = requests.post(ORS_URL, json=body, headers=headers, timeout=30)
        except requests.exceptions.RequestException as e:
            last_exc = e
            wait = sleep_s * (2 ** attempt)
            print(f"[{a.name}→{b.name}] Netzwerkfehler (Versuch {attempt+1}/{max_retries}): {e}")
            time.sleep(wait)
            attempt += 1
            continue

        if r.status_code == 200:
            break

        # 404: no routable point
        if r.status_code == 404:
            error_text = r.text if r.text else "no routable point"
            return None, "error_404", error_text

        # 429: Rate limit
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            if ra:
                try:
                    wait = int(ra)
                except ValueError:
                    wait = min(60, sleep_s * (2 ** attempt) * 5)
            else:
                wait = min(60, sleep_s * (2 ** attempt) * 5)
            print(f"[{a.name}→{b.name}] Rate Limit (429, Versuch {attempt+1}/{max_retries}), warte {wait}s")
            time.sleep(wait)
            attempt += 1
            continue

        # 5xx: Server error
        if 500 <= r.status_code < 600:
            wait = min(60, sleep_s * (2 ** attempt))
            print(f"[{a.name}→{b.name}] Serverfehler {r.status_code} (Versuch {attempt+1}/{max_retries}), warte {wait:.1f}s")
            time.sleep(wait)
            attempt += 1
            continue

        # Sonstige Fehler: permanent
        return None, "error_permanent", f"Status {r.status_code}: {r.text[:200]}"

    else:
        # Max retries erreicht
        if last_exc is not None:
            return None, "error_network", f"Nach {max_retries} Versuchen: {str(last_exc)[:200]}"
        else:
            return None, "error_timeout", f"Max retries nach {max_retries} Versuchen"

    # Erfolgreich (200)
    try:
        data = r.json()
    except ValueError:
        return None, "error_json", f"Invalid JSON (status {r.status_code})"

    meters = None

    # Versuche verschiedene ORS-Response-Formen
    if "features" in data and data["features"]:
        try:
            meters = data["features"][0]["properties"]["segments"][0]["distance"]
        except (KeyError, IndexError):
            meters = None

    if meters is None and "routes" in data and data["routes"]:
        try:
            meters = data["routes"][0]["summary"]["distance"]
        except (KeyError, IndexError):
            try:
                meters = data["routes"][0]["segments"][0]["distance"]
            except (KeyError, IndexError):
                meters = None

    if meters is None:
        return None, "error_parse", f"Cannot extract distance from response: {str(data)[:200]}"

    dist = meters / 1000.0
    cache[(k1, k2)] = dist
    _append_cache_row(CACHE_CSV, k1, k2, dist)
    time.sleep(sleep_s)
    return dist, "success", ""


# =========================
# Main: Build Distance Dataset
# =========================
def main():
    print("=" * 70)
    print("Distance Dataset Builder")
    print("=" * 70)

    # Lade Städte
    cities = load_cities(CITIES_CSV)
    print(f"✓ {len(cities)} Städte geladen\n")

    # Lade bestehenden Cache
    cache = _load_cache(CACHE_CSV)
    print(f"✓ Bestehender Cache: {len(cache)} Einträge\n")

    # Statistik
    total_pairs = len(cities) * (len(cities) - 1) // 2  # ungerichtete Paare
    cached = 0
    success = 0
    error_404 = 0
    error_429 = 0
    error_network = 0
    error_other = 0
    api_requests_made = 0  # Zähle tatsächliche API-Anfragen (nicht cached)

    # Iteriere: für jede Stadt S → alle anderen Städte
    print("Starte systematische Distanzberechnung:")
    print(f"Insgesamt {total_pairs} Stadt-Paare zu berechnen/prüfen")
    print(f"Daily Limit: {MAX_API_REQUESTS_PER_DAY} API-Anfragen\n")

    pair_count = 0
    for i, city_a in enumerate(cities):
        for j, city_b in enumerate(cities):
            if i >= j:  # Nur jedes Paar einmal (ungerichtet)
                continue

            pair_count += 1
            pair_label = f"{city_a.name} ↔ {city_b.name}"

            # Prüfe Daily Limit vor API-Call
            # (nur non-cached Anfragen zählen)
            if (city_a.name, city_b.country) not in cache and api_requests_made >= MAX_API_REQUESTS_PER_DAY:
                print(f"\n⚠️  Daily Limit erreicht ({MAX_API_REQUESTS_PER_DAY} API-Anfragen)")
                print(f"Unterbrochen bei Paar {pair_count}/{total_pairs}")
                print(f"Bereits verarbeitet: {cached + success + error_404 + error_429 + error_network + error_other}/{total_pairs}")
                print(f"Gestoppte Arbeit, kann später fortgesetzt werden (Cache wird weiterverwendet)")
                break

            # Berechne Distanz
            dist, status, error_msg = get_route_distance(city_a, city_b, cache)

            # Zähle nur non-cached API-Anfragen
            if status != "cached":
                api_requests_made += 1

            # Logge und sammle Statistik
            _append_log(BUILD_LOG, city_a.name, city_b.name, status, dist, error_msg)

            if status == "cached":
                cached += 1
                print(f"[{pair_count:5d}/{total_pairs}] {pair_label:50s} [CACHED] {dist:.1f} km")
            elif status == "success":
                success += 1
                print(f"[{pair_count:5d}/{total_pairs}] {pair_label:50s} [OK]     {dist:.1f} km")
            elif status == "error_404":
                error_404 += 1
                print(f"[{pair_count:5d}/{total_pairs}] {pair_label:50s} [404]    (no routable point)")
            elif status == "error_429":
                error_429 += 1
                print(f"[{pair_count:5d}/{total_pairs}] {pair_label:50s} [429]    (rate limit)")
            elif status == "error_network":
                error_network += 1
                print(f"[{pair_count:5d}/{total_pairs}] {pair_label:50s} [NET]    {error_msg[:50]}")
            else:
                error_other += 1
                print(f"[{pair_count:5d}/{total_pairs}] {pair_label:50s} [ERR]    {error_msg[:50]}")

    print("\n" + "=" * 70)
    print("Zusammenfassung:")
    print("=" * 70)
    print(f"Insgesamt geprüft: {pair_count} Paare")
    print(f"  - Aus Cache:         {cached:5d} ({100*cached/pair_count:.1f}%)")
    print(f"  - Erfolgreich:       {success:5d} ({100*success/pair_count:.1f}%)")
    print(f"  - Fehler (404):      {error_404:5d} ({100*error_404/pair_count:.1f}%)")
    print(f"  - Fehler (429):      {error_429:5d} ({100*error_429/pair_count:.1f}%)")
    print(f"  - Fehler (Netzwerk): {error_network:5d} ({100*error_network/pair_count:.1f}%)")
    print(f"  - Sonstige Fehler:   {error_other:5d} ({100*error_other/pair_count:.1f}%)")
    print(f"\nAPI-Anfragen gemacht: {api_requests_made}/{MAX_API_REQUESTS_PER_DAY} (Daily Limit)")
    print("\nCache gespeichert in: " + CACHE_CSV)
    print("Log gespeichert in:   " + BUILD_LOG)
    print("=" * 70)


if __name__ == "__main__":
    main()
