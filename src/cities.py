import pandas as pd
import requests
import numpy as np
from pathlib import Path
from urllib.parse import quote

# Datei laden
data_xlsx = Path(__file__).parent / "archive" / "cities.xlsx"
data_csv = Path(__file__).parent / "archive" / "cities.csv"

# Datei laden (nur CSV)
data_csv = Path(__file__).parent / "archive" / "cities.csv"
if not data_csv.exists():
    raise FileNotFoundError(f"{data_csv} nicht gefunden")

# Versuche CSV mit mehreren Encodings zu lesen (Windows Dateien oft cp1252 / latin-1)
# Wichtig: Datei verwendet Semikolon als Trennzeichen und Komma als Dezimaltrennzeichen
read_kwargs = dict(sep=';', decimal=',')
try:
    df = pd.read_csv(data_csv, **read_kwargs)
    source = data_csv
except UnicodeDecodeError:
    try:
        df = pd.read_csv(data_csv, encoding='utf-8-sig', **read_kwargs)
        source = data_csv
        print("Datei mit 'utf-8-sig' erfolgreich gelesen.")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(data_csv, encoding='cp1252', **read_kwargs)
            source = data_csv
            print("Datei mit 'cp1252' (Windows-1252) erfolgreich gelesen.")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(data_csv, encoding='latin-1', **read_kwargs)
                source = data_csv
                print("Datei mit 'latin-1' erfolgreich gelesen.")
            except Exception as e:
                raise RuntimeError("Fehler beim Lesen der CSV-Datei: ungültige Kodierung. Bitte konvertiere die Datei zu UTF-8 oder gib das passende Encoding an.") from e

print(f"Geladene Datei: {source}")

# Normiere Spaltennamen und entferne führende/trailende Leerzeichen
df.columns = [c.strip().lower() for c in df.columns]

# Konvertiere lat/lon zu Float (entferne Leerzeichen und ersetze Komma, falls nötig)
for col in ('lat', 'lon'):
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'\s+','', regex=True).str.replace(',', '.', regex=False), errors='coerce')

# Falls nach der Konvertierung NaNs entstanden sind, zeige problematische Zeilen
if df[['lat','lon']].isna().any().any():
    bad = df[df[['lat','lon']].isna().any(axis=1)]
    raise RuntimeError(f"Einige Zeilen haben ungültige Koordinaten (lat/lon):\n{bad}")

print("Spalten:", df.columns.tolist())
print("Erste Zeilen:\n", df.head())

# Erwartete Spalten prüfen
expected = {"city", "country", "lat", "lon"}
missing = expected - set(df.columns)
if missing:
    raise RuntimeError(f"Fehlende Spalten in der Eingabedatei: {missing}")

# Sicherstellen, dass Koordinaten da sind
df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
if df.empty:
    raise RuntimeError("Keine Städte mit gültigen Koordinaten gefunden")

if len(df) > 100:
    print(f"⚠️ Achtung: {len(df)} Städte — die Anfrage kann wegen URL-Länge oder Serverlimits fehlschlagen. Erwäge Chunking oder lokale Berechnung.")

coords_list = [f"{row.lon},{row.lat}" for row in df.itertuples()]
coords = ";".join(coords_list)
# Quote characters that could cause problems (allow ; , :)
url = f"https://router.project-osrm.org/table/v1/driving/{quote(coords, safe=';,:')}"

try:
    r = requests.get(url, params={"annotations": "distance"}, timeout=60)
    r.raise_for_status()
except requests.RequestException as e:
    resp_text = getattr(e.response, "text", None) if getattr(e, 'response', None) is not None else None
    raise RuntimeError(f"Request an OSRM fehlgeschlagen: {e}\nResponse text: {resp_text}")

j = r.json()
if "distances" not in j:
    raise RuntimeError(f"Keine 'distances' in Antwort: {j}")

dist = np.array(j["distances"])  # Meter

# Städte mit mindestens einem Nachbarn < 100 km
close = []

n = len(df)
all_indices = np.arange(n)
for i, city in enumerate(df["city"]):
    # entferne Distanz zu sich selbst (0)
    neighbors = np.delete(dist[i], i)
    mask = neighbors < 100_000  # Meter
    if np.any(mask):
        # Bestimme die Originalindices der nahen Nachbarn
        idxs = np.delete(all_indices, i)[np.where(mask)[0]]
        neighbor_names = df.loc[idxs, "city"].tolist()
        close.append({
            "city": df.loc[i, "city"],
            "country": df.loc[i, "country"],
            "neighbors_within_100km": "; ".join(neighbor_names)
        })

if close:
    result = pd.DataFrame(close)
    out_file = Path(__file__).parent / "cities_with_neighbor_within_100km.csv"
    result.to_csv(out_file, index=False)
    print(f"Fertig! {len(result)} Städte mit mindestens einem Nachbarn < 100 km gespeichert: {out_file}")
    print("Städteliste:", result["city"].tolist())
else:
    print("Keine Stadt hat einen Nachbarn innerhalb 100 km.")