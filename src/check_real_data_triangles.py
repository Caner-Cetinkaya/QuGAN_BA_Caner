#!/usr/bin/env python3
"""
Check REAL 4-city tuples for triangle-inequality consistency.

Default behavior:
- iterate over ALL unordered 4-city combinations
- build the 6-edge tuple in the project order
    (e_ab, e_bc, e_cd, e_da, e_ac, e_bd)
- verify the 4 implied triangles
    abc, abd, acd, bcd
- log progress
- optionally save valid / invalid / missing tuples to CSV
- optionally save a JSON summary

This script is intentionally standalone so training_qgan.py stays unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


EDGE_NAMES = ("e_ab", "e_bc", "e_cd", "e_da", "e_ac", "e_bd")
TRIANGLE_NAMES = ("abc", "abd", "acd", "bcd")


class CsvStreamWriter:
    """Write rows incrementally so we do not keep huge lists in memory."""

    def __init__(self, path: str | None):
        self.path = Path(path) if path else None
        self._file = None
        self._writer = None
        self._fieldnames: list[str] | None = None
        self.rows_written = 0

    def write_row(self, row: dict[str, Any]) -> None:
        if self.path is None:
            return
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("w", newline="", encoding="utf-8")
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        assert self._writer is not None
        self._writer.writerow(row)
        self.rows_written += 1

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def load_cities(path: str = "cities.csv") -> list[dict[str, Any]]:
    """Load cities using the same tolerant parsing style as training_qgan.py."""
    read_kwargs = dict(sep=";", decimal=",", dtype=str)
    df = None

    candidates = [Path(path)]
    script_dir = Path(__file__).parent
    if not Path(path).is_absolute():
        candidates.append(script_dir / path)
        candidates.append(script_dir / "archive" / Path(path).name)

    for candidate in candidates:
        if not candidate.exists():
            continue
        for enc in (None, "utf-8-sig", "cp1252", "latin-1"):
            try:
                if enc is None:
                    df = pd.read_csv(candidate, **read_kwargs)
                else:
                    df = pd.read_csv(candidate, encoding=enc, **read_kwargs)
                break
            except Exception:
                continue
        if df is not None:
            break

    if df is None:
        raise FileNotFoundError(f"cities.csv not found: checked {[str(c) for c in candidates]}")

    df.columns = [c.strip().lower() for c in df.columns]
    if "latitude" in df.columns:
        df = df.rename(columns={"latitude": "lat"})
    if "longitude" in df.columns:
        df = df.rename(columns={"longitude": "lon"})

    for col in ("lat", "lon"):
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"\s+", "", regex=True)
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cities: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        cities.append(
            {
                "name": str(r["city"]).strip(),
                "country": str(r["country"]).strip(),
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
            }
        )
    return cities


def _pair_key(a: dict[str, Any], b: dict[str, Any]) -> tuple[str, str]:
    """Stable cache key for an unordered city pair."""
    ka = f"{a['name']}|{a['country']}"
    kb = f"{b['name']}|{b['country']}"
    return tuple(sorted((ka, kb)))


def load_distance_cache(cache_path: str = "distance_cache.csv") -> dict[tuple[str, str], float]:
    cache_file = Path(cache_path)
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_file}")

    cache: dict[tuple[str, str], float] = {}
    with cache_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cache[(row["k1"], row["k2"])] = float(row["distance_km"])
    return cache


def edges_from_city_quad(
    city_quad: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
    cache: dict[tuple[str, str], float],
) -> tuple[bool, dict[str, Any]]:
    """Build the full 6-edge tuple for one fixed 4-city combination."""
    a, b, c, d = city_quad

    pair_keys = [
        _pair_key(a, b),  # e_ab
        _pair_key(b, c),  # e_bc
        _pair_key(c, d),  # e_cd
        _pair_key(d, a),  # e_da
        _pair_key(a, c),  # e_ac
        _pair_key(b, d),  # e_bd
    ]

    missing_idx = [i for i, pair in enumerate(pair_keys) if pair not in cache]
    record: dict[str, Any] = {
        "cities": [a, b, c, d],
        "pair_keys": pair_keys,
    }

    if missing_idx:
        record["missing_indices"] = missing_idx
        record["missing_edges"] = [EDGE_NAMES[i] for i in missing_idx]
        return False, record

    record["edges_km"] = np.array([cache[pair] for pair in pair_keys], dtype=float)
    return True, record


def triangle_ok(x: float, y: float, z: float, eps: float = 1e-6) -> bool:
    return (
        x <= y + z + eps
        and y <= x + z + eps
        and z <= x + y + eps
    )


def triangle_margin(x: float, y: float, z: float) -> float:
    """Positive means valid, negative means violation severity."""
    return min((y + z - x), (x + z - y), (x + y - z))


def evaluate_tuple(edges: np.ndarray, eps: float = 1e-6) -> dict[str, Any]:
    ab, bc, cd, da, ac, bd = [float(v) for v in edges]
    triangles = {
        "abc": (ab, bc, ac),
        "abd": (ab, da, bd),
        "acd": (ac, cd, da),
        "bcd": (bc, cd, bd),
    }

    tri_ok: dict[str, bool] = {}
    tri_margin: dict[str, float] = {}
    for name, (x, y, z) in triangles.items():
        tri_ok[name] = triangle_ok(x, y, z, eps=eps)
        tri_margin[name] = triangle_margin(x, y, z)

    return {
        "tuple_ok": all(tri_ok.values()),
        "triangle_ok": tri_ok,
        "triangle_margin": tri_margin,
    }


def make_row(record: dict[str, Any], evaluation: dict[str, Any] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {}

    cities = record.get("cities")
    if cities:
        labels = ["a", "b", "c", "d"]
        for label, city in zip(labels, cities):
            row[f"city_{label}"] = city["name"]
            row[f"country_{label}"] = city["country"]
            row[f"lat_{label}"] = city["lat"]
            row[f"lon_{label}"] = city["lon"]

    edges = record.get("edges_km")
    if edges is not None:
        for name, value in zip(EDGE_NAMES, edges):
            row[name] = float(value)

    pair_keys = record.get("pair_keys")
    if pair_keys:
        for i, pair in enumerate(pair_keys):
            row[f"pair_{EDGE_NAMES[i]}"] = f"{pair[0]} <-> {pair[1]}"

    if evaluation is not None:
        row["tuple_ok"] = bool(evaluation["tuple_ok"])
        for name in TRIANGLE_NAMES:
            row[f"ok_{name}"] = bool(evaluation["triangle_ok"][name])
            row[f"margin_{name}"] = float(evaluation["triangle_margin"][name])

    if "missing_edges" in record:
        row["missing_edges"] = ",".join(record["missing_edges"])
        row["missing_indices"] = ",".join(map(str, record.get("missing_indices", [])))

    return row


def iter_city_quads(
    cities: list[dict[str, Any]],
    mode: str,
    seed: int,
    samples: int,
) -> tuple[Iterable[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]], int]:
    if mode == "all":
        total = math.comb(len(cities), 4)
        return combinations(cities, 4), total

    rng = np.random.default_rng(seed)

    def _sample_iter():
        for _ in range(samples):
            idx = rng.choice(len(cities), size=4, replace=False)
            yield tuple(cities[i] for i in idx)

    return _sample_iter(), samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Check triangle inequality on real 4-city tuples")
    parser.add_argument("--cities", type=str, default="cities.csv", help="Path to cities.csv")
    parser.add_argument("--cache", type=str, default="distance_cache.csv", help="Path to distance_cache.csv")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "sample"],
        help="Check all unordered 4-city combinations or random samples",
    )
    parser.add_argument("--samples", type=int, default=10000, help="Used only when --mode sample")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sample mode")
    parser.add_argument("--limit", type=int, default=None, help="Optional early stop for quick tests")
    parser.add_argument("--eps", type=float, default=1e-6, help="Tolerance for triangle checks")
    parser.add_argument("--progress-every", type=int, default=50000, help="Print progress every N tuples")
    parser.add_argument("--save-valid", type=str, default=None, help="Optional CSV path for valid tuples")
    parser.add_argument("--save-invalid", type=str, default=None, help="Optional CSV path for invalid tuples")
    parser.add_argument("--save-missing", type=str, default=None, help="Optional CSV path for tuples with missing cache edges")
    parser.add_argument("--summary-json", type=str, default=None, help="Optional JSON path for summary stats")
    args = parser.parse_args()

    cities = load_cities(args.cities)
    cache = load_distance_cache(args.cache)
    quad_iter, total_quads = iter_city_quads(cities, args.mode, args.seed, args.samples)

    if args.limit is not None:
        total_to_process = min(total_quads, args.limit)
    else:
        total_to_process = total_quads

    print("=" * 72)
    print("REAL DATA TRIANGLE CHECK")
    print("=" * 72)
    print(f"cities: {args.cities}")
    print(f"cache:  {args.cache}")
    print(f"n_cities={len(cities)}  cache_edges={len(cache)}")
    print(f"mode={args.mode}  total_to_process={total_to_process}  eps={args.eps}")
    if args.mode == "sample":
        print(f"samples={args.samples}  seed={args.seed}")

    summary: dict[str, Any] = {
        "mode": args.mode,
        "n_cities": len(cities),
        "cache_edges": len(cache),
        "total_possible_unordered_4_city_combinations": math.comb(len(cities), 4),
        "total_requested": total_to_process,
        "processed": 0,
        "complete_tuples": 0,
        "missing_cache": 0,
        "valid_tuples": 0,
        "invalid_tuples": 0,
        "triangle_fail_counts": {name: 0 for name in TRIANGLE_NAMES},
        "worst_margin_by_triangle": {name: None for name in TRIANGLE_NAMES},
    }

    valid_writer = CsvStreamWriter(args.save_valid)
    invalid_writer = CsvStreamWriter(args.save_invalid)
    missing_writer = CsvStreamWriter(args.save_missing)

    try:
        for idx, quad in enumerate(quad_iter, start=1):
            if args.limit is not None and idx > args.limit:
                break

            summary["processed"] = idx
            success, record = edges_from_city_quad(quad, cache)

            if not success:
                summary["missing_cache"] += 1
                missing_writer.write_row(make_row(record))
            else:
                summary["complete_tuples"] += 1
                evaluation = evaluate_tuple(record["edges_km"], eps=args.eps)

                if evaluation["tuple_ok"]:
                    summary["valid_tuples"] += 1
                    valid_writer.write_row(make_row(record, evaluation))
                else:
                    summary["invalid_tuples"] += 1
                    invalid_writer.write_row(make_row(record, evaluation))
                    for tri_name in TRIANGLE_NAMES:
                        if not evaluation["triangle_ok"][tri_name]:
                            summary["triangle_fail_counts"][tri_name] += 1

                for tri_name in TRIANGLE_NAMES:
                    margin = evaluation["triangle_margin"][tri_name]
                    old = summary["worst_margin_by_triangle"][tri_name]
                    if old is None or margin < old:
                        summary["worst_margin_by_triangle"][tri_name] = margin

            if args.progress_every > 0 and (
                idx % args.progress_every == 0 or idx == total_to_process
            ):
                complete = max(summary["complete_tuples"], 1)
                valid_rate = summary["valid_tuples"] / complete
                print(
                    f"[{idx:>9d}/{total_to_process}] "
                    f"complete={summary['complete_tuples']}  "
                    f"missing={summary['missing_cache']}  "
                    f"valid={summary['valid_tuples']}  "
                    f"invalid={summary['invalid_tuples']}  "
                    f"valid_rate_on_complete={valid_rate:.4%}"
                )
    finally:
        valid_writer.close()
        invalid_writer.close()
        missing_writer.close()

    complete = max(summary["complete_tuples"], 1)
    summary["valid_rate_on_complete"] = summary["valid_tuples"] / complete
    summary["invalid_rate_on_complete"] = summary["invalid_tuples"] / complete
    summary["missing_rate_on_processed"] = summary["missing_cache"] / max(summary["processed"], 1)
    summary["saved_valid_rows"] = valid_writer.rows_written
    summary["saved_invalid_rows"] = invalid_writer.rows_written
    summary["saved_missing_rows"] = missing_writer.rows_written

    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    if args.save_valid:
        print(f"saved valid tuples   -> {Path(args.save_valid)} ({valid_writer.rows_written} rows)")
    if args.save_invalid:
        print(f"saved invalid tuples -> {Path(args.save_invalid)} ({invalid_writer.rows_written} rows)")
    if args.save_missing:
        print(f"saved missing tuples -> {Path(args.save_missing)} ({missing_writer.rows_written} rows)")

    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"saved summary json   -> {path}")


if __name__ == "__main__":
    main()
