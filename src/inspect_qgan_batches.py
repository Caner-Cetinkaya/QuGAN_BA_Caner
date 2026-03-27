import numpy as np

import cities
from training_qgan import (
    load_cities,
    load_distance_cache,
    create_batch_real,
)
from generator import QGenerator
from config import BATCH_SIZE, SEED


EDGE_NAMES = ["ab", "bc", "cd", "da", "ac", "bd"]


def print_stats(name, batch):
    print(f"\n{name}")
    print("-" * 60)
    for i, edge_name in enumerate(EDGE_NAMES):
        col = batch[:, i]
        print(
            f"{edge_name:>2} | "
            f"mean={col.mean():.4f} | "
            f"std={col.std():.4f} | "
            f"min={col.min():.4f} | "
            f"max={col.max():.4f}"
        )

FAKE_MODE = "constant"   # "generator", "constant", "uniform"

def main():
    rng = np.random.default_rng(SEED)

    cities = load_cities("cities.csv")
    cache = load_distance_cache("distance_cache.csv")

    gen = QGenerator(n_layer=2, seed=SEED)

    # fester Real-Batch
    batch_real = create_batch_real(cities, BATCH_SIZE, cache, rng)

    # fester Fake-Batch
    batch_real = create_batch_real(cities, BATCH_SIZE, cache, rng)

    if FAKE_MODE == "generator":
            gen = QGenerator(n_layer=2, seed=SEED)
            noise_batch = rng.uniform(0, 1, size=(BATCH_SIZE, 6))
            batch_fake = gen.batch_forward(noise_batch)

    elif FAKE_MODE == "constant":
            batch_fake = np.full((BATCH_SIZE, 6), 0.5, dtype=float)

    elif FAKE_MODE == "uniform":
            batch_fake = rng.uniform(0.0, 1.0, size=(BATCH_SIZE, 6))

    else:
            raise ValueError(f"Unknown FAKE_MODE: {FAKE_MODE}")

    print("=" * 60)
    print("QGAN BATCH INSPECTION")
    print("=" * 60)
    print(f"Batch size: {BATCH_SIZE}")

    print_stats("REAL BATCH", batch_real)
    print_stats("FAKE BATCH", batch_fake)

    print("\nDifference (fake - real)")
    print("-" * 60)
    for i, edge_name in enumerate(EDGE_NAMES):
        real_col = batch_real[:, i]
        fake_col = batch_fake[:, i]
        print(
            f"{edge_name:>2} | "
            f"Δmean={(fake_col.mean() - real_col.mean()):+.4f} | "
            f"Δstd={(fake_col.std() - real_col.std()):+.4f}"
        )


if __name__ == "__main__":
    main()