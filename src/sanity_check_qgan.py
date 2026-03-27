import numpy as np

from training_qgan import (
    load_cities,
    load_distance_cache,
    create_batch_real,
    train_discriminator_step,
    _disc_probs_from_edges,
)
from generator import QGenerator
from discriminator import QDiscriminator
from config import BATCH_SIZE, SEED


def main():
    rng = np.random.default_rng(SEED)

    # Modelle laden
    disc = QDiscriminator(n_layer=2, seed=SEED)
    gen = QGenerator(n_layer=2, seed=SEED)

    # Daten laden
    cities = load_cities("cities.csv")
    cache = load_distance_cache("distance_cache.csv")

    # Fester Real-Batch
    batch_real = create_batch_real(cities, BATCH_SIZE, cache, rng)

    # Fester Fake-Batch vom aktuellen Generator
    #noise_batch = rng.uniform(0, 1, size=(BATCH_SIZE, 6))
    #batch_fake = gen.batch_forward(noise_batch)
    #batch_fake = np.full((BATCH_SIZE, 6), 0.5)
    batch_fake = rng.uniform(0.0, 1.0, size=(BATCH_SIZE, 6))

    # Initiale Scores
    real_scores = np.array(_disc_probs_from_edges(disc, disc.weights, batch_real), dtype=float)
    fake_scores = np.array(_disc_probs_from_edges(disc, disc.weights, batch_fake), dtype=float)

    print("=" * 60)
    print("QGAN SANITY CHECK: DISCRIMINATOR ONLY")
    print("=" * 60)
    print(f"Batch size: {BATCH_SIZE}")
    print("\nInitial:")
    print(f"  Real score mean: {real_scores.mean():.4f}")
    print(f"  Fake score mean: {fake_scores.mean():.4f}")
    print(f"  Real min/max:    {real_scores.min():.4f} / {real_scores.max():.4f}")
    print(f"  Fake min/max:    {fake_scores.min():.4f} / {fake_scores.max():.4f}")

    # Nur D trainieren, G bleibt fest
    for step in range(1, 301):
        d_metrics = train_discriminator_step(disc, batch_real, batch_fake)

        if step % 25 == 0 or step == 1:
            real_scores = np.array(_disc_probs_from_edges(disc, disc.weights, batch_real), dtype=float)
            fake_scores = np.array(_disc_probs_from_edges(disc, disc.weights, batch_fake), dtype=float)

            print(f"\nStep {step}")
            print(f"  Disc loss total: {d_metrics['disc_loss']:.6f}")
            print(f"  Disc loss real:  {d_metrics['disc_loss_real']:.6f}")
            print(f"  Disc loss fake:  {d_metrics['disc_loss_fake']:.6f}")
            print(f"  Real score mean: {real_scores.mean():.4f}")
            print(f"  Fake score mean: {fake_scores.mean():.4f}")
            print(f"  Real min/max:    {real_scores.min():.4f} / {real_scores.max():.4f}")
            print(f"  Fake min/max:    {fake_scores.min():.4f} / {fake_scores.max():.4f}")
            print(f"  Grad norm:       {d_metrics['disc_grad_norm']:.6f}")

    print("\n" + "=" * 60)
    print("ERWARTUNG")
    print("=" * 60)
    print("Wenn der Discriminator grundsätzlich funktioniert, sollte gelten:")
    print("  - Real score mean -> Richtung 1")
    print("  - Fake score mean -> Richtung 0")
    print("  - disc_loss_real sinkt")
    print("  - disc_loss_fake sinkt")
    print("\nWenn das NICHT passiert, liegt das Problem eher bei:")
    print("  - Discriminator-Architektur")
    print("  - Quantum-Circuit / Encoding")
    print("  - Optimierbarkeit / Gradienten")
    print("und nicht primär an der adversarialen Dynamik.")


if __name__ == "__main__":
    main()