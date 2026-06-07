import numpy as np
from generator import QGenerator


def triangle_penalty_per_sample(batch: np.ndarray) -> np.ndarray:
    ab = batch[:, 0]
    bc = batch[:, 1]
    cd = batch[:, 2]
    da = batch[:, 3]
    ac = batch[:, 4]
    bd = batch[:, 5]

    violation = (
        np.maximum(0.0, ab - (bc + ac)) +
        np.maximum(0.0, bc - (ab + ac)) +
        np.maximum(0.0, ac - (ab + bc)) +

        np.maximum(0.0, ab - (da + bd)) +
        np.maximum(0.0, da - (ab + bd)) +
        np.maximum(0.0, bd - (ab + da)) +

        np.maximum(0.0, ac - (cd + da)) +
        np.maximum(0.0, cd - (ac + da)) +
        np.maximum(0.0, da - (ac + cd)) +

        np.maximum(0.0, bc - (cd + bd)) +
        np.maximum(0.0, cd - (bc + bd)) +
        np.maximum(0.0, bd - (bc + cd))
    )
    return violation / 12.0


def triangle_valid_mask_np(batch: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    ab = batch[:, 0]
    bc = batch[:, 1]
    cd = batch[:, 2]
    da = batch[:, 3]
    ac = batch[:, 4]
    bd = batch[:, 5]

    def tri_ok(x, y, z):
        return (x <= y + z + tol) & (y <= x + z + tol) & (z <= x + y + tol)

    return (
        tri_ok(ab, bc, ac)
        & tri_ok(ab, da, bd)
        & tri_ok(ac, cd, da)
        & tri_ok(bc, cd, bd)
    )


def main():
    # an deinen aktuellen Test anpassen
    gen = QGenerator(n_layer=2, seed=0)

    # falls du im Training mit init_std=0.5 arbeitest:
    gen.weights = gen.rng.normal(0.0, 0.5, size=gen.weights.shape)

    rng = np.random.default_rng(12345)

    candidates = []
    tries = 2000

    for _ in range(tries):
        noise = rng.uniform(0.0, 2 * np.pi, size=(1, 6))
        out = np.asarray(gen.batch_forward(noise), dtype=float)

        penalty = float(triangle_penalty_per_sample(out)[0])
        valid = bool(triangle_valid_mask_np(out)[0])

        if not valid:
            candidates.append((penalty, noise[0].copy(), out[0].copy()))

    candidates.sort(key=lambda x: x[0], reverse=True)

    print(f"Gefundene ungültige Samples: {len(candidates)}")
    print()

    top_k = min(3, len(candidates))
    for i in range(top_k):
        penalty, noise, out = candidates[i]
        print(f"=== Kandidat {i+1} ===")
        print("noise =", np.round(noise, 6))
        print("out   =", np.round(out, 6))
        print("penalty =", round(penalty, 8))
        print()

    if top_k == 0:
        print("Keine ungültigen Samples gefunden.")
        print("Dann entweder tries erhöhen oder eine andere Initialisierung testen.")


if __name__ == "__main__":
    main()