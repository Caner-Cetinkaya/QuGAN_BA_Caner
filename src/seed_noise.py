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


def find_invalid_for_config(
    seed: int,
    init_std: float,
    n_layers: int = 2,
    n_tries: int = 1000,
    latent_dim: int = 6,
    search_seed: int = 12345,
):
    gen = QGenerator(n_layer=n_layers, seed=seed)

    # gleiches Prinzip wie in deinem Trainingsskript
    gen.weights = gen.rng.normal(0.0, init_std, size=gen.weights.shape)

    rng = np.random.default_rng(search_seed)

    best = None
    found = []

    for _ in range(n_tries):
        noise = rng.uniform(0.0, 2 * np.pi, size=(1, latent_dim))
        out = np.asarray(gen.batch_forward(noise), dtype=float)

        penalty = float(triangle_penalty_per_sample(out)[0])
        valid = bool(triangle_valid_mask_np(out)[0])

        if (best is None) or (penalty > best["penalty"]):
            best = {
                "seed": seed,
                "init_std": init_std,
                "noise": noise[0].copy(),
                "out": out[0].copy(),
                "penalty": penalty,
                "valid": valid,
            }

        if not valid:
            found.append({
                "seed": seed,
                "init_std": init_std,
                "noise": noise[0].copy(),
                "out": out[0].copy(),
                "penalty": penalty,
            })

    return found, best


def main():
    seeds = list(range(0, 31))
    init_stds = [0.5, 1.0, 1.5, 2.0, 3.0]
    n_tries = 1000
    n_layers = 2

    all_found = []
    global_best = None

    for seed in seeds:
        for init_std in init_stds:
            found, best = find_invalid_for_config(
                seed=seed,
                init_std=init_std,
                n_layers=n_layers,
                n_tries=n_tries,
            )

            print(
                f"seed={seed:2d}  init_std={init_std:>3}  "
                f"invalid_found={len(found):4d}  "
                f"best_penalty={best['penalty']:.6f}"
            )

            if global_best is None or best["penalty"] > global_best["penalty"]:
                global_best = best

            all_found.extend(found)

    print("\n" + "=" * 70)
    print(f"Gesamtzahl ungültiger Samples: {len(all_found)}")

    if all_found:
        all_found.sort(key=lambda x: x["penalty"], reverse=True)
        print("\nTop 3 ungültige Kandidaten:\n")
        for i, item in enumerate(all_found[:3], start=1):
            print(f"--- Kandidat {i} ---")
            print(f"seed     : {item['seed']}")
            print(f"init_std : {item['init_std']}")
            print(f"penalty  : {item['penalty']:.6f}")
            print(f"noise    : {np.round(item['noise'], 6)}")
            print(f"output   : {np.round(item['out'], 6)}")
            print()
    else:
        print("\nKeine ungültigen Samples gefunden.")
        print("Bestes (stärkst verletzendes) gültiges/fast-gültiges Sample:")
        print(f"seed     : {global_best['seed']}")
        print(f"init_std : {global_best['init_std']}")
        print(f"penalty  : {global_best['penalty']:.6f}")
        print(f"valid    : {global_best['valid']}")
        print(f"noise    : {np.round(global_best['noise'], 6)}")
        print(f"output   : {np.round(global_best['out'], 6)}")


if __name__ == "__main__":
    main()