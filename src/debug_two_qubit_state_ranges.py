from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml


OBSERVABLES = {
    "X": qml.PauliX,
    "Y": qml.PauliY,
    "Z": qml.PauliZ,
}


def build_device() -> qml.devices.Device:
    return qml.device("default.qubit", wires=2, shots=None)


def make_circuits(dev):
    @qml.qnode(dev)
    def state_circuit(noise0: float, noise1: float, w):
        qml.RY(noise0, wires=0)
        qml.RY(noise1, wires=1)

        qml.RX(w[0, 0], wires=0)
        qml.RY(w[0, 1], wires=0)
        qml.RZ(w[0, 2], wires=0)

        qml.RX(w[1, 0], wires=1)
        qml.RY(w[1, 1], wires=1)
        qml.RZ(w[1, 2], wires=1)

        qml.CNOT(wires=[0, 1])
        return qml.state()

    def make_expval_qnode(obs0: str, obs1: str):
        obs_cls0 = OBSERVABLES[obs0]
        obs_cls1 = OBSERVABLES[obs1]

        @qml.qnode(dev)
        def expval_circuit(noise0: float, noise1: float, w):
            qml.RY(noise0, wires=0)
            qml.RY(noise1, wires=1)

            qml.RX(w[0, 0], wires=0)
            qml.RY(w[0, 1], wires=0)
            qml.RZ(w[0, 2], wires=0)

            qml.RX(w[1, 0], wires=1)
            qml.RY(w[1, 1], wires=1)
            qml.RZ(w[1, 2], wires=1)

            qml.CNOT(wires=[0, 1])
            return [qml.expval(obs_cls0(0)), qml.expval(obs_cls1(1))]

        return expval_circuit

    return state_circuit, make_expval_qnode


def sample_weights(seed: int, init_std: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, init_std, size=(2, 3))


def bloch_from_state(state: np.ndarray) -> dict[str, np.ndarray]:
    rho = np.outer(state, np.conj(state))
    rho0 = np.array([[rho[0, 0] + rho[1, 1], rho[0, 2] + rho[1, 3]],
                     [rho[2, 0] + rho[3, 1], rho[2, 2] + rho[3, 3]]], dtype=complex)
    rho1 = np.array([[rho[0, 0] + rho[2, 2], rho[0, 1] + rho[2, 3]],
                     [rho[1, 0] + rho[3, 2], rho[1, 1] + rho[3, 3]]], dtype=complex)

    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

    def bloch_vec(rho_single: np.ndarray) -> np.ndarray:
        return np.array([
            np.real(np.trace(rho_single @ pauli_x)),
            np.real(np.trace(rho_single @ pauli_y)),
            np.real(np.trace(rho_single @ pauli_z)),
        ])

    return {"q0": bloch_vec(rho0), "q1": bloch_vec(rho1)}


def scan_ranges(state_circuit, make_expval_qnode, weights: np.ndarray, grid: int) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    angles = np.linspace(0.0, 2.0 * np.pi, grid)
    rows: list[dict[str, float]] = []

    qnodes = {f"{a}{b}": make_expval_qnode(a, b) for a in "XYZ" for b in "XYZ"}

    for noise0 in angles:
        for noise1 in angles:
            row = {
                "noise0": float(noise0),
                "noise1": float(noise1),
            }
            state = np.array(state_circuit(float(noise0), float(noise1), weights), dtype=complex)
            bloch = bloch_from_state(state)
            row.update({
                "q0_bloch_x": float(bloch["q0"][0]),
                "q0_bloch_y": float(bloch["q0"][1]),
                "q0_bloch_z": float(bloch["q0"][2]),
                "q1_bloch_x": float(bloch["q1"][0]),
                "q1_bloch_y": float(bloch["q1"][1]),
                "q1_bloch_z": float(bloch["q1"][2]),
            })

            for key, qnode in qnodes.items():
                val0, val1 = qnode(float(noise0), float(noise1), weights)
                row[f"exp_{key}_q0"] = float(val0)
                row[f"exp_{key}_q1"] = float(val1)
            rows.append(row)

    df = pd.DataFrame(rows)

    summary: dict[str, dict[str, float]] = {}
    for col in df.columns:
        if col.startswith("exp_") or "bloch" in col:
            summary[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
            }
    return df, summary


def save_plots(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("2-Qubit debug: single-qubit expectation ranges", fontsize=16, fontweight="bold")

    plot_specs = [
        ("q0_bloch_x", "q0_bloch_y", "Q0: X vs Y"),
        ("q0_bloch_z", None, "Q0: Z histogram"),
        ("q1_bloch_x", "q1_bloch_y", "Q1: X vs Y"),
        ("q1_bloch_z", None, "Q1: Z histogram"),
        ("exp_ZZ_q0", "exp_ZZ_q1", "Measured Z outputs"),
        ("exp_YY_q0", "exp_YY_q1", "Measured Y outputs"),
    ]

    for ax, (c1, c2, title) in zip(axes.flat, plot_specs):
        if c2 is None:
            ax.hist(df[c1], bins=30, density=True, alpha=0.8)
            ax.set_xlabel(c1)
        else:
            ax.scatter(df[c1], df[c2], s=10, alpha=0.4)
            ax.set_xlabel(c1)
            ax.set_ylabel(c2)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "two_qubit_debug_plots.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a 2-qubit toy circuit with the same Y-embedding idea as the generator. "
            "Scans noise in [0, 2pi] and reports state / Bloch / measurement ranges."
        )
    )
    parser.add_argument("--grid", type=int, default=25, help="Number of angles per axis for the scan.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for weight initialization.")
    parser.add_argument("--init-std", type=float, default=0.5, help="Std for the 2-qubit toy weights.")
    parser.add_argument("--noise0", type=float, default=1.0, help="Example noise angle for qubit 0.")
    parser.add_argument("--noise1", type=float, default=2.0, help="Example noise angle for qubit 1.")
    parser.add_argument("--output-dir", type=Path, default=Path("two_qubit_debug_output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dev = build_device()
    state_circuit, make_expval_qnode = make_circuits(dev)
    weights = sample_weights(args.seed, args.init_std)

    example_state = np.array(state_circuit(args.noise0, args.noise1, weights), dtype=complex)
    example_bloch = bloch_from_state(example_state)

    df, summary = scan_ranges(state_circuit, make_expval_qnode, weights, args.grid)
    df.to_csv(args.output_dir / "two_qubit_scan.csv", index=False)

    with open(args.output_dir / "two_qubit_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "init_std": args.init_std,
                "grid": args.grid,
                "example_noise": [args.noise0, args.noise1],
                "weights": weights.tolist(),
                "example_state_real_imag": [[float(z.real), float(z.imag)] for z in example_state],
                "example_bloch": {k: v.tolist() for k, v in example_bloch.items()},
                "ranges": summary,
            },
            f,
            indent=2,
        )

    save_plots(df, args.output_dir)

    print("=== 2-QUBIT DEBUG SUMMARY ===")
    print(f"weights shape: {weights.shape}")
    print("weights:")
    print(np.round(weights, 4))
    print(f"example noise: ({args.noise0:.4f}, {args.noise1:.4f})")
    print("example state amplitudes [real, imag]:")
    print(np.round([[z.real, z.imag] for z in example_state], 4))
    print("example Bloch vectors:")
    for q, vec in example_bloch.items():
        print(f"  {q}: {np.round(vec, 4)}")

    key_cols = [
        "q0_bloch_x", "q0_bloch_y", "q0_bloch_z",
        "q1_bloch_x", "q1_bloch_y", "q1_bloch_z",
        "exp_ZZ_q0", "exp_ZZ_q1", "exp_YY_q0", "exp_YY_q1",
    ]
    print("\nKey ranges over scan [0, 2pi] x [0, 2pi]:")
    for col in key_cols:
        stats = summary[col]
        print(
            f"  {col}: min={stats['min']:.4f}  max={stats['max']:.4f}  "
            f"mean={stats['mean']:.4f}  std={stats['std']:.4f}"
        )

    print(f"\nSaved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
