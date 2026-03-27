#!/usr/bin/env python3
"""
Robust debug script for QuGAN batch/broadcasting issues.

What this version fixes compared to the first debug script:
- automatically detects whether the current discriminator returns 1 readout or 6 readouts
- keeps shared target vectors outside optional test branches
- can emulate the legacy Q0-only score even when the discriminator already returns 6 readouts
- cleanly tests multi-readout behavior by patching a fresh discriminator when needed

Goals:
1) Compare batched vs single-sample execution for the generator.
2) Compare batched vs single-sample execution for the discriminator.
3) Compare batched vs loop-based gradients for discriminator losses.
4) Expose readout statistics to see whether one qubit or all qubits carry signal.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from discriminator import QDiscriminator
from generator import QGenerator


# -----------------------------
# Synthetic data helpers
# -----------------------------

def sample_real_like_edges(batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate geometrically consistent 6-edge samples from 4 random 2D points.

    Output shape: (B, 6), normalized to [0, 1] per sample by dividing by that
    sample's maximum edge length.
    """
    batch = []
    for _ in range(batch_size):
        pts = rng.uniform(-1.0, 1.0, size=(4, 2))
        a, b, c, d = pts
        edges = np.array(
            [
                np.linalg.norm(a - b),
                np.linalg.norm(b - c),
                np.linalg.norm(c - d),
                np.linalg.norm(d - a),
                np.linalg.norm(a - c),
                np.linalg.norm(b - d),
            ],
            dtype=float,
        )
        scale = max(float(edges.max()), 1e-12)
        batch.append(np.clip(edges / scale, 0.0, 1.0))
    return np.asarray(batch, dtype=float)


# -----------------------------
# Debug models and helpers
# -----------------------------

def patch_discriminator_multireadout(disc: QDiscriminator) -> None:
    """Monkey-patch the discriminator qnode to return all 6 Z readouts."""

    @qml.qnode(disc.dev, interface="autograd", diff_method="backprop")
    def circuit(edge_weights, weights):
        qml.AngleEmbedding(edge_weights * np.pi, wires=range(disc.n_qubits), rotation="Y")
        qml.AngleEmbedding(edge_weights * np.pi, wires=range(disc.n_qubits), rotation="Z")

        for layer in range(disc.n_layer):
            for qubit in range(disc.n_qubits):
                qml.RX(weights[layer, qubit, 0], wires=qubit)
            for qubit in range(disc.n_qubits):
                qml.RY(weights[layer, qubit, 1], wires=qubit)
            for qubit in range(disc.n_qubits):
                qml.RZ(weights[layer, qubit, 2], wires=qubit)
            for qubit in range(disc.n_qubits):
                qml.CNOT(wires=[qubit, (qubit + 1) % disc.n_qubits])
            for qubit in range(disc.n_qubits):
                qml.CNOT(wires=[qubit, (qubit + 2) % disc.n_qubits])

        return [qml.expval(qml.PauliZ(i)) for i in range(disc.n_qubits)]

    disc.circuit = circuit


@dataclass
class CompareResult:
    name: str
    batch_shape: tuple
    loop_shape: tuple
    max_abs_diff: float
    mean_abs_diff: float
    batch_preview: np.ndarray
    loop_preview: np.ndarray



def _to_numpy(x) -> np.ndarray:
    return np.asarray(x, dtype=float)



def _stack_readouts(z, expected_batch_size: int | None = None, expected_n_qubits: int = 6) -> pnp.ndarray:
    """Normalize QNode output shapes to (B, Q) or (1, Q)."""
    if isinstance(z, (list, tuple)):
        elems = [pnp.array(v) for v in z]
        if len(elems) == 0:
            raise ValueError("empty readout list")

        # Single sample + multi-readout often comes back as [scalar, scalar, ...]
        if all(e.ndim == 0 for e in elems):
            return pnp.stack(elems, axis=0)[pnp.newaxis, :]   # (Q,) -> (1,Q)

        # Batched multi-readout usually comes back as [array(B,), array(B,), ...]
        if all(e.ndim == 1 for e in elems):
            return pnp.stack(elems, axis=1)                   # (B,Q)

        # Fallback for rarer nested forms: stack on a new axis, then normalize below.
        z_arr = pnp.stack(elems, axis=0)
    else:
        z_arr = pnp.array(z)

    if z_arr.ndim == 0:
        return z_arr[pnp.newaxis, pnp.newaxis]
    if z_arr.ndim == 1:
        if expected_batch_size is not None and z_arr.shape[0] == expected_batch_size:
            return z_arr[:, pnp.newaxis]  # (B,) -> (B,1)
        return z_arr[pnp.newaxis, :]      # (Q,) -> (1,Q)
    if z_arr.ndim == 2 and z_arr.shape[0] == expected_n_qubits:
        return z_arr.T
    return z_arr



def detect_disc_readout_mode(disc: QDiscriminator) -> str:
    """Return 'single' for one readout and 'multi' for n_qubits readouts."""
    probe = np.linspace(0.1, 0.9, disc.n_qubits, dtype=float)
    z = disc.circuit(pnp.array(probe, dtype=float), pnp.array(disc.weights, dtype=float))
    z_arr = _stack_readouts(z, expected_batch_size=None, expected_n_qubits=disc.n_qubits)
    flat = z_arr.reshape(-1)

    if flat.size == 1:
        return "single"
    if flat.size == disc.n_qubits:
        return "multi"
    return f"unexpected:{tuple(z_arr.shape)}"



def disc_probs_q0_batch(disc: QDiscriminator, weights, edges_batch: np.ndarray) -> pnp.ndarray:
    """Legacy score path: use Q0 if multi-readout is present, else the single scalar."""
    z = disc.circuit(pnp.array(edges_batch, dtype=float), weights)
    z_arr = _stack_readouts(z, expected_batch_size=len(edges_batch), expected_n_qubits=disc.n_qubits)

    if z_arr.shape[1] == 1:
        z0 = z_arr[:, 0]
    elif z_arr.shape[1] == disc.n_qubits:
        z0 = z_arr[:, 0]
    else:
        raise ValueError(f"unexpected discriminator readout shape: {tuple(z_arr.shape)}")

    return 0.5 * (z0 + 1.0)



def disc_probs_q0_loop(disc: QDiscriminator, weights, edges_batch: np.ndarray) -> pnp.ndarray:
    out = []
    for sample in edges_batch:
        z = disc.circuit(pnp.array(sample, dtype=float), weights)
        z_arr = _stack_readouts(z, expected_batch_size=None, expected_n_qubits=disc.n_qubits).reshape(-1)
        if z_arr.size not in (1, disc.n_qubits):
            raise ValueError(f"unexpected discriminator readout size in loop: {z_arr.size}")
        z0 = z_arr[0]
        out.append(0.5 * (z0 + 1.0))
    return pnp.stack(out)



def disc_readouts_multi_batch(disc: QDiscriminator, weights, edges_batch: np.ndarray) -> pnp.ndarray:
    z = disc.circuit(pnp.array(edges_batch, dtype=float), weights)
    z_arr = _stack_readouts(z, expected_batch_size=len(edges_batch), expected_n_qubits=disc.n_qubits)
    if z_arr.shape[1] != disc.n_qubits:
        raise ValueError(f"multi-readout path expected {disc.n_qubits} columns, got {tuple(z_arr.shape)}")
    return z_arr



def disc_readouts_multi_loop(disc: QDiscriminator, weights, edges_batch: np.ndarray) -> pnp.ndarray:
    rows = []
    for sample in edges_batch:
        z = disc.circuit(pnp.array(sample, dtype=float), weights)
        z_arr = _stack_readouts(z, expected_batch_size=None, expected_n_qubits=disc.n_qubits).reshape(-1)
        if z_arr.size != disc.n_qubits:
            raise ValueError(f"multi-readout loop expected {disc.n_qubits} values, got {z_arr.size}")
        rows.append(z_arr)
    return pnp.stack(rows, axis=0)



def generator_batch(gen: QGenerator, weights, noise_batch: np.ndarray) -> pnp.ndarray:
    return gen.batch_forward(noise_batch, weights=weights)



def generator_loop(gen: QGenerator, weights, noise_batch: np.ndarray) -> pnp.ndarray:
    rows = []
    old_weights = gen.weights
    try:
        gen.weights = weights
        for sample in noise_batch:
            rows.append(gen.forward(np.asarray(sample, dtype=float)))
    finally:
        gen.weights = old_weights
    return pnp.stack(rows, axis=0)



def compare_arrays(name: str, batch_arr, loop_arr, preview_rows: int = 3) -> CompareResult:
    batch_np = _to_numpy(batch_arr)
    loop_np = _to_numpy(loop_arr)
    diff = np.abs(batch_np - loop_np)
    return CompareResult(
        name=name,
        batch_shape=batch_np.shape,
        loop_shape=loop_np.shape,
        max_abs_diff=float(diff.max()) if diff.size else 0.0,
        mean_abs_diff=float(diff.mean()) if diff.size else 0.0,
        batch_preview=batch_np[:preview_rows],
        loop_preview=loop_np[:preview_rows],
    )



def print_compare(result: CompareResult) -> None:
    print(f"\n[{result.name}]")
    print(f"  batch shape: {result.batch_shape}")
    print(f"  loop  shape: {result.loop_shape}")
    print(f"  max |Δ|:     {result.max_abs_diff:.12e}")
    print(f"  mean |Δ|:    {result.mean_abs_diff:.12e}")
    print("  batch preview:")
    print(result.batch_preview)
    print("  loop preview:")
    print(result.loop_preview)



def mse_loss(preds: pnp.ndarray, targets: pnp.ndarray) -> pnp.ndarray:
    return pnp.mean((preds - targets) ** 2)



def gradient_compare_q0_path(disc: QDiscriminator, real_batch: np.ndarray, fake_batch: np.ndarray) -> None:
    print("\n=== Gradient compare: discriminator Q0/legacy score path ===")
    weights0 = pnp.array(disc.weights, dtype=float, requires_grad=True)
    targets_real = pnp.ones(len(real_batch), dtype=float)
    targets_fake = pnp.zeros(len(fake_batch), dtype=float)

    def loss_batch(weights):
        real_preds = disc_probs_q0_batch(disc, weights, real_batch)
        fake_preds = disc_probs_q0_batch(disc, weights, fake_batch)
        return 0.5 * (mse_loss(real_preds, targets_real) + mse_loss(fake_preds, targets_fake))

    def loss_loop(weights):
        real_preds = disc_probs_q0_loop(disc, weights, real_batch)
        fake_preds = disc_probs_q0_loop(disc, weights, fake_batch)
        return 0.5 * (mse_loss(real_preds, targets_real) + mse_loss(fake_preds, targets_fake))

    grad_batch = qml.grad(loss_batch)(weights0)
    grad_loop = qml.grad(loss_loop)(weights0)
    cmp = compare_arrays("disc grad (Q0 path) batch vs loop", grad_batch, grad_loop)
    print_compare(cmp)
    print(f"  ||grad_batch||: {float(pnp.linalg.norm(grad_batch)):.12e}")
    print(f"  ||grad_loop ||: {float(pnp.linalg.norm(grad_loop)):.12e}")



def gradient_compare_multi_readout(disc: QDiscriminator, real_batch: np.ndarray, fake_batch: np.ndarray) -> None:
    print("\n=== Gradient compare: discriminator multi-readout mean-head ===")
    weights0 = pnp.array(disc.weights, dtype=float, requires_grad=True)
    targets_real = pnp.ones(len(real_batch), dtype=float)
    targets_fake = pnp.zeros(len(fake_batch), dtype=float)

    def probs_from_readouts_batch(weights, edges_batch):
        z = disc_readouts_multi_batch(disc, weights, edges_batch)
        return 0.5 * (pnp.mean(z, axis=1) + 1.0)

    def probs_from_readouts_loop(weights, edges_batch):
        z = disc_readouts_multi_loop(disc, weights, edges_batch)
        return 0.5 * (pnp.mean(z, axis=1) + 1.0)

    def loss_batch(weights):
        real_preds = probs_from_readouts_batch(weights, real_batch)
        fake_preds = probs_from_readouts_batch(weights, fake_batch)
        return 0.5 * (mse_loss(real_preds, targets_real) + mse_loss(fake_preds, targets_fake))

    def loss_loop(weights):
        real_preds = probs_from_readouts_loop(weights, real_batch)
        fake_preds = probs_from_readouts_loop(weights, fake_batch)
        return 0.5 * (mse_loss(real_preds, targets_real) + mse_loss(fake_preds, targets_fake))

    grad_batch = qml.grad(loss_batch)(weights0)
    grad_loop = qml.grad(loss_loop)(weights0)
    cmp = compare_arrays("disc grad (multi-readout) batch vs loop", grad_batch, grad_loop)
    print_compare(cmp)
    print(f"  ||grad_batch||: {float(pnp.linalg.norm(grad_batch)):.12e}")
    print(f"  ||grad_loop ||: {float(pnp.linalg.norm(grad_loop)):.12e}")



def print_readout_stats(title: str, z_real: np.ndarray, z_fake: np.ndarray) -> None:
    z_real = _to_numpy(z_real)
    z_fake = _to_numpy(z_fake)
    delta = z_real.mean(axis=0) - z_fake.mean(axis=0)
    print(f"\n=== {title} ===")
    print("Per-qubit mean(real):", np.round(z_real.mean(axis=0), 6))
    print("Per-qubit std (real):", np.round(z_real.std(axis=0), 6))
    print("Per-qubit mean(fake):", np.round(z_fake.mean(axis=0), 6))
    print("Per-qubit std (fake):", np.round(z_fake.std(axis=0), 6))
    print("Per-qubit Δmean    :", np.round(delta, 6))
    print("Best |Δmean| qubit :", int(np.argmax(np.abs(delta))))



def try_step_sanity(loss_fn: Callable, weights0: pnp.ndarray, lr: float = 1e-2) -> None:
    loss_before = float(loss_fn(weights0))
    grad = qml.grad(loss_fn)(weights0)
    weights1 = weights0 - lr * grad
    loss_after = float(loss_fn(weights1))
    print(f"  tiny-step sanity: loss {loss_before:.8f} -> {loss_after:.8f} (lr={lr})")



def main() -> None:
    parser = argparse.ArgumentParser(description="Debug QuGAN broadcasting and gradient consistency")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gen-init-std", type=float, default=None, help="Optional reinit std for generator weights")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print("=" * 72)
    print("QuGAN DEBUG v2: batch vs loop / broadcasting / gradient sanity")
    print("=" * 72)
    print(f"seed={args.seed}  layers={args.layers}  batch_size={args.batch_size}")
    print(f"cwd={Path.cwd()}")

    # Initialize models from the current local files.
    disc_current = QDiscriminator(n_layer=args.layers, seed=args.seed)
    gen = QGenerator(n_layer=args.layers, seed=args.seed)

    if args.gen_init_std is not None:
        gen.weights = gen.rng.normal(0.0, args.gen_init_std, size=gen.weights.shape)
        print(f"[debug] Generator weights reinitialized with std={args.gen_init_std}")

    disc_mode = detect_disc_readout_mode(disc_current)
    print(f"[debug] detected current discriminator readout mode: {disc_mode}")

    # Synthetic but geometry-consistent real samples.
    real_batch = sample_real_like_edges(args.batch_size, rng)
    noise_batch = rng.uniform(0.0, 1.0, size=(args.batch_size, 6))
    fake_batch = _to_numpy(gen.batch_forward(noise_batch))

    print("\nInput batch shapes:")
    print(f"  real_batch: {real_batch.shape}")
    print(f"  noise_batch: {noise_batch.shape}")
    print(f"  fake_batch: {fake_batch.shape}")

    targets_real = pnp.ones(len(real_batch), dtype=float)
    targets_fake = pnp.zeros(len(fake_batch), dtype=float)

    # 1) Generator batching sanity.
    gen_cmp = compare_arrays(
        "generator batch_forward vs forward-loop",
        generator_batch(gen, gen.weights, noise_batch),
        generator_loop(gen, gen.weights, noise_batch),
    )
    print_compare(gen_cmp)

    # 2) Always test the legacy/Q0 score path. This works for both single and multi-readout discs.
    disc_q0_prob_cmp_real = compare_arrays(
        "disc probs Q0/legacy path batch vs loop (real)",
        disc_probs_q0_batch(disc_current, disc_current.weights, real_batch),
        disc_probs_q0_loop(disc_current, disc_current.weights, real_batch),
    )
    print_compare(disc_q0_prob_cmp_real)

    disc_q0_prob_cmp_fake = compare_arrays(
        "disc probs Q0/legacy path batch vs loop (fake)",
        disc_probs_q0_batch(disc_current, disc_current.weights, fake_batch),
        disc_probs_q0_loop(disc_current, disc_current.weights, fake_batch),
    )
    print_compare(disc_q0_prob_cmp_fake)

    gradient_compare_q0_path(disc_current, real_batch, fake_batch)

    weights0 = pnp.array(disc_current.weights, dtype=float, requires_grad=True)

    def q0_loss(weights):
        real_preds = disc_probs_q0_batch(disc_current, weights, real_batch)
        fake_preds = disc_probs_q0_batch(disc_current, weights, fake_batch)
        return 0.5 * (mse_loss(real_preds, targets_real) + mse_loss(fake_preds, targets_fake))

    print("\n=== Tiny-step sanity: current discriminator Q0/legacy path ===")
    try_step_sanity(q0_loss, weights0)

    # 3) Multi-readout tests.
    #    If the current discriminator is already multi-readout, use it directly.
    #    Otherwise patch a fresh discriminator.
    if disc_mode == "multi":
        disc_multi = disc_current
        print("\n[debug] current discriminator already has multi-readout; using it directly.")
    else:
        disc_multi = QDiscriminator(n_layer=args.layers, seed=args.seed)
        patch_discriminator_multireadout(disc_multi)
        print("\n[debug] patched a fresh discriminator to multi-readout mode.")

    z_real_batch = disc_readouts_multi_batch(disc_multi, disc_multi.weights, real_batch)
    z_real_loop = disc_readouts_multi_loop(disc_multi, disc_multi.weights, real_batch)
    z_fake_batch = disc_readouts_multi_batch(disc_multi, disc_multi.weights, fake_batch)
    z_fake_loop = disc_readouts_multi_loop(disc_multi, disc_multi.weights, fake_batch)

    print_compare(compare_arrays("disc readouts multi batch vs loop (real)", z_real_batch, z_real_loop))
    print_compare(compare_arrays("disc readouts multi batch vs loop (fake)", z_fake_batch, z_fake_loop))
    print_readout_stats("Multi-readout discriminator statistics", z_real_batch, z_fake_batch)

    probs_multi_batch_real = 0.5 * (pnp.mean(z_real_batch, axis=1) + 1.0)
    probs_multi_loop_real = 0.5 * (pnp.mean(z_real_loop, axis=1) + 1.0)
    print_compare(compare_arrays("disc probs multi-readout mean-head batch vs loop (real)", probs_multi_batch_real, probs_multi_loop_real))

    probs_multi_batch_fake = 0.5 * (pnp.mean(z_fake_batch, axis=1) + 1.0)
    probs_multi_loop_fake = 0.5 * (pnp.mean(z_fake_loop, axis=1) + 1.0)
    print_compare(compare_arrays("disc probs multi-readout mean-head batch vs loop (fake)", probs_multi_batch_fake, probs_multi_loop_fake))

    gradient_compare_multi_readout(disc_multi, real_batch, fake_batch)

    weights1 = pnp.array(disc_multi.weights, dtype=float, requires_grad=True)

    def multi_loss(weights):
        real_preds = 0.5 * (pnp.mean(disc_readouts_multi_batch(disc_multi, weights, real_batch), axis=1) + 1.0)
        fake_preds = 0.5 * (pnp.mean(disc_readouts_multi_batch(disc_multi, weights, fake_batch), axis=1) + 1.0)
        return 0.5 * (mse_loss(real_preds, targets_real) + mse_loss(fake_preds, targets_fake))

    print("\n=== Tiny-step sanity: multi-readout discriminator mean-head ===")
    try_step_sanity(multi_loss, weights1)

    print("\nDone. If any batch-vs-loop diff is noticeably > 1e-9 to 1e-7, that is suspicious.")
    print("If gradients differ strongly between batch and loop, the bug is very likely in batching/broadcasting.")


if __name__ == "__main__":
    main()
