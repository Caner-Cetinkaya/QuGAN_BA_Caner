"""
run_hybrid_angle_experiments.py

Starts the four predefined Hybrid-GAN experiments sequentially.

Usage from your src folder:
    python .\run_hybrid_angle_experiments.py

Expected files in the same folder:
    training_hybrid_qgen_cdisc_classicplot.py
    valid_tuples.csv
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------

TRAIN_SCRIPT = Path("training_hybrid_qgen_cdisc_classicplot.py")
VALID_TUPLES = Path("valid_tuples.csv")

BASE_OUTPUT_DIR = Path("logs") / "hybrid_angle_experiments3"

# Fixed hyperparameters for all runs
G_LR = "0.001"
D_LR = "0.005"
LAYERS = "2"
LATENT_DISTRIBUTION = "angle"


EXPERIMENTS = [
    {
        "name": "exp01_angle_seed1_bs1_steps20000",
        "steps": 20000,
        "seed": 1,
        "batch_size": 1,
    },
    {
        "name": "exp02_angle_seed1_bs10_steps2000",
        "steps": 2000,
        "seed": 1,
        "batch_size": 10,
    },
    {
        "name": "exp03_angle_seed2_bs1_steps20000",
        "steps": 20000,
        "seed": 2,
        "batch_size": 1,
    },
    {
        "name": "exp04_angle_seed2_bs10_steps2000",
        "steps": 2000,
        "seed": 2,
        "batch_size": 10,
    },
]


def check_required_files() -> None:
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(
            f"Training script not found: {TRAIN_SCRIPT.resolve()}\n"
            "Put this runner script into the same src folder as "
            "training_hybrid_qgen_cdisc_classicplot.py."
        )

    if not VALID_TUPLES.exists():
        raise FileNotFoundError(
            f"Valid tuples file not found: {VALID_TUPLES.resolve()}\n"
            "Put valid_tuples.csv into the same src folder or update VALID_TUPLES."
        )


def run_experiment(exp: dict) -> None:
    name = exp["name"]
    steps = exp["steps"]
    seed = exp["seed"]
    batch_size = exp["batch_size"]

    output_dir = BASE_OUTPUT_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "terminal_output.log"

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--steps",
        str(steps),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
        "--g-lr",
        G_LR,
        "--d-lr",
        D_LR,
        "--layers",
        LAYERS,
        "--latent-distribution",
        LATENT_DISTRIBUTION,
        "--valid-tuples",
        str(VALID_TUPLES),
        "--output-dir",
        str(output_dir),
    ]

    header = (
        "\n"
        "============================================================\n"
        f"Starting {name}\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Steps: {steps} | Seed: {seed} | Batch size: {batch_size} | "
        f"latent={LATENT_DISTRIBUTION}\n"
        f"Output dir: {output_dir.resolve()}\n"
        "Command:\n"
        + " ".join(command)
        + "\n"
        "============================================================\n"
    )

    print(header)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(header)
        f.flush()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            f.write(line)
            f.flush()

        return_code = process.wait()

        footer = (
            "\n"
            "============================================================\n"
            f"Finished {name} with exit code {return_code}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "============================================================\n"
        )

        print(footer)
        f.write(footer)

    if return_code != 0:
        raise RuntimeError(f"Experiment {name} failed with exit code {return_code}.")


def main() -> None:
    check_required_files()
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Starting Hybrid-GAN angle experiment batch.")
    print(f"[INFO] Base output dir: {BASE_OUTPUT_DIR.resolve()}")

    for exp in EXPERIMENTS:
        run_experiment(exp)

    print("\n[INFO] All experiments finished.")
    print(f"[INFO] Results saved in: {BASE_OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
