import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunParams:
    disc_lr: float
    gen_lr: float
    disc_steps_per_gen: int
    grad_clip_norm: float
    training_steps: int
    batch_size: int
    disc_warmup_steps: int
    latent_distribution: str
    label_real: float
    label_fake: float


@dataclass
class RunResult:
    run_id: str
    run_dir: str
    status: str
    score: float
    notes: str
    final_loss_d: float
    final_loss_g: float
    final_score_real: float
    final_score_fake_d: float
    final_score_fake_g: float
    mean_loss_d_last: float
    mean_loss_g_last: float
    std_loss_d_last: float
    std_loss_g_last: float
    max_grad_norm_d: float
    max_grad_norm_g: float
    mean_fake_g_near_zero_last: float
    mean_fake_g_near_one_last: float
    params: Dict[str, float | int | str]


def persist_results(search_root: Path, results: List[RunResult]) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    for r in results:
        rows.append(
            {
                "run_id": r.run_id,
                "run_dir": r.run_dir,
                "status": r.status,
                "score": r.score,
                "notes": r.notes,
                "final_loss_d": r.final_loss_d,
                "final_loss_g": r.final_loss_g,
                "final_score_real": r.final_score_real,
                "final_score_fake_d": r.final_score_fake_d,
                "final_score_fake_g": r.final_score_fake_g,
                "mean_loss_d_last": r.mean_loss_d_last,
                "mean_loss_g_last": r.mean_loss_g_last,
                "std_loss_d_last": r.std_loss_d_last,
                "std_loss_g_last": r.std_loss_g_last,
                "max_grad_norm_d": r.max_grad_norm_d,
                "max_grad_norm_g": r.max_grad_norm_g,
                "mean_fake_g_near_zero_last": r.mean_fake_g_near_zero_last,
                "mean_fake_g_near_one_last": r.mean_fake_g_near_one_last,
                **r.params,
            }
        )

    results_df = pd.DataFrame(rows)
    if len(results_df) > 0:
        results_df = results_df.sort_values(["status", "score"], ascending=[True, False])

    csv_path = search_root / "search_results.csv"
    json_path = search_root / "search_results.json"
    results_df.to_csv(csv_path, index=False)
    json_path.write_text(results_df.to_json(orient="records", indent=2), encoding="utf-8")
    return results_df


def _stable_tail(df: pd.DataFrame, frac: float = 0.2) -> pd.DataFrame:
    n = len(df)
    start = max(0, int(n * (1.0 - frac)))
    return df.iloc[start:].copy()


def evaluate_run(metrics_path: Path) -> Dict[str, float | str]:
    df = pd.read_csv(metrics_path)
    tail = _stable_tail(df, frac=0.2)

    final = df.iloc[-1]
    final_loss_d = float(final["loss_d"])
    final_loss_g = float(final["loss_g"])
    final_score_real = float(final["score_real"])
    final_score_fake_d = float(final["score_fake_d"])
    final_score_fake_g = float(final["score_fake_g"])

    mean_loss_d_last = float(tail["loss_d"].mean())
    mean_loss_g_last = float(tail["loss_g"].mean())
    std_loss_d_last = float(tail["loss_d"].std(ddof=0))
    std_loss_g_last = float(tail["loss_g"].std(ddof=0))

    max_grad_norm_d = float(df["grad_norm_d"].max())
    max_grad_norm_g = float(df["grad_norm_g"].replace([np.inf, -np.inf], np.nan).fillna(0.0).max())

    mean_fake_g_near_zero_last = float(tail["fake_g_near_zero_frac"].mean())
    mean_fake_g_near_one_last = float(tail["fake_g_near_one_frac"].mean())

    # Heuristic score: favor balanced, non-exploding adversarial equilibrium.
    score = 0.0
    score -= abs(mean_loss_d_last - 0.69)
    score -= abs(mean_loss_g_last - 0.69)
    score -= 0.7 * abs(final_score_real - 0.5)
    score -= 0.7 * abs(final_score_fake_d - 0.5)
    score -= 0.4 * abs(final_score_fake_g - 0.5)
    score -= 0.5 * std_loss_d_last
    score -= 0.5 * std_loss_g_last

    if max_grad_norm_d > 20.0:
        score -= 1.0
    if max_grad_norm_g > 20.0:
        score -= 1.0
    if max_grad_norm_g < 1e-6:
        score -= 2.0
    if mean_fake_g_near_zero_last > 0.8 or mean_fake_g_near_one_last > 0.8:
        score -= 1.0

    if np.isnan(final_loss_g):
        score -= 5.0

    notes: List[str] = []
    if max_grad_norm_d > 20.0 or max_grad_norm_g > 20.0:
        notes.append("possible-gradient-explosion")
    if max_grad_norm_g < 1e-6:
        notes.append("generator-stalled")
    if mean_fake_g_near_zero_last > 0.8:
        notes.append("generator-saturated-near-zero")
    if mean_fake_g_near_one_last > 0.8:
        notes.append("generator-saturated-near-one")
    if abs(final_score_real - final_score_fake_d) < 0.05:
        notes.append("balanced-discriminator")

    return {
        "score": float(score),
        "notes": ",".join(notes) if notes else "ok",
        "final_loss_d": final_loss_d,
        "final_loss_g": final_loss_g,
        "final_score_real": final_score_real,
        "final_score_fake_d": final_score_fake_d,
        "final_score_fake_g": final_score_fake_g,
        "mean_loss_d_last": mean_loss_d_last,
        "mean_loss_g_last": mean_loss_g_last,
        "std_loss_d_last": std_loss_d_last,
        "std_loss_g_last": std_loss_g_last,
        "max_grad_norm_d": max_grad_norm_d,
        "max_grad_norm_g": max_grad_norm_g,
        "mean_fake_g_near_zero_last": mean_fake_g_near_zero_last,
        "mean_fake_g_near_one_last": mean_fake_g_near_one_last,
    }


def ratio_ok(disc_lr: float, gen_lr: float) -> bool:
    ratio = disc_lr / gen_lr
    return 0.5 <= ratio <= 2.0


def coarse_candidates() -> List[RunParams]:
    disc_lrs = [5e-4, 1e-3, 2e-3]
    gen_lrs = [5e-4, 1e-3, 2e-3]
    d_steps = [1, 2]
    clips = [0.5, 1.0]

    combos: List[RunParams] = []
    for dlr in disc_lrs:
        for glr in gen_lrs:
            if not ratio_ok(dlr, glr):
                continue
            for ds in d_steps:
                for clip in clips:
                    combos.append(
                        RunParams(
                            disc_lr=dlr,
                            gen_lr=glr,
                            disc_steps_per_gen=ds,
                            grad_clip_norm=clip,
                            training_steps=300,
                            batch_size=16,
                            disc_warmup_steps=0,
                            latent_distribution="uniform",
                            label_real=1.0,
                            label_fake=0.0,
                        )
                    )

    # Prioritize balanced settings first.
    combos.sort(
        key=lambda p: (
            abs((p.disc_lr / p.gen_lr) - 1.0),
            abs(p.grad_clip_norm - 1.0),
            abs(p.disc_steps_per_gen - 1),
        )
    )
    return combos


def refine_candidates(best_params: List[RunParams], already_used: set[RunParams]) -> List[RunParams]:
    refined: List[RunParams] = []
    scales = [0.5, 1.0, 2.0]
    clip_values = [0.5, 1.0, 2.0]

    for bp in best_params:
        for sd in scales:
            for sg in scales:
                dlr = min(max(bp.disc_lr * sd, 2.5e-4), 4e-3)
                glr = min(max(bp.gen_lr * sg, 2.5e-4), 4e-3)
                if not ratio_ok(dlr, glr):
                    continue
                for clip in clip_values:
                    candidate = RunParams(
                        disc_lr=round(dlr, 7),
                        gen_lr=round(glr, 7),
                        disc_steps_per_gen=bp.disc_steps_per_gen,
                        grad_clip_norm=clip,
                        training_steps=400,
                        batch_size=bp.batch_size,
                        disc_warmup_steps=bp.disc_warmup_steps,
                        latent_distribution=bp.latent_distribution,
                        label_real=bp.label_real,
                        label_fake=bp.label_fake,
                    )
                    if candidate in already_used:
                        continue
                    refined.append(candidate)

    refined.sort(
        key=lambda p: (
            abs((p.disc_lr / p.gen_lr) - 1.0),
            abs(p.grad_clip_norm - 1.0),
        )
    )
    return refined


def run_training(search_root: Path, idx: int, params: RunParams) -> RunResult:
    run_id = f"run_{idx:03d}"
    runs_root = search_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CGAN_SEED"] = str(42 + idx)
    env["CGAN_BATCH_SIZE"] = str(params.batch_size)
    env["CGAN_TRAINING_STEPS"] = str(params.training_steps)
    env["CGAN_DISC_STEPS_PER_GEN"] = str(params.disc_steps_per_gen)
    env["CGAN_DISC_LEARNING_RATE"] = str(params.disc_lr)
    env["CGAN_GEN_LEARNING_RATE"] = str(params.gen_lr)
    env["CGAN_DISC_WARMUP_STEPS"] = str(params.disc_warmup_steps)
    env["CGAN_GRAD_CLIP_NORM"] = str(params.grad_clip_norm)
    env["CGAN_LABEL_REAL"] = str(params.label_real)
    env["CGAN_LABEL_FAKE"] = str(params.label_fake)
    env["CGAN_LOSS_TYPE"] = "bce"
    env["CGAN_LATENT_DISTRIBUTION"] = params.latent_distribution
    env["CGAN_RUN_NAME"] = run_id
    env["CGAN_LOG_ROOT"] = str(runs_root)

    script_path = Path(__file__).parent / "training_cgan_comparable.py"
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=Path(__file__).parent,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    matched_runs = sorted(runs_root.glob(f"*_{run_id}"))
    if not matched_runs:
        return RunResult(
            run_id=run_id,
            run_dir="",
            status="failed",
            score=-999.0,
            notes=f"run-dir-not-found | rc={proc.returncode}",
            final_loss_d=np.nan,
            final_loss_g=np.nan,
            final_score_real=np.nan,
            final_score_fake_d=np.nan,
            final_score_fake_g=np.nan,
            mean_loss_d_last=np.nan,
            mean_loss_g_last=np.nan,
            std_loss_d_last=np.nan,
            std_loss_g_last=np.nan,
            max_grad_norm_d=np.nan,
            max_grad_norm_g=np.nan,
            mean_fake_g_near_zero_last=np.nan,
            mean_fake_g_near_one_last=np.nan,
            params=asdict(params),
        )

    run_dir = matched_runs[-1]
    metrics_path = run_dir / "metrics.csv"

    if proc.returncode != 0 or not metrics_path.exists():
        err_note = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown-error"
        return RunResult(
            run_id=run_id,
            run_dir=str(run_dir),
            status="failed",
            score=-999.0,
            notes=f"train-failed | {err_note}",
            final_loss_d=np.nan,
            final_loss_g=np.nan,
            final_score_real=np.nan,
            final_score_fake_d=np.nan,
            final_score_fake_g=np.nan,
            mean_loss_d_last=np.nan,
            mean_loss_g_last=np.nan,
            std_loss_d_last=np.nan,
            std_loss_g_last=np.nan,
            max_grad_norm_d=np.nan,
            max_grad_norm_g=np.nan,
            mean_fake_g_near_zero_last=np.nan,
            mean_fake_g_near_one_last=np.nan,
            params=asdict(params),
        )

    eval_dict = evaluate_run(metrics_path)
    return RunResult(
        run_id=run_id,
        run_dir=str(run_dir),
        status="ok",
        score=float(eval_dict["score"]),
        notes=str(eval_dict["notes"]),
        final_loss_d=float(eval_dict["final_loss_d"]),
        final_loss_g=float(eval_dict["final_loss_g"]),
        final_score_real=float(eval_dict["final_score_real"]),
        final_score_fake_d=float(eval_dict["final_score_fake_d"]),
        final_score_fake_g=float(eval_dict["final_score_fake_g"]),
        mean_loss_d_last=float(eval_dict["mean_loss_d_last"]),
        mean_loss_g_last=float(eval_dict["mean_loss_g_last"]),
        std_loss_d_last=float(eval_dict["std_loss_d_last"]),
        std_loss_g_last=float(eval_dict["std_loss_g_last"]),
        max_grad_norm_d=float(eval_dict["max_grad_norm_d"]),
        max_grad_norm_g=float(eval_dict["max_grad_norm_g"]),
        mean_fake_g_near_zero_last=float(eval_dict["mean_fake_g_near_zero_last"]),
        mean_fake_g_near_one_last=float(eval_dict["mean_fake_g_near_one_last"]),
        params=asdict(params),
    )


def write_report(search_root: Path, results_df: pd.DataFrame, top_n: int = 5) -> None:
    report_path = search_root / "REPORT.md"
    top_df = results_df[results_df["status"] == "ok"].sort_values("score", ascending=False).head(top_n)

    lines: List[str] = []
    lines.append("# CGAN Hyperparameter Search Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Total runs: {len(results_df)}")
    lines.append(f"Successful runs: {(results_df['status'] == 'ok').sum()}")
    lines.append("")
    lines.append("## Method")
    lines.append("- Phase 1: coarse search over balanced learning rates, discriminator steps, and gradient clipping.")
    lines.append("- Phase 2: local refinement around best coarse candidates.")
    lines.append("- Scoring objective: balanced adversarial equilibrium, low oscillation, and no gradient pathologies.")
    lines.append("")

    if len(top_df) == 0:
        lines.append("## Result")
        lines.append("No successful runs were found.")
    else:
        lines.append("## Top Configurations")
        for _, row in top_df.iterrows():
            lines.append(
                "- "
                f"{row['run_id']}: score={row['score']:.4f}, "
                f"d_lr={row['disc_lr']}, g_lr={row['gen_lr']}, "
                f"d_steps={row['disc_steps_per_gen']}, clip={row['grad_clip_norm']}, "
                f"loss_d={row['mean_loss_d_last']:.4f}, loss_g={row['mean_loss_g_last']:.4f}, "
                f"score_real={row['final_score_real']:.4f}, score_fake_d={row['final_score_fake_d']:.4f}, "
                f"notes={row['notes']}"
            )

        best = top_df.iloc[0]
        lines.append("")
        lines.append("## Recommended Next Default")
        lines.append(f"- disc_learning_rate: {best['disc_lr']}")
        lines.append(f"- gen_learning_rate: {best['gen_lr']}")
        lines.append(f"- disc_steps_per_gen: {int(best['disc_steps_per_gen'])}")
        lines.append(f"- grad_clip_norm: {best['grad_clip_norm']}")
        lines.append(f"- training_steps (search context): {int(best['training_steps'])}")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    search_root = Path("logs") / f"hparam_search_cgan_{timestamp}"
    search_root.mkdir(parents=True, exist_ok=True)

    coarse_limit = int(os.getenv("CGAN_SEARCH_COARSE_RUNS", "6"))
    refine_limit = int(os.getenv("CGAN_SEARCH_REFINE_RUNS", "4"))

    coarse = coarse_candidates()[:coarse_limit]
    used = set(coarse)
    results: List[RunResult] = []

    print(f"[SEARCH] Output root: {search_root}")
    print(f"[SEARCH] Coarse runs: {len(coarse)}")

    idx = 1
    for params in coarse:
        print(f"[SEARCH] Running {idx}/{len(coarse)} (coarse): {params}")
        result = run_training(search_root, idx, params)
        print(f"[SEARCH] -> {result.status}, score={result.score:.4f}, notes={result.notes}")
        results.append(result)
        persist_results(search_root, results)
        idx += 1

    ok_coarse = [r for r in results if r.status == "ok"]
    ok_coarse.sort(key=lambda r: r.score, reverse=True)
    top_seed = [RunParams(**r.params) for r in ok_coarse[:3]]

    refine = refine_candidates(top_seed, used)[:refine_limit]
    print(f"[SEARCH] Refinement runs: {len(refine)}")
    for params in refine:
        used.add(params)
        print(f"[SEARCH] Running {idx}/{len(coarse) + len(refine)} (refine): {params}")
        result = run_training(search_root, idx, params)
        print(f"[SEARCH] -> {result.status}, score={result.score:.4f}, notes={result.notes}")
        results.append(result)
        persist_results(search_root, results)
        idx += 1

    results_df = persist_results(search_root, results)
    write_report(search_root, results_df)

    print(f"[SEARCH] Results CSV: {search_root / 'search_results.csv'}")
    print(f"[SEARCH] Results JSON: {search_root / 'search_results.json'}")
    print(f"[SEARCH] Report: {search_root / 'REPORT.md'}")


if __name__ == "__main__":
    main()
