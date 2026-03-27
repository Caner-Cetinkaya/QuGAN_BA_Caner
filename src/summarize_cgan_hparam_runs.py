from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class RunSummary:
    run_dir: str
    source: str
    disc_lr: float
    gen_lr: float
    disc_steps_per_gen: int
    grad_clip_norm: float
    training_steps: int
    score: float
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
    notes: str


def _tail(df: pd.DataFrame, frac: float = 0.2) -> pd.DataFrame:
    start = max(0, int(len(df) * (1.0 - frac)))
    return df.iloc[start:].copy()


def evaluate_metrics(metrics_path: Path) -> Dict[str, float | str]:
    df = pd.read_csv(metrics_path)
    tail = _tail(df)
    final = df.iloc[-1]

    mean_loss_d_last = float(tail["loss_d"].mean())
    mean_loss_g_last = float(tail["loss_g"].mean())
    std_loss_d_last = float(tail["loss_d"].std(ddof=0))
    std_loss_g_last = float(tail["loss_g"].std(ddof=0))
    max_grad_norm_d = float(df["grad_norm_d"].max())
    max_grad_norm_g = float(df["grad_norm_g"].max())

    score = 0.0
    score -= abs(mean_loss_d_last - 0.69)
    score -= abs(mean_loss_g_last - 0.69)
    score -= 0.7 * abs(float(final["score_real"]) - 0.5)
    score -= 0.7 * abs(float(final["score_fake_d"]) - 0.5)
    score -= 0.4 * abs(float(final["score_fake_g"]) - 0.5)
    score -= 0.5 * std_loss_d_last
    score -= 0.5 * std_loss_g_last
    if max_grad_norm_d > 20.0:
        score -= 1.0
    if max_grad_norm_g > 20.0:
        score -= 1.0
    if max_grad_norm_g < 1e-6:
        score -= 2.0

    notes: List[str] = []
    if max_grad_norm_d > 20.0 or max_grad_norm_g > 20.0:
        notes.append("possible-gradient-explosion")
    if max_grad_norm_g < 1e-6:
        notes.append("generator-stalled")
    if abs(float(final["score_real"]) - float(final["score_fake_d"])) < 0.05:
        notes.append("balanced-discriminator")

    return {
        "score": float(score),
        "final_loss_d": float(final["loss_d"]),
        "final_loss_g": float(final["loss_g"]),
        "final_score_real": float(final["score_real"]),
        "final_score_fake_d": float(final["score_fake_d"]),
        "final_score_fake_g": float(final["score_fake_g"]),
        "mean_loss_d_last": mean_loss_d_last,
        "mean_loss_g_last": mean_loss_g_last,
        "std_loss_d_last": std_loss_d_last,
        "std_loss_g_last": std_loss_g_last,
        "max_grad_norm_d": max_grad_norm_d,
        "max_grad_norm_g": max_grad_norm_g,
        "notes": ",".join(notes) if notes else "ok",
    }


def collect_runs(base: Path) -> List[RunSummary]:
    summaries: List[RunSummary] = []

    # Runs from partial grid-search logs.
    for cfg in base.glob("logs/hparam_search_cgan_*/runs/*/config.json"):
        run_dir = cfg.parent
        metrics = run_dir / "metrics.csv"
        if not metrics.exists():
            continue
        conf = json.loads(cfg.read_text(encoding="utf-8"))
        ev = evaluate_metrics(metrics)
        summaries.append(
            RunSummary(
                run_dir=str(run_dir),
                source="grid-search",
                disc_lr=float(conf.get("disc_learning_rate", np.nan)),
                gen_lr=float(conf.get("gen_learning_rate", np.nan)),
                disc_steps_per_gen=int(conf.get("disc_steps_per_gen", 1)),
                grad_clip_norm=float(conf.get("gradient_clip_norm", np.nan)),
                training_steps=int(conf.get("training_steps", len(pd.read_csv(metrics)))),
                score=float(ev["score"]),
                final_loss_d=float(ev["final_loss_d"]),
                final_loss_g=float(ev["final_loss_g"]),
                final_score_real=float(ev["final_score_real"]),
                final_score_fake_d=float(ev["final_score_fake_d"]),
                final_score_fake_g=float(ev["final_score_fake_g"]),
                mean_loss_d_last=float(ev["mean_loss_d_last"]),
                mean_loss_g_last=float(ev["mean_loss_g_last"]),
                std_loss_d_last=float(ev["std_loss_d_last"]),
                std_loss_g_last=float(ev["std_loss_g_last"]),
                max_grad_norm_d=float(ev["max_grad_norm_d"]),
                max_grad_norm_g=float(ev["max_grad_norm_g"]),
                notes=str(ev["notes"]),
            )
        )

    # Manually controlled runs.
    for cfg in base.glob("logs/manual_hparam_eval_20260323/*/config.json"):
        run_dir = cfg.parent
        metrics = run_dir / "metrics.csv"
        if not metrics.exists():
            continue
        conf = json.loads(cfg.read_text(encoding="utf-8"))
        ev = evaluate_metrics(metrics)
        summaries.append(
            RunSummary(
                run_dir=str(run_dir),
                source="manual",
                disc_lr=float(conf.get("disc_learning_rate", np.nan)),
                gen_lr=float(conf.get("gen_learning_rate", np.nan)),
                disc_steps_per_gen=int(conf.get("disc_steps_per_gen", 1)),
                grad_clip_norm=float(conf.get("gradient_clip_norm", np.nan)),
                training_steps=int(conf.get("training_steps", len(pd.read_csv(metrics)))),
                score=float(ev["score"]),
                final_loss_d=float(ev["final_loss_d"]),
                final_loss_g=float(ev["final_loss_g"]),
                final_score_real=float(ev["final_score_real"]),
                final_score_fake_d=float(ev["final_score_fake_d"]),
                final_score_fake_g=float(ev["final_score_fake_g"]),
                mean_loss_d_last=float(ev["mean_loss_d_last"]),
                mean_loss_g_last=float(ev["mean_loss_g_last"]),
                std_loss_d_last=float(ev["std_loss_d_last"]),
                std_loss_g_last=float(ev["std_loss_g_last"]),
                max_grad_norm_d=float(ev["max_grad_norm_d"]),
                max_grad_norm_g=float(ev["max_grad_norm_g"]),
                notes=str(ev["notes"]),
            )
        )

    summaries.sort(key=lambda r: r.score, reverse=True)
    return summaries


def main() -> None:
    root = Path(__file__).parent
    output_dir = root / "logs" / f"hparam_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(root)
    if not runs:
        raise RuntimeError("No completed hyperparameter runs found.")

    df = pd.DataFrame([r.__dict__ for r in runs])
    csv_path = output_dir / "combined_results.csv"
    json_path = output_dir / "combined_results.json"
    md_path = output_dir / "REPORT.md"
    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")

    best = runs[0]
    lines: List[str] = []
    lines.append("# CGAN Hyperparameter Test Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Evaluated completed runs: {len(runs)}")
    lines.append("")
    lines.append("## Ranking (best first)")
    for idx, r in enumerate(runs, start=1):
        lines.append(
            f"- {idx}. score={r.score:.4f}, d_lr={r.disc_lr}, g_lr={r.gen_lr}, "
            f"d_steps={r.disc_steps_per_gen}, clip={r.grad_clip_norm}, "
            f"loss_d={r.mean_loss_d_last:.4f}, loss_g={r.mean_loss_g_last:.4f}, "
            f"real={r.final_score_real:.4f}, fake_d={r.final_score_fake_d:.4f}, "
            f"fake_g={r.final_score_fake_g:.4f}, notes={r.notes}, source={r.source}"
        )

    lines.append("")
    lines.append("## Recommended Configuration")
    lines.append(f"- disc_learning_rate: {best.disc_lr}")
    lines.append(f"- gen_learning_rate: {best.gen_lr}")
    lines.append(f"- disc_steps_per_gen: {best.disc_steps_per_gen}")
    lines.append(f"- grad_clip_norm: {best.grad_clip_norm}")
    lines.append("- disc_warmup_steps: 0")
    lines.append("- latent_distribution: uniform")
    lines.append("- label_real / label_fake: 1.0 / 0.0")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- Ziel war ein balanciertes adversariales Spiel ohne Gradienten-Explosion und ohne Generator-Stillstand.")
    lines.append("- Die besten Runs zeigen moderate, stabile Verlaufswerte statt extremer Sättigung.")
    lines.append("- Für finale Entscheidungen sollte ein längerer Follow-up-Run (z.B. >= 1000 Schritte) mit der Top-Konfiguration erfolgen.")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[SUMMARY] CSV: {csv_path}")
    print(f"[SUMMARY] JSON: {json_path}")
    print(f"[SUMMARY] REPORT: {md_path}")


if __name__ == "__main__":
    main()
