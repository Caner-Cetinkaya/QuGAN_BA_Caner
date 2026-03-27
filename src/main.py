# main.py
import os, sys, subprocess
import glob
import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np
from training_qgen import run  # Quer: run trainiert QGen und liefert run_dir zurück

def train_many(seeds, steps=300, loss_type="pce"):
    """Startet mehrere Runs und gibt die run_dirs zurück"""
    run_dirs = []
    for s in seeds:
        print(f"[main] starte Run seed={s}")
        #Hier evtl noch für z0 sowas wie z0= np.random.default_rngs().standard_normal(3) um random z0 für jeden seed zu forcen
        rd = run(loss_type=loss_type, seed=s, steps=steps)  # Quer: Aufruf training_qgen.run
        print(f"[main] fertig: {rd}")
        run_dirs.append(rd)
    return run_dirs

def summarize_runs(run_dirs, out_csv=os.path.join("logs", "summary.csv")):
    """Liest alle metrics.csv und schreibt eine Zusammenfassung"""
    # Quer: nutzt metrics.csv aus jedem run_dir, die von training_qgen.run geschrieben wurden
    rows = []
    for rd in run_dirs:
        metrics = os.path.join(rd, "metrics.csv")
        if not os.path.exists(metrics):
            print(f"[WARN] keine metrics.csv in {rd} gefunden – skip")
            continue
        df = pd.read_csv(metrics)

        # robust cast
        for col in ["step","loss","w1","w2","w3","entropy","var"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["step","loss"])
        if df.empty:
            print(f"[WARN] leeres metrics in {rd} – skip")
            continue

        final = df.iloc[-1]
        min_idx = df["loss"].idxmin()
        min_row = df.loc[min_idx]

        rows.append({
            "run_dir": rd,
            "final_loss": float(final["loss"]),
            "final_entropy": float(final.get("entropy", float("nan"))),
            "final_var": float(final.get("var", float("nan"))),
            "min_loss": float(min_row["loss"]),
            "min_loss_step": int(min_row["step"]),
            "steps_logged": int(df["step"].max()),
        })

    if not rows:
        print("[main] keine lauffähigen Runs für Summary gefunden.")
        return None

    summary = pd.DataFrame(rows).sort_values("final_loss").reset_index(drop=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    summary.to_csv(out_csv, index=False)

    print("\n=== Zusammenfassung (beste final_loss oben) ===")
    with pd.option_context('display.max_colwidth', None):
        print(summary.to_string(index=False))
    print("\nSummary gespeichert:", out_csv)
    return summary

def plot_all_losses(run_dirs, out_png=os.path.join("logs", "plot_all_runs.png"), smooth=20):
    """Zeichnet alle Loss-Kurven"""
    # Quer: liest metrics.csv aus jedem run_dir (aus training_qgen.run) und zeichnet loss
    plt.figure()
    any_plotted = False
    for rd in run_dirs:
        metrics = os.path.join(rd, "metrics.csv")
        if not os.path.exists(metrics):
            continue
        df = pd.read_csv(metrics)
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
        df = df.dropna(subset=["step","loss"])
        if df.empty:
            continue

        name = os.path.basename(rd)
        if smooth and smooth > 1:
            df = df.copy()
            df["loss_smooth"] = df["loss"].rolling(smooth, min_periods=1).mean()
            plt.plot(df["step"], df["loss_smooth"], label=f"{name} (roll {smooth})")
        else:
            plt.plot(df["step"], df["loss"], label=name)
        any_plotted = True

    if not any_plotted:
        print("[main] nichts zu plotten, keine validen metrics.")
        return None

    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Generator loss – alle Runs")
    plt.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.show()
    print("Sammel-Plot gespeichert:", out_png)
    return out_png

def plot_last_run(run_dir):
    """Erzeugt die Einzel-Plots für den letzten Run über vorhandenen Skripte"""
    py = sys.executable
    metrics = os.path.join(run_dir, "metrics.csv")
    if os.path.exists(metrics):
        subprocess.run([py, "plot_loss.py", metrics], check=True)
        subprocess.run([py, "plot_weights.py", metrics], check=True)
    else:
        print(f"[main] WARN: {metrics} nicht gefunden – Einzelplots übersprungen.")

def main():
    seeds = [0, 1, 2, 3, 4]
    steps = 300
    loss_type = "pce"  # "pce" oder "log"

    
    run_dirs = train_many(seeds, steps=steps, loss_type=loss_type)
    print("\n[main] alle Runs fertig:")
    for rd in run_dirs:
        print("  ", rd)

    for rd in run_dirs:
        print(f"[main] Plots für {rd}")
        plot_last_run(rd)

    summarize_runs(run_dirs)
    plot_all_losses(run_dirs, smooth=20)

if __name__ == "__main__":
    main()
