import os, sys, glob
import pandas as pd
import matplotlib.pyplot as plt

def find_latest_metrics():
    # Sucht die zuletzt erzeugte metrics.csv unter logs/qgen_*/
    candidates = sorted(glob.glob(os.path.join("logs", "qgen_*", "metrics.csv")))
    return candidates[-1] if candidates else None

# Quer: wird von main.plot_last_run mit Pfad übergeben; sonst nehme den neuesten Run
path = sys.argv[1] if len(sys.argv) > 1 else find_latest_metrics()
if path is None or not os.path.exists(path):
    raise FileNotFoundError("Keine metrics.csv gefunden. Übergib den Pfad als Argument oder lege Logs in logs/qgen_*/ ab.")

df = pd.read_csv(path)

df["step"] = pd.to_numeric(df["step"], errors="coerce")
df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
df = df.dropna(subset=["step", "loss"])

plt.figure()
plt.plot(df["step"], df["loss"])
plt.xlabel("step")
plt.ylabel("loss")
plt.title("Generator loss")
plt.tight_layout()

out_png = os.path.join(os.path.dirname(path), "plot_loss.png")
plt.savefig(out_png, dpi=150)
plt.show()

print("Gepickt:", path)
print("Gespeichert:", out_png)
