import pennylane as qml
import torch
import torch.nn as nn
import os
import numpy as np
from generator import QGen
from discriminator import QDiscriminator
from tsp import TSPDataset

if __name__ == "__main__":
    gen = QGen()

    # Ein einzelner Noise-Vektor (Quer: wird in gen.quantum_circuit / gen.forward benutzt)
    z = np.array([0.1, -0.5, 0.3])
    print("Noise z:", z)

    # Quantum Output (Expvals) anschauen
    expvals = np.array(gen.quantum_circuit(z, gen.weigths))  # direkter QNode-Aufruf
    print("Erwartungswerte (σZ in [-1,1]):", expvals)

    # Skaliert auf [0,1]
    y = 0.5 * (expvals + 1.0)
    print("Skaliert auf [0,1]:", y)

    # Softmax
    probs = gen._softmax(y)
    print("Softmax-Output (Summe sollte ~1 sein):", probs, "Summe:", probs.sum())

    # Vergleiche mit forward (Quer: nutzt die gleiche Pipeline intern)
    probs_forward = gen.forward(z)
    print("forward(z) gibt:", probs_forward)

    # --- Discriminator-Test mit TSP-Kanten ---
    # Pfad zur CSV im archive-Ordner (direkt hier im Projekt)
    data_dir = os.path.join(os.path.dirname(__file__), "archive")
    file_name = "tiny.csv"  # kann auf small.csv / medium.csv / large.csv geändert werden

    disc = QDiscriminator()
    try:
        ds = TSPDataset(data_dir, file_name=file_name).load()
        pts, edges = ds.sample_four_edges(seed=0)
        print("Zufällige 4 Punkte aus CSV:", pts)
        print("Kantenlängen (zyklisch verbunden):", edges)
        score = disc.forward(edges)
        print("Discriminator-Score für Kanten:", score)
    except Exception as e:
        # Fallback, falls Datei nicht gefunden oder nicht geladen werden kann
        print("Konnte Datensatz nicht laden, Fallback-Kanten:", e)
        edges = np.array([0.5, 0.6, 0.4, 0.5], dtype=float)  # erfüllt Dreiecksungleichung
        score = disc.forward(edges)
        print("Discriminator-Score (Fallback-Kanten):", score)
