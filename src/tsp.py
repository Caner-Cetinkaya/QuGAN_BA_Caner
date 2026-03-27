import os
import numpy as np
import pandas as pd
import zipfile

class TSPDataset:
    def __init__(self, zip_path: str, file_name: str = "tiny.csv", normalize: bool = True):
        # zip_path: Pfad zur ZIP mit CSV oder Verzeichnis mit CSV-Datei
        # file_name: Datei innerhalb der ZIP oder im Verzeichnis; normalize: Min-Max auf [0,1]
        self.zip_path = zip_path
        self.file_name = file_name
        self.normalize = normalize
        self._xy = None  

    def load(self):
        # Lädt Koordinaten aus ZIP oder aus Verzeichnis (CSV) und normalisiert optional
        if os.path.isdir(self.zip_path):
            csv_path = os.path.join(self.zip_path, self.file_name)
            df = pd.read_csv(csv_path, header=None, names=["x", "y"])
        else:
            with zipfile.ZipFile(self.zip_path, "r") as zf:
                with zf.open(self.file_name) as f:
                    df = pd.read_csv(f, header=None, names=["x", "y"])
        xy = df[["x", "y"]].to_numpy(dtype=float)
        if self.normalize:
            xy = self._minmax_normalize(xy)
        self._xy = xy.astype(np.float32)
        return self

    @property
    def xy(self) -> np.ndarray:
        if self._xy is None:
            raise RuntimeError("Bitte zuerst load aufrufen.")
        return self._xy

    @staticmethod
    def _minmax_normalize(xy: np.ndarray) -> np.ndarray:
        mn = xy.min(axis=0)
        mx = xy.max(axis=0)
        den = np.where((mx - mn) > 0, (mx - mn), 1.0)
        return (xy - mn) / den

    def sample_three(self, seed: int | None = None) -> np.ndarray:
        # Wählt drei zufällige Punkte aus dem geladenen Datensatz
        if self._xy is None:
            raise RuntimeError("Bitte zuerst load aufrufen.")
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.xy), size=3, replace=False)
        return self.xy[idx]

    @staticmethod
    def sample_three_uniform(seed: int | None = None) -> np.ndarray:
        # Alternative: drei zufällige Punkte im Einheitsquadrat
        rng = np.random.default_rng(seed)
        return rng.random((3, 2)).astype(np.float32)

    def sample_four_edges(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Wählt vier zufällige Punkte aus dem Datensatz, verbindet sie zyklisch und gibt Kantenlängen zurück.
        
        Returns:
            pts: Array shape (4, 2) — vier 2D-Punkte
            edges: Array shape (4,) — vier Kantenlängen (zyklisches Quadrat/Polygon)
        """
        if self._xy is None:
            raise RuntimeError("Bitte zuerst load aufrufen.")
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.xy), size=4, replace=False)
        pts = self.xy[idx]
        # Zyklische Kanten: (p0->p1, p1->p2, p2->p3, p3->p0)
        rolled = np.roll(pts, -1, axis=0)
        edges = np.linalg.norm(rolled - pts, axis=1)
        return pts, edges.astype(np.float32)

    def sample_four_edges_flat(self, seed: int | None = None) -> np.ndarray:
        """
        Wie sample_four_edges, aber gibt vier Kantenlängen als flaches 1D-Array zurück.
        Nützlich für training_qdis.py (shape (4,)).
        
        Returns:
            edges: Array shape (4,) — vier Kantenlängen
        """
        _, edges = self.sample_four_edges(seed=seed)
        return edges

    @staticmethod
    def check_triangle_inequality(edges: np.ndarray) -> bool:
        """
        Prüft, ob vier Kantenlängen die Dreiecksungleichung erfüllen.
        Für ein Quadrat/Polygon mit 4 Kanten: jeweils die Summe von 3 muss >= der 4-ten sein.
        
        Args:
            edges: Array shape (4,) oder (n, 4)
        
        Returns:
            True wenn Bedingung erfüllt, False sonst
        """
        edges = np.asarray(edges, dtype=float)
        if edges.ndim == 1:
            edges = edges[np.newaxis, :]  # shape (1, 4)
        
        # Prüfe für jede Reihe: sum(3 smallest) >= largest
        for row in edges:
            sorted_edges = np.sort(row)
            if np.sum(sorted_edges[:3]) < sorted_edges[3]:
                return False
        return True
