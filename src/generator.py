"""
QGenerator: 6-Qubit Quantum Circuit für TSP Edge Generation

Architektur (Explizite Layer-Struktur):
========================================
NOISE EMBEDDING → [FOR LAYER LOOP] → MESSUNG → SCALING

1. NOISE EMBEDDING (einmalig):
   - AngleEmbedding mit Y-Rotationen
   - Kodiert 6 Random-Noise-Werte auf 6 Qubits
   - Qubit-Zuordnung: Q0←noise[0], Q1←noise[1], ..., Q5←noise[5]
   - Normalisiert auf [0,1] Bereich

2. VQC LAYER LOOP (for l in range(n_layers)):
   Pro Layer:
   - RX-Rotationen auf allen Qubits (trainierbar, weights[layer, qubit, 0])
   - RY-Rotationen auf allen Qubits (trainierbar, weights[layer, qubit, 1])
   - RZ-Rotationen auf allen Qubits (trainierbar, weights[layer, qubit, 2])
   - CNOT-Verschränkung zirkulär: Q0→Q1, Q1→Q2, ..., Q5→Q0
   
   Insgesamt: n_layer × 6 Qubits × 3 Gates = 36 trainierbare Parameter (für n_layer=2)

3. MESSUNG:
   - Z-Erwartungswert auf jedem der 6 Qubits
   - Output pro Qubit ∈ [-1, 1] → normalisiert zu [0, 1]
   - Final Output: 6 Werte ∈ [0, 5000] km (Kantenlängen)
"""

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp


class QGenerator:
    def __init__(self, n_layer: int, seed: int | None = 0):
        """
        Initialisiert den 6-Qubit-Generator.
        
        Args:
            n_layer: Anzahl der Entangling-Lagen
            seed: Random seed für Reproduzierbarkeit
        """
        print(f"[QGenerator.__init__] Initialisiere 6-Qubit Quantum Circuit mit {n_layer} Entangling-Lagen")
        
        self.n_qubits = 6
        self.n_layer = n_layer
        self.rng = np.random.default_rng(seed)
        self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=None)
        
        print(f"[QGenerator.__init__] Quantum Device: {self.dev}")
        print(f"[QGenerator.__init__] Qubit-Zuordnung (Noise -> Kantengeneration):")
        print(f"  Q0 <- noise[0]  |  Q1 <- noise[1]  |  Q2 <- noise[2]  |  Q3 <- noise[3]  |  Q4 <- noise[4]  |  Q5 <- noise[5]")

        @qml.qnode(self.dev, interface="autograd", diff_method="backprop")
        def circuit(noise_vector, weights):
            """
            Quantum Circuit zur Kantengenerierung aus Rausch.
            Unterstuetzt Broadcasting: noise_vector kann Shape (6,) oder (B, 6) sein.
            """
            # 1. EMBEDDING: Noise-Kodierung via AngleEmbedding (wie Discriminator)
            #    Broadcasting wird automatisch korrekt behandelt
            qml.AngleEmbedding(noise_vector * np.pi, wires=range(self.n_qubits), rotation="Y")
            
            # 2. VQC LAYERS: Trainierbare Rotationen + Entanglement
            for layer in range(self.n_layer):
                # RX-Rotationen
                for qubit in range(self.n_qubits):
                    qml.RX(weights[layer, qubit, 0], wires=qubit)
                
                # RY-Rotationen
                for qubit in range(self.n_qubits):
                    qml.RY(weights[layer, qubit, 1], wires=qubit)
                
                # RZ-Rotationen
                for qubit in range(self.n_qubits):
                    qml.RZ(weights[layer, qubit, 2], wires=qubit)
                
                # CNOT-Verschränkung (zirkulär)
                for qubit in range(self.n_qubits):
                    target = (qubit + 1) % self.n_qubits
                    qml.CNOT(wires=[qubit, target])

                for qubit in range(self.n_qubits):
                    target = (qubit + 2) % self.n_qubits
                    qml.CNOT(wires=[qubit, target])
            
            # 3. MESSUNG: Z-Erwartungswert auf jedem Qubit
            measurements = [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            return measurements

        self.circuit = circuit
        
        # Initialisiere Gewichte zufällig (kleine Standardabweichung)
        self.weights = self.rng.normal(0, 1, size=(n_layer, self.n_qubits, 3))
        print(f"[QGenerator.__init__] Gewichte initialisiert: Shape {self.weights.shape}")
        print(f"[QGenerator.__init__] Mean: {np.mean(self.weights):.6f} | Std: {np.std(self.weights):.6f}\n")

    def _to_edge_length(self, z_values: np.ndarray) -> np.ndarray:
        """
        Konvertiert Z-Erwartungswerte [-1, 1] zu normalisierten Kantenlängen [0, 1].
        
        Formel: edge_normalized = 0.5 * (z + 1)
        - z = -1 → 0.0
        - z = 0  → 0.5 (Mitte)
        - z = 1  → 1.0
        
        WICHTIG: Gibt NORMALISIERT [0,1] zurück, NICHT km!
        Die Umrechnung zu km erfolgt erst bei Visualisierung.
        """
        # Normalisiere zu [0, 1]
        p = 0.5 * (z_values + 1.0)
        return p

    def forward(self, noise_vector: np.ndarray) -> np.ndarray:
        """
        Generiert 6 Kantenlängen aus einem Rausch-Vektor.
        
        Args:
            noise_vector: Array mit 6 Rausch-Werten ∈ [0, 1]
        
        Returns:
            Array mit 6 normalisierten Kantenlaengen ∈ [0, 1]
            (Discriminator erwartet [0,1] normalisierte Werte)
        """
        noise_vector = np.asarray(noise_vector, dtype=float)
        if noise_vector.shape != (6,):
            raise ValueError(f"noise_vector muss shape (6,) haben, bekam {noise_vector.shape}")
        
        # Normalisiere auf [0, 1] für Embedding
        noise_normalized = np.clip(noise_vector, 0.0, 1.0)
        
        # Führe Quantum Circuit aus
        z_values = self.circuit(noise_normalized, self.weights)
        z_values = np.asarray(z_values, dtype=float)
        
        # Konvertiere zu Kantenlängen
        edges_normalized = self._to_edge_length(z_values)
        
        return edges_normalized

    def batch_forward(self, noise_batch: np.ndarray, weights=None) -> np.ndarray:
        """
        Generiert einen Batch von Kantenlaengen (vectorisiert fuer Broadcasting).
        
        Args:
            noise_batch: Shape (N, 6) - N Rausch-Vektoren
            weights: Optional trainierbare Gewichte (für PennyLane Differentiation).
                    Wenn None, nutzt self.weights
        
        Returns:
            Array mit Shape (N, 6) Kantenlaengen
        """
        if weights is None:
            weights = self.weights
        
        # WICHTIG: pnp statt np, damit Autograd nicht bricht!
        noise_batch = pnp.array(noise_batch, dtype=float)
        
        # BUG-2 FIX: 1D-Input (6,) zu (1,6) umformen, damit Output-Shape (1,6) ist
        # laut Docstring muss batch_forward immer shape (N,6) zurueckgeben
        squeezed = False
        if noise_batch.ndim == 1:
            noise_batch = noise_batch[pnp.newaxis, :]  # (1, 6)
            squeezed = True
        
        # Batch: nutze QNode Broadcasting
        noise_normalized = pnp.clip(noise_batch, 0.0, 1.0)
        z_values = self.circuit(noise_normalized, weights)
        
        # z_values ist list[expval], wobei jedes expval Shape (B,) hat
        # Konvertiere zu (B, 6)
        if isinstance(z_values, (list, tuple)):
            z_arr = pnp.stack(z_values, axis=1)  # (B, 6)
        else:
            z_arr = pnp.array(z_values)
            # Falls shape (6, B), transponiere zu (B, 6)
            if z_arr.ndim == 2 and z_arr.shape[0] == 6:
                z_arr = z_arr.T
        
        edges = self._to_edge_length(z_arr)  # shape (N, 6) oder (1, 6)
        # BUG-2 FIX: bei urspruenglich 1D-Input (1,6) zurueckgeben (nicht quetschen)
        _ = squeezed  # Flag genutzt um zu dokumentieren dass wir 1D->2D expandiert haben
        return edges  # NICHT zu np.asarray casten!
