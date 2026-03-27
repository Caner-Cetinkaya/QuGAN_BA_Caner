"""
QDiscriminator: 6-Qubit Quantum Circuit für TSP Edge Classification

Architektur (Explizite Layer-Struktur):
========================================
EMBEDDING → [FOR LAYER LOOP] → MESSUNG

1. EMBEDDING (einmalig):
   - AngleEmbedding mit Y-Rotationen
   - Kodiert 6 Kantenlängen auf 6 Qubits
   - Qubit-Zuordnung: Q0←e_ab, Q1←e_bc, Q2←e_cd, Q3←e_da, Q4←e_ac, Q5←e_bd

2. VQC LAYER LOOP (for l in range(n_layers)):
   Pro Layer:
   - RX-Rotationen auf allen Qubits (trainierbar, weights[layer, qubit, 0])
   - RY-Rotationen auf allen Qubits (trainierbar, weights[layer, qubit, 1])
   - RZ-Rotationen auf allen Qubits (trainierbar, weights[layer, qubit, 2])
   - CNOT-Verschränkung zirkulär: Q0→Q1, Q1→Q2, ..., Q5→Q0
   
   Insgesamt: n_layer × 6 Qubits × 3 Gates = 36 trainierbare Parameter (für n_layer=2)

3. MESSUNG:
   - Z-Erwartungswert auf Qubit 0
   - Output ∈ [-1, 1] → normalisiert zu P(real) ∈ [0, 1]
"""

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp


class QDiscriminator:
    def __init__(self, n_layer: int, seed: int | None = 0):
        """
        Initialisiert den 6-Qubit-Discriminator.
        
        Args:
            n_layer: Anzahl der Entangling-Lagen (mehr Lagen = mehr Expressiveness, aber auch mehr Parameter)
            seed: Random seed für Reproduzierbarkeit
        
        Qubit-Zuordnung (1-zu-1 Mapping):
            - Qubit 0 ← e_ab (Kante von Punkt A zu B)
            - Qubit 1 ← e_bc (Kante von Punkt B zu C)
            - Qubit 2 ← e_cd (Kante von Punkt C zu D)
            - Qubit 3 ← e_da (Kante von Punkt D zu A)
            - Qubit 4 ← e_ac (Diagonale von Punkt A zu C)
            - Qubit 5 ← e_bd (Diagonale von Punkt B zu D)
        """
        print(f"[QDiscriminator.__init__] Initialisiere 6-Qubit Quantum Circuit mit {n_layer} Entangling-Lagen")
        self.n_qubits = 6
        self.n_layer = n_layer
        self.rng = np.random.default_rng(seed)
        self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=None)
        
        
        print(f"[QDiscriminator.__init__] Quantum Device: {self.dev}")
        print(f"[QDiscriminator.__init__] Qubit-Zuordnung:")
        print(f"  Q0 <- e_ab  |  Q1 <- e_bc  |  Q2 <- e_cd  |  Q3 <- e_da  |  Q4 <- e_ac  |  Q5 <- e_bd")

        @qml.qnode(self.dev, interface="autograd", diff_method="backprop")
        def circuit(edge_weights, weights):
            """
            Quantum Circuit für die Diskriminierung.
            Explizite Layer-Struktur: Embedding → [Layer Loop] → Messung
            
            1. EMBEDDING: Daten-Kodierung (einmalig)
               - AngleEmbedding: Kantenlängen (normalisiert auf [0,1]) → Y-Rotation auf Qubits
               - edge_weights[i] * π → Rotationswinkel für Qubit i
            
            2. VQC LAYER LOOP: n_layer Iterationen
               Für jede Layer l:
                 a) RX-Rotationen auf allen Qubits (trainierbar)
                 b) RY-Rotationen auf allen Qubits (trainierbar)
                 c) RZ-Rotationen auf allen Qubits (trainierbar)
                 d) CNOT-Verschränkung: Q0-Q1, Q1-Q2, ..., Q5-Q0 (zirkulär)
            
            3. MESSUNG: Klassisches Output
               - Erwartungswert des Z-Operators auf Qubit 0
               - Output ∈ [-1, 1] → wird zu [0, 1] normalisiert für P(real)
            """
            # ===== EMBEDDING: Input-Kodierung (einmalig) =====
            qml.AngleEmbedding(edge_weights * np.pi, wires=range(self.n_qubits), rotation="Y")
            qml.AngleEmbedding(edge_weights * np.pi, wires=range(self.n_qubits), rotation="Z")
            
            # ===== VQC LAYER LOOP: Trainierbare Schichten =====
            # weights Shape: (n_layer, n_qubits, 3) für RX, RY, RZ pro Qubit pro Layer
            for layer in range(self.n_layer):
                # RX-Rotationen auf allen Qubits
                for qubit in range(self.n_qubits):
                    qml.RX(weights[layer, qubit, 0], wires=qubit)
                
                # RY-Rotationen auf allen Qubits
                for qubit in range(self.n_qubits):
                    qml.RY(weights[layer, qubit, 1], wires=qubit)
                
                # RZ-Rotationen auf allen Qubits
                for qubit in range(self.n_qubits):
                    qml.RZ(weights[layer, qubit, 2], wires=qubit)
                
                # CNOT-Verschränkung: Zirkulär (Q0-Q1, Q1-Q2, ..., Q5-Q0)
                for qubit in range(self.n_qubits):
                    target = (qubit + 1) % self.n_qubits
                    qml.CNOT(wires=[qubit, target])

                for qubit in range(self.n_qubits):
                    target = (qubit + 2) % self.n_qubits
                    qml.CNOT(wires=[qubit, target])
            
            # ===== MESSUNG: Output =====
            #return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            return qml.expval(qml.PauliZ(2))

        self.circuit = circuit
        # Gewichte: (n_layer, n_qubits, 3) für RX, RY, RZ pro Qubit pro Layer
        self.weights = 0.5 * self.rng.standard_normal((self.n_layer, self.n_qubits, 3))
        print(f"[QDiscriminator.__init__] Gewichte initialisiert mit Shape {self.weights.shape} ({self.n_layer * self.n_qubits * 3} Parameter)\n")


   # @staticmethod
    #def _to_prob(z_values: float | pnp.ndarray) -> float:
    #    z_arr = np.asarray(z_values, dtype=float)
    #    z_mean = float(np.mean(z_arr))
    #    return 0.5 * (z_mean + 1.0)

    @staticmethod
    def _to_prob(z_scalar: float | pnp.ndarray) -> float:
        return float(0.5 * (z_scalar + 1.0))
    
    @staticmethod
    def _check_triangle_inequality(edges: np.ndarray) -> bool | np.ndarray:
        """
        Prueft ob Kantenlängen die Polygon-Bedingung erfuellen:
        Summe der drei kleineren Kanten >= groesste Kante.

        Args:
            edges: Array shape (4,) oder (n, 4)

        Returns:
            bool fuer (4,)-Input; bool-Array shape (n,) fuer (n,4)-Input.
        """
        edges = np.asarray(edges, dtype=float)
        # BUG-1 FIX: (n, 4) wird jetzt korrekt als Batch behandelt
        if edges.ndim == 2 and edges.shape[1] == 4:
            sorted_edges = np.sort(edges, axis=1)  # (n, 4)
            return (np.sum(sorted_edges[:, :3], axis=1) >= sorted_edges[:, 3])
        if edges.shape != (4,):
            raise ValueError(f"edges muss shape (4,) oder (n,4) haben, bekam {edges.shape}")
        sorted_edges = np.sort(edges)
        return bool(np.sum(sorted_edges[:3]) >= sorted_edges[3])

    def forward(self, edge_weights: np.ndarray, verbose: bool = False) -> float:
        """
        Klassifiziert ein einzelnes 6-Kanten-Tupel.
        
        Args:
            edge_weights: Array mit 6 Kantenlängen [e_ab, e_bc, e_cd, e_da, e_ac, e_bd]
                         normalisiert auf [0,1]
            verbose: Ob Debug-Output gedruckt werden soll
        
        Returns:
            float: Wahrscheinlichkeit P(real) ∈ [0, 1]
                   - Wert nahe 1.0 → "Das sieht nach echten Kanten aus"
                   - Wert nahe 0.0 → "Das sieht nach gefälschten Kanten aus"
        
        Ablauf:
        1. Validiere Input (shape muss (6,) sein)
        2. Normalisiere Kantenlängen ([0,1] erwartet)
        3. Führe Quantum Circuit aus
        4. Normalisiere Output zu Wahrscheinlichkeit
        """
        edge_weights = np.asarray(edge_weights, dtype=float)
        if edge_weights.shape != (6,):
            raise ValueError(f"edge_weights muss shape (6,) haben, bekam {edge_weights.shape}")
        
        # NORMALISIERUNG: Erwarte bereits [0,1] normalisiert vom Generator/Sampler
        edge_weights_normalized = np.clip(edge_weights, 0.0, 1.0)
        
        # Optional Debug
        if verbose:
            edge_names = ["e_ab", "e_bc", "e_cd", "e_da", "e_ac", "e_bd"]
            print(f"  [forward] Input Kanten (normalized [0,1]):")
            for i, (name, val_norm) in enumerate(zip(edge_names, edge_weights_normalized)):
                print(f"    Q{i} <- {name}: {val_norm:.4f}")
        
        z = self.circuit(edge_weights_normalized, self.weights)
        prob = self._to_prob(z)
        
        if verbose:
            print(f"  [forward] Circuit Output: Z={z:.4f} -> P(real)={prob:.4f}\n")
        #if verbose:
        #    z_arr = np.asarray(z, dtype=float)
        #    print(f"  [forward] Circuit Output Zs={z_arr} -> mean(Z)={z_arr.mean():.4f} -> P(real)={prob:.4f}\n")
        
        return prob

    def batch_forward(self, X: np.ndarray) -> np.ndarray:
        """
        Klassifiziert einen Batch von mehreren 6-Kanten-Tupeln.
        
        Args:
            X: Shape (N, 6) - N Samples à 6 Kanten
        
        Returns:
            Array mit N Wahrscheinlichkeiten
        """
        X = np.asarray(X, dtype=float)
        return np.array([self.forward(xy) for xy in X], dtype=float)

    def score_edges(self, edges6: np.ndarray) -> float:
        """
        Mittelt die Scores über mehrere Kanten-Sets (Shape N,6).
        Gibt den Durchschnitt der Wahrscheinlichkeiten zurück.
        """
        ps = self.batch_forward(edges6)
        return float(ps.mean())
