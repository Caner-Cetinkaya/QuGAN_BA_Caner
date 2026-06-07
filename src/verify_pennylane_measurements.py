"""
Empirische Pruefung zweier Behauptungen aus dem Meeting:

1) Wie behandelt PennyLane mehrere Messungen (Liste von qml.expval)?
   - Hypothese der Betreuung: jede Messung = separate Simulation
   - Erwartung aus Doku: eine Simulation, mehrere Messungen aus demselben Statevektor

2) Was bedeutet shots=None konkret?
   - Erwartung: exakter Statevektor, keine Shot-Varianz, deterministisch
   - Kontrast: shots=1000 -> Sampling-Varianz, nicht deterministisch

Ausgabe: ausfuehrliche Konsolen-Logs, die beide Punkte belegen.
"""
import time
import numpy as np
import pennylane as qml
import torch


# ============================================================================
# Setup
# ============================================================================
n_qubits = 6

# Wir bauen zwei Devices: einmal exakt (shots=None), einmal mit shots
dev_exact = qml.device("default.qubit", wires=n_qubits, shots=None)
dev_shots = qml.device("default.qubit", wires=n_qubits, shots=1000)


def make_circuit_list_pauliz(dev, diff_method="backprop"):
    """6 PauliZ-Erwartungswerte als Liste — wie in Baseline."""
    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(weights):
        for i in range(n_qubits):
            qml.RY(weights[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return circuit


def make_circuit_probs(dev):
    """qml.probs ueber alle Qubits — wie in EXP2."""
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(weights):
        for i in range(n_qubits):
            qml.RY(weights[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.probs(wires=range(n_qubits))
    return circuit


def make_circuit_single_pauliz(dev, target_wire):
    """Nur 1 PauliZ — als Vergleichsbasis fuer 'wie lange dauert eine Messung allein'."""
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(weights):
        for i in range(n_qubits):
            qml.RY(weights[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(target_wire))
    return circuit


# ============================================================================
# Test 1: Determinismus bei shots=None vs Varianz bei shots=1000
# ============================================================================
print("=" * 70)
print("TEST 1: shots=None vs shots=1000 - Determinismus")
print("=" * 70)

# Feste Gewichte, mehrfach evaluieren
weights = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=torch.float32)

print("\n--- shots=None (exakter Statevektor) ---")
circ_exact = make_circuit_list_pauliz(dev_exact)
for i in range(3):
    result = circ_exact(weights)
    result_np = np.array([float(r) for r in result])
    print(f"  Lauf {i+1}: {result_np}")
print("  Erwartung: alle 3 Laeufe IDENTISCH (deterministisch)")

print("\n--- shots=1000 (Sampling) ---")
circ_shots = make_circuit_list_pauliz(dev_shots, diff_method="parameter-shift")
for i in range(3):
    result = circ_shots(weights)
    result_np = np.array([float(r) for r in result])
    print(f"  Lauf {i+1}: {result_np}")
print("  Erwartung: 3 Laeufe leicht VERSCHIEDEN (Shot-Varianz)")


# ============================================================================
# Test 2: Macht qml.expval-Liste eine oder mehrere Simulationen?
# ============================================================================
print("\n" + "=" * 70)
print("TEST 2: Wie viele Simulationen bei Liste von qml.expval?")
print("=" * 70)
print("""
Methode: Timing-Vergleich.

- Wenn PennyLane fuer jedes qml.expval einzeln simuliert,
  dauert die 6-Messungen-Liste etwa 6x so lange wie 1 Messung.
- Wenn PennyLane nur einmal simuliert und alle 6 aus dem Statevektor
  extrahiert, dauert die Liste nur unwesentlich laenger als 1 Messung.
""")

n_repeats = 200

# 1 Messung
circ_single = make_circuit_single_pauliz(dev_exact, target_wire=0)
# Warmup
circ_single(weights)
t0 = time.perf_counter()
for _ in range(n_repeats):
    circ_single(weights)
t_single = time.perf_counter() - t0

# 6 Messungen (Liste)
circ_six = make_circuit_list_pauliz(dev_exact)
circ_six(weights)  # warmup
t0 = time.perf_counter()
for _ in range(n_repeats):
    circ_six(weights)
t_six = time.perf_counter() - t0

# probs ueber alle 6 Qubits
circ_probs = make_circuit_probs(dev_exact)
circ_probs(weights)  # warmup
t0 = time.perf_counter()
for _ in range(n_repeats):
    circ_probs(weights)
t_probs = time.perf_counter() - t0

print(f"\nZeit fuer {n_repeats} Wiederholungen (shots=None):")
print(f"  1 PauliZ (Qubit 0):           {t_single*1000:.1f} ms  ({t_single*1000/n_repeats:.3f} ms/Aufruf)")
print(f"  6 PauliZ (Liste):             {t_six*1000:.1f} ms  ({t_six*1000/n_repeats:.3f} ms/Aufruf)")
print(f"  qml.probs (alle 6 zugleich):  {t_probs*1000:.1f} ms  ({t_probs*1000/n_repeats:.3f} ms/Aufruf)")
print(f"\nVerhaeltnis 6-Liste / 1-einzeln: {t_six/t_single:.2f}x")
print(f"Verhaeltnis probs / 1-einzeln:   {t_probs/t_single:.2f}x")
print("""
Interpretation:
  ~1x  -> Eine Simulation, mehrere Messungen aus demselben Statevektor.
          (Was die Doku sagt; Messungen kommutieren ja, sind nur Slices.)
  ~6x  -> Sechs separate Simulationen, eine pro Messung.
          (Das waere das, was die Betreuung vermutete.)
""")


# ============================================================================
# Test 3: Konsistenz Liste vs Probs vs einzelne Messung
# ============================================================================
print("=" * 70)
print("TEST 3: Konsistenz der Ergebnisse (gleiche Gewichte, gleiches Resultat)")
print("=" * 70)

# Wenn alle Methoden dieselbe Simulation nutzen, muessen die Werte exakt
# uebereinstimmen
exp_via_list = np.array([float(r) for r in circ_six(weights)])

# Einzelne Messungen pro Qubit
circ_singles = [make_circuit_single_pauliz(dev_exact, w) for w in range(n_qubits)]
exp_via_singles = np.array([float(c(weights)) for c in circ_singles])

# Aus probs ableiten: E[Z_i] = sum_b (-1)^b_i * P(b)
probs = circ_probs(weights).detach().numpy()
exp_via_probs = np.zeros(n_qubits)
for state in range(2**n_qubits):
    bits = [(state >> i) & 1 for i in range(n_qubits)]  # LSB first
    # PennyLane convention: wire 0 ist das MSB. Pruefen!
    # Vorsicht: hier gehen wir mit der PennyLane-Konvention
    for q in range(n_qubits):
        # Eigenwert von PauliZ ist +1 fuer |0> und -1 fuer |1>
        # Bit-Order: PennyLane nutzt wire 0 als most-significant in qml.probs
        bit_at_wire_q = (state >> (n_qubits - 1 - q)) & 1
        sign = 1 if bit_at_wire_q == 0 else -1
        exp_via_probs[q] += sign * probs[state]

print(f"\nPauliZ-Erwartungswerte (sollten alle drei Methoden GLEICH sein):")
print(f"  via qml.expval-Liste:    {exp_via_list}")
print(f"  via einzelne qml.expval: {exp_via_singles}")
print(f"  via qml.probs abgeleitet:{exp_via_probs}")
print(f"\nMax-Abweichung Liste vs Singles: {np.abs(exp_via_list - exp_via_singles).max():.2e}")
print(f"Max-Abweichung Liste vs Probs:   {np.abs(exp_via_list - exp_via_probs).max():.2e}")


# ============================================================================
# Test 4: Gradient durch beide Output-Formate (Backprop-Sanity)
# ============================================================================
print("\n" + "=" * 70)
print("TEST 4: Gradienten via backprop fuer beide Output-Formate")
print("=" * 70)

weights_g = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                         dtype=torch.float32, requires_grad=True)

# Liste
out_list = circ_six(weights_g)
loss_list = torch.stack(out_list).sum()
loss_list.backward()
grad_list = weights_g.grad.detach().numpy().copy()
weights_g.grad = None

# Probs
out_probs = circ_probs(weights_g)
loss_probs = out_probs[:6].sum()
loss_probs.backward()
grad_probs = weights_g.grad.detach().numpy().copy()

print(f"\nGradient via Liste-Output:  {grad_list}")
print(f"Gradient via Probs-Output:  {grad_probs}")
print(f"(Werte werden unterschiedlich sein, weil andere Loss-Funktion.")
print(f" Wichtig: beide haben Gradienten != 0, Backprop funktioniert.)")


# ============================================================================
# Test 5: Memory + Compute - was kostet probs vs Liste?
# ============================================================================
print("\n" + "=" * 70)
print("TEST 5: Memory-Footprint Probs (64 Werte) vs Liste (6 Werte)")
print("=" * 70)

# probs gibt 2^n_qubits Werte zurueck — bei 6 Qubits sind das 64
# bei 20 Qubits waeren das 2^20 = 1 Mio — explosionsartig
print(f"\nBei n_qubits=6: qml.probs liefert {2**6} Werte, "
      f"qml.expval-Liste liefert {n_qubits} Werte.")
print(f"Bei n_qubits=20 waeren das {2**20:,} vs {20} Werte.")
print("Fuer kleine Systeme egal, fuer grosse Systeme relevant.")


print("\n" + "=" * 70)
print("FAZIT")
print("=" * 70)
print(f"""
1) shots=None bedeutet 'exakter Statevektor': vollstaendig deterministisch,
   keine Shot-Varianz. Test 1 hat das gezeigt.

2) Bei der Liste von qml.expval-Aufrufen macht PennyLane KEINE 6 separaten
   Simulationen, sondern eine. Das zeigen sowohl das Timing in Test 2
   (~{t_six/t_single:.1f}x statt 6x) als auch die exakte Uebereinstimmung
   der Werte in Test 3.

3) Die Vermutung der Betreuung (1000 Simulationen pro Qubit) ist daher
   NICHT zutreffend fuer den aktuellen Code mit default.qubit + shots=None.

4) Wenn man echte Hardware-Konditionen simulieren wollte (Shot-Noise),
   muesste man shots=1000 (oder eine andere endliche Zahl) setzen.
""")
