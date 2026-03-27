from generator import QGen
import numpy as np
import math
import matplotlib.pyplot as plt

# Kleiner Visual-Test: nutzt QGen.forward, interpretiert die drei Ausgaben als Kantenlängen
gen = QGen()
z = np.random.randn(3)  # latentes Rausch-Input
weights = gen.forward(z)  # Quer: ruft generator.forward -> quantum_circuit

a, b, c = weights  # a=AB, b=BC, c=CA
x3 = (c**2 - b**2 + a**2) / (2*a)           # Herleitung Koordinate x3
y3 = math.sqrt(max(c**2 - x3**2, 0))        # Herleitung Koordinate y3

points = np.array([
    [0, 0],
    [a, 0],
    [x3, y3]
])
tri = np.vstack([points, points[0]])  # zurück zum Startpunkt für Plot

plt.plot(tri[:,0], tri[:,1], marker='o')
plt.axis('equal')
plt.show()
