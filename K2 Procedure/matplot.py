import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# Dati grezzi forniti
raw_data = {
    1000: {19: 6, 18: 25, 17: 32, 16 : 37},
    5000: {16: 97, 18: 1, 17: 2},
    10000: {16: 90, 17: 1, 15: 9},
    20000: {16: 85, 17: 1, 15: 14},
    50000: {15: 90, 16: 10},
    100000: {15: 32, 16: 24, 14: 30, 13: 14},
    500000 : { 13: 25, 14: 20, 15: 40, 11 : 15}
}

# Estrai i dati in formato lungo
x_samples = []
y_differences = []
z_ratios = []

for samples, differences in raw_data.items():
    total = sum(differences.values())
    for diff, count in differences.items():
        x_samples.append(samples)
        y_differences.append(diff)
        z_ratios.append(count / total)

# Crea il grafico 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Dimensioni delle barre
dx = 1000
dy = 0.4
dz = z_ratios

# Coordinate base delle barre
xs = np.array(x_samples)
ys = np.array(y_differences)
zs = np.zeros_like(xs)

ax.bar3d(xs, ys, zs, dx, dy, dz, shade=True)

# Etichette assi
ax.set_xlabel("Number of Samples")
ax.set_ylabel("Difference")
ax.set_zlabel("Relative Ratio")
ax.set_title("3D Distribution of Structural Differences Water Network")

plt.tight_layout()
plt.show()
