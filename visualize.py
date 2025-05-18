import random
import numpy as np
import matplotlib.pyplot as plt
import os

# Constants
DT = 1e-3
DURATION = 1.0
N_STEPS = int(DURATION / DT)
TIME = np.arange(N_STEPS) * DT
DATA_PATH = './data/control_dataset'

X = np.load(os.path.join(DATA_PATH, 'controlX_1ms.npy'))
y = np.load(os.path.join(DATA_PATH, 'controlY_1ms.npy'))

def compute_energy(signal):
    return np.sum(signal.squeeze() ** 2) * DT

def plot_wave(ax, signal, label, color):
    energy = compute_energy(signal)
    ax.plot(TIME, signal.squeeze(), color=color, lw=2)
    ax.fill_between(TIME, 0, signal.squeeze(), step="pre", alpha=0.3, color=color)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(f"{label}", color=color, fontsize=12, weight='bold')
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("I(t)", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    # Show energy text in matching color
    ax.text(0.95, 0.9, f"Energy: {energy:.3f}", transform=ax.transAxes,
            fontsize=10, color=color, ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

def get_triplets(X, y):
    total_triplets = len(X) // 3
    indices = list(range(total_triplets))
    random.shuffle(indices) # show random examples each time
    indices = indices[:3]
    triplets = []
    for i in indices:
        triplet = (X[3*i], X[3*i + 1], X[3*i + 2])
        triplets.append(triplet)
    return triplets

triplets = get_triplets(X, y)

# Colors for the three types
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

n_show = 3
fig, axs = plt.subplots(n_show, 3, figsize=(14, 4 * n_show))
fig.suptitle("Control Dataset Examples with Energy", fontsize=16, weight='bold')

for i in range(n_show):
    m, o, j = triplets[i]
    plot_wave(axs[i, 0], m, "Matched", colors[0])
    plot_wave(axs[i, 1], o, "Off-Frequency", colors[1])
    plot_wave(axs[i, 2], j, "Jittered", colors[2])
    axs[i, 0].set_ylabel(f"Example {i+1}", fontsize=12, weight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("examples.png", dpi=300)
# plt.show()
