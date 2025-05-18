import numpy as np
import os

SAVE_PATH = './data/control_dataset'
os.makedirs(SAVE_PATH, exist_ok=True)

# Constants
DT = 1e-3
DURATION = 1.0
N_STEPS = int(DURATION / DT)

FREQS = [2, 5, 10]  # Hz
DELTA_F = [1, 2, 3]
N_SAMPLES_PER_FREQ = 10
JITTER = 0.02

np.random.seed(42) # For reproducibility

def make_regular_spike_train(n_spikes, T, phase=0.0):
    return np.array([phase + i * T for i in range(n_spikes) if (phase + i * T) < DURATION])

def make_off_freq_spike_train(base_times, f_base, f_off):
    T_base = 1.0 / f_base
    T_off = 1.0 / f_off
    return np.clip((base_times - base_times[0]) * (T_off / T_base) + base_times[0], 0, DURATION)

def make_jittered_spike_train(base_times, jitter):
    return np.clip(base_times + np.random.uniform(-jitter, jitter, base_times.shape), 0, DURATION)

def to_binary(spike_times):
    binary = np.zeros(N_STEPS)
    indices = np.clip((spike_times / DT).astype(int), 0, N_STEPS - 1)
    binary[indices] = 1
    return binary

X = []
y = []

for f in FREQS:
    T = 1.0 / f
    n_spikes = int(DURATION / T) + 1

    for _ in range(N_SAMPLES_PER_FREQ):
        # Global random phase
        phase = np.random.uniform(0, T)

        # MATCHED
        base_times = make_regular_spike_train(n_spikes, T, phase)
        base_times = base_times[:n_spikes]
        binary_matched = to_binary(base_times)

        # OFF-FREQUENCY
        df = np.random.choice(DELTA_F) * np.random.choice([-1, 1])
        f_off = f + df
        off_times = make_off_freq_spike_train(base_times, f, f_off)
        binary_off = to_binary(off_times[:n_spikes])

        # JITTERED
        jittered_times = make_jittered_spike_train(base_times, JITTER)
        binary_jitter = to_binary(jittered_times[:n_spikes])

        # Save
        X.extend([binary_matched, binary_off, binary_jitter])
        y.extend([1, 0, 0])

X = np.array(X).reshape(-1, N_STEPS, 1)
y = np.array(y)

print(f"Generated {len(X)} samples.")

np.save(os.path.join(SAVE_PATH, 'controlX_1ms.npy'), X)
np.save(os.path.join(SAVE_PATH, 'controlY_1ms.npy'), y)
