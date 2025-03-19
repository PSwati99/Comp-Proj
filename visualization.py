import numpy as np
import matplotlib.pyplot as plt
from config import freq_min, freq_max, num_freq_points
from utils import transmission_coefficient, spin_wave_amplitude
from simulation import run_monte_carlo

def plot_displacement():
    """Plots domain wall displacement vs. time for 22 GHz and 70 GHz."""
    freqs = [22, 70]
    plt.figure(figsize=(8,6))
    for f in freqs:
        t, X_avg = run_monte_carlo(f)
        plt.plot(t*1e9, X_avg, label=f'{f} GHz')
    plt.xlabel("Time (ns)")
    plt.ylabel("Displacement (m)")
    plt.title("Domain Wall Displacement vs. Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("graph_displacement.png")
    plt.show()

def plot_velocity(velocity_data):
    """Plots initial and steady-state velocity vs. frequency."""
    frequencies = velocity_data[:,0]
    initial_vel = velocity_data[:,1]
    steady_vel = velocity_data[:,2]

    plt.figure(figsize=(8,6))
    plt.plot(frequencies, initial_vel, label="Initial Velocity")
    plt.plot(frequencies, steady_vel, label="Steady-State Velocity")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid(True)
    plt.savefig("graph_velocity.png")
    plt.show()

def plot_transmission():
    """Plots transmission coefficient and spin-wave amplitude vs. frequency."""
    frequencies = np.linspace(freq_min, freq_max, num_freq_points)
    T_vals = [transmission_coefficient(f) for f in frequencies]
    amp_vals = [spin_wave_amplitude(f) for f in frequencies]

    plt.figure(figsize=(8,6))
    plt.plot(frequencies, T_vals, label="Transmission Coefficient")
    plt.plot(frequencies, amp_vals, label="Spin-Wave Amplitude")
    plt.xlabel("Frequency (GHz)")
    plt.grid(True)
    plt.savefig("graph_transmission.png")
    plt.show()
