#!/usr/bin/env python3
"""
Refined Test Code for 1D Domain Wall Motion Simulation (Closer to Paper)

Implements the one-dimensional phenomenological model from Phys. Rev. B 86, 054445 (2012):

  (1 + α²) Ẋ = A_const * sin(2ϕ) - T*u + (1 - T)*α*u
  (1 + α²) ϕ̇ = -A_const * sin(2ϕ) + (1 - T)*u + T*α*u

with:
  - alpha = 0.01 (paper value)
  - u_const = 5.0 (reduced for realistic domain wall shifts)
  - Smooth transmission_coefficient
  - Stronger decay in spin_wave_amplitude
  - Smaller random tilt to reduce velocity noise
  - Displacement plotted in nm
  - Quick test settings (fewer frequencies, fewer time steps)

Author: [Your Name]
Date: [Today's Date]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

# =============================================================================
# Material and Simulation Parameters (from paper)
# =============================================================================
alpha = 0.01           # Paper-like damping
Ms = 8.6e5             # Saturation magnetization (A/m)
mu0 = 4 * np.pi * 1e-7 # Vacuum permeability (H/m)
Nx_minus_Ny = 0.05
K_d = 0.5 * mu0 * Ms**2 * Nx_minus_Ny  # Effective anisotropy (J/m^3)
gamma = 2.21e5         # Gyromagnetic ratio (m/(A*s))
A_const = gamma * K_d / (mu0 * Ms)     # Effective constant (s^-1)

# Reduced magnon velocity for smaller domain wall displacement
u_const = 5.0

# =============================================================================
# Simulation Parameters (Quick Test)
# =============================================================================
t_start = 0.0
t_end = 1e-6                  # 1000 ns total simulation time
num_t_points = 500000         # 500k time steps for faster runtime
t_eval = np.linspace(t_start, t_end, num_t_points)

# Frequency sweep for testing: 50 points from 20 to 80 GHz
test_frequencies = np.linspace(20, 80, 50)

# =============================================================================
# One-Dimensional Model ODE System (from paper)
# =============================================================================
def domain_wall_ode(t, y, T, u):
    """
    1D phenomenological model equations for domain wall motion:
    
    (1 + α²) Ẋ = A_const * sin(2ϕ) - T*u + (1 - T)*α*u
    (1 + α²) ϕ̇ = -A_const * sin(2ϕ) + (1 - T)*u + T*α*u
    
    y[0] = X(t): wall displacement (m)
    y[1] = ϕ(t): wall tilt angle (rad)
    """
    phi = y[1]
    dXdt = (A_const * np.sin(2*phi) - T*u + (1 - T)*alpha*u) / (1 + alpha**2)
    dphidt = (-A_const * np.sin(2*phi) + (1 - T)*u + T*alpha*u) / (1 + alpha**2)
    return [dXdt, dphidt]

# =============================================================================
# Transmission Coefficient & Spin-Wave Amplitude
# =============================================================================
def transmission_coefficient(f_GHz):
    """
    Smooth transition from 0.5 -> 1.0 around 55 GHz.
    """
    return 0.5 + 0.5 * np.tanh((f_GHz - 55) / 5)

def spin_wave_amplitude(f_GHz):
    """
    Stronger decay with frequency, returning smaller amplitude at high f.
    """
    return 1.0 / (1 + f_GHz / 10)

# =============================================================================
# Monte Carlo Simulation (1D Model)
# =============================================================================
def run_monte_carlo(f_GHz, runs=10):
    """
    Runs Monte Carlo simulations for a given frequency using the 1D model.
    Prints progress to the terminal.
    Returns: (time array, averaged X(t)) in meters
    """
    print(f"Starting Monte Carlo simulation for {f_GHz:.1f} GHz...")
    T = transmission_coefficient(f_GHz)
    amp = spin_wave_amplitude(f_GHz)
    # Effective velocity with amplitude
    u = u_const * amp

    results = np.zeros(num_t_points)
    start_sim = time.time()
    
    for i in range(runs):
        if i % 2 == 0:
            elapsed = time.time() - start_sim
            print(f"  Run {i+1}/{runs} complete. Elapsed time: {elapsed:.2f} sec")

        # Initial condition: X(0)=0, ϕ(0)=π/2 + small random tilt
        y0 = [0.0, np.pi/2 + np.random.normal(0, 0.001)]
        sol = solve_ivp(domain_wall_ode, [t_start, t_end], y0, args=(T, u),
                        t_eval=t_eval, rtol=1e-8, atol=1e-10)
        results += sol.y[0]
    
    print(f"Completed {runs} Monte Carlo runs for {f_GHz:.1f} GHz.\n")
    return t_eval, results / runs

# =============================================================================
# Rolling-Window Velocity Calculation
# =============================================================================
def calculate_velocity(X, t, window_size=50):
    """
    Calculates robust initial & steady-state velocities 
    via rolling-window average of the gradient.
    """
    # Convert X(t) from meters to m
    # velocity in m/s
    initial_velocity = np.mean(np.gradient(X[:window_size], t[:window_size]))
    steady_velocity = np.mean(np.gradient(X[-window_size:], t[-window_size:]))
    return initial_velocity, steady_velocity

# =============================================================================
# Velocity Sweep Over Frequencies
# =============================================================================
def run_velocity_sweep():
    """
    Runs the 1D model over a frequency range, returning initial and steady velocities.
    """
    print("Starting velocity sweep over frequencies...\n")
    velocity_results = []
    total = len(test_frequencies)
    start_sweep = time.time()
    
    for idx, f in enumerate(test_frequencies):
        t, X_avg = run_monte_carlo(f, runs=10)
        v_initial, v_steady = calculate_velocity(X_avg, t, window_size=50)
        velocity_results.append((f, v_initial, v_steady))
        
        if idx % 5 == 0:
            elapsed = time.time() - start_sweep
            estimated_total = (elapsed / (idx + 1)) * total
            remaining = estimated_total - elapsed
            print(f"  Processed {idx+1}/{total} freqs. Estimated remaining time: {remaining:.2f} sec")
    
    print("Velocity sweep completed.\n")
    return np.array(velocity_results)

# =============================================================================
# Plotting Functions
# =============================================================================
def plot_displacement_curves():
    """
    Plots domain wall displacement vs. time for f=22 and f=70 GHz (in nm).
    """
    print("Generating domain wall displacement plot...\n")
    freqs_to_plot = [22, 70]
    plt.figure(figsize=(8, 6))
    for f in freqs_to_plot:
        t, X_avg = run_monte_carlo(f, runs=10)
        # Convert X(t) from m to nm for plotting
        plt.plot(t * 1e9, X_avg * 1e9, label=f"{f} GHz")
    plt.xlabel("Time (ns)")
    plt.ylabel("Domain Wall Displacement (nm)")
    plt.title("Domain Wall Displacement vs. Time (1D Model)")
    plt.legend()
    plt.grid(True)
    plt.savefig("test_displacement2.png")
    plt.show()

def plot_velocity_curves(velocity_data):
    """
    Plots initial & steady-state velocity vs. frequency (in m/s).
    """
    print("Generating velocity vs. frequency plot...\n")
    frequencies = velocity_data[:, 0]
    v_initial = velocity_data[:, 1]
    v_steady = velocity_data[:, 2]
    
    plt.figure(figsize=(8, 6))
    plt.plot(frequencies, v_initial, 'o-', label="Initial Velocity")
    plt.plot(frequencies, v_steady, 's-', label="Steady-State Velocity")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Domain Wall Velocity vs. Frequency (1D Model)")
    plt.legend()
    plt.grid(True)
    plt.savefig("test_velocity2.png")
    plt.show()

def plot_transmission_and_amplitude():
    """
    Plots the transmission coefficient and spin-wave amplitude vs. frequency.
    """
    print("Generating transmission coefficient and amplitude plot...\n")
    freq_range = np.linspace(20, 80, 100)
    T_vals = np.array([transmission_coefficient(f) for f in freq_range])
    amp_vals = np.array([spin_wave_amplitude(f) for f in freq_range])
    
    plt.figure(figsize=(8, 6))
    plt.plot(freq_range, T_vals, 'k-', label="Transmission Coefficient T")
    plt.plot(freq_range, amp_vals, 'r-', label="Spin-Wave Amplitude ρ")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Value")
    plt.title("Transmission Coefficient and Spin-Wave Amplitude")
    plt.legend()
    plt.grid(True)
    plt.savefig("test_transmission2.png")
    plt.show()

# =============================================================================
# Main Execution
# =============================================================================
def main():
    total_start = time.time()
    print("Starting 1D Model Test Simulation (Refined)...\n")
    
    velocity_data = run_velocity_sweep()
    plot_displacement_curves()
    plot_velocity_curves(velocity_data)
    plot_transmission_and_amplitude()
    
    total_end = time.time()
    print(f"Test simulation completed in {total_end - total_start:.2f} seconds.\n")

if __name__ == "__main__":
    main()
