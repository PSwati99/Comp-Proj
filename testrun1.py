#!/usr/bin/env python3
"""
Optimized Test Code for 1D Domain Wall Motion Simulation (Refined)

This code implements the one-dimensional phenomenological model from
Phys. Rev. B 86, 054445 (2012):

  (1 + α²) Ẋ = A_const * sin(2ϕ) - T*u + (1 - T)*α*u
  (1 + α²) ϕ̇ = -A_const * sin(2ϕ) + (1 - T)*u + T*α*u

where:
  - A_const = γ * K_d / (μ₀ * M_s)
  - y[0] = X(t) is the domain wall displacement
  - y[1] = ϕ(t) is the tilt angle

Modifications include:
    - Tuned damping (α = 0.005) and increased magnon velocity (u_const = 20.0)
    - Smooth transmission coefficient function
    - Gradually decaying spin-wave amplitude function
    - Rolling-window velocity calculation for robust estimates
    - Detailed progress printouts during simulation

Runtime is scaled (fewer time steps, frequencies, and Monte Carlo runs) 
so that it completes in ~5 minutes on your laptop.

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
alpha = 0.005            # Lower damping to allow more motion
Ms = 8.6e5               # Saturation magnetization (A/m)
mu0 = 4 * np.pi * 1e-7   # Vacuum permeability (H/m)
Nx_minus_Ny = 0.05       # Demagnetization factor difference
K_d = 0.5 * mu0 * Ms**2 * Nx_minus_Ny  # Effective anisotropy (J/m^3)
gamma = 2.21e5           # Gyromagnetic ratio (m/(A*s))
A_const = gamma * K_d / (mu0 * Ms)     # Effective constant (s^-1)
u_const = 20.0           # Increased effective magnon velocity for stronger drive

# =============================================================================
# Simulation Parameters for Quick Test
# =============================================================================
t_start = 0.0
t_end = 1e-6                 # 1000 ns total simulation time
num_t_points = 500000        # 500K time steps for faster runtime
t_eval = np.linspace(t_start, t_end, num_t_points)

# Frequency sweep for testing: 50 points from 20 to 80 GHz
test_frequencies = np.linspace(20, 80, 55)

# =============================================================================
# One-Dimensional Model ODE System (from paper)
# =============================================================================
def domain_wall_ode(t, y, T, u):
    """
    1D phenomenological model equations for domain wall motion:
    
    (1 + α²) Ẋ = A_const * sin(2ϕ) - T*u + (1 - T)*α*u
    (1 + α²) ϕ̇ = -A_const * sin(2ϕ) + (1 - T)*u + T*α*u
    
    where:
      y[0] = X(t): wall displacement,
      y[1] = ϕ(t): wall tilt angle.
    """
    phi = y[1]
    dXdt = (A_const * np.sin(2 * phi) - T * u + (1 - T) * alpha * u) / (1 + alpha**2)
    dphidt = (-A_const * np.sin(2 * phi) + (1 - T) * u + T * alpha * u) / (1 + alpha**2)
    return [dXdt, dphidt]

# =============================================================================
# Transmission Coefficient and Spin-Wave Amplitude Functions
# =============================================================================
def transmission_coefficient(f_GHz):
    """
    Smooth transition for transmission coefficient.
    For frequencies below ~55 GHz, T ~0.5, transitioning smoothly to T ~1.0 above 55 GHz.
    """
    return 0.5 + 0.5 * np.tanh((f_GHz - 55) / 5)

def spin_wave_amplitude(f_GHz):
    """
    A gradually decaying spin-wave amplitude with frequency.
    """
    return 1.0 / (1 + f_GHz / 100)

# =============================================================================
# Monte Carlo Simulation using the 1D Model
# =============================================================================
def run_monte_carlo(f_GHz, runs=10):
    """
    Performs Monte Carlo simulations for a given frequency using the 1D model.
    Includes terminal progress updates.
    Returns the time array and the averaged displacement X(t).
    """
    print(f"Starting Monte Carlo simulation for {f_GHz:.1f} GHz...")
    T = transmission_coefficient(f_GHz)
    u = u_const
    results = np.zeros(num_t_points)
    start_sim = time.time()
    
    for i in range(runs):
        if i % 2 == 0:
            elapsed = time.time() - start_sim
            print(f"  Run {i+1}/{runs} complete. Elapsed time: {elapsed:.2f} sec")
        # Initial condition: X(0)=0, ϕ(0)=π/2 with a small random perturbation
        y0 = [0.0, np.pi/2 + np.random.normal(0, 0.005)]
        sol = solve_ivp(domain_wall_ode, [t_start, t_end], y0, args=(T, u),
                        t_eval=t_eval, rtol=1e-8, atol=1e-10)
        results += sol.y[0]
    
    print(f"Completed {runs} Monte Carlo runs for {f_GHz:.1f} GHz.\n")
    return t_eval, results / runs

# =============================================================================
# Robust Velocity Calculation using a Rolling Window
# =============================================================================
def calculate_velocity(X, t, window_size=50):
    """
    Calculates robust estimates for the initial and steady-state velocities
    using a rolling window average of the gradient.
    """
    initial_velocity = np.mean(np.gradient(X[:window_size], t[:window_size]))
    steady_velocity = np.mean(np.gradient(X[-window_size:], t[-window_size:]))
    return initial_velocity, steady_velocity

# =============================================================================
# Velocity Sweep Over Frequencies
# =============================================================================
def run_velocity_sweep():
    """
    Runs Monte Carlo simulations over a range of frequencies using the 1D model,
    and calculates initial and steady-state velocities.
    Prints progress updates.
    Returns an array with columns: frequency, initial velocity, steady-state velocity.
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
            print(f"  Processed {idx+1}/{total} frequencies. Estimated remaining time: {remaining:.2f} sec")
    
    print("Velocity sweep completed.\n")
    return np.array(velocity_results)

# =============================================================================
# Plotting Functions
# =============================================================================
def plot_displacement_curves():
    """
    Generates and plots domain wall displacement vs. time for selected frequencies (22 and 70 GHz)
    using the 1D model.
    """
    print("Generating domain wall displacement plot...\n")
    freqs_to_plot = [22, 70]
    plt.figure(figsize=(8, 6))
    for f in freqs_to_plot:
        t, X_avg = run_monte_carlo(f, runs=10)
        plt.plot(t * 1e9, X_avg * 1e9, label=f"{f} GHz")
    plt.xlabel("Time (ns)")
    plt.ylabel("Domain Wall Displacement (m)")
    plt.title("Domain Wall Displacement vs. Time (1D Model)")
    plt.legend()
    plt.grid(True)
    plt.savefig("test_displacement1.png")
    plt.show()

def plot_velocity_curves(velocity_data):
    """
    Generates and plots the initial and steady-state velocities vs. frequency based on the 1D model.
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
    plt.savefig("test_velocity1.png")
    plt.show()

def plot_transmission_and_amplitude():
    """
    Generates and plots the transmission coefficient and spin-wave amplitude as functions of frequency.
    """
    print("Generating transmission coefficient and amplitude plot...\n")
    freq_range = np.linspace(20, 80, 100)
    T_vals = np.array([transmission_coefficient(f) for f in freq_range])
    amp_vals = np.array([spin_wave_amplitude(f) for f in freq_range])
    
    plt.figure(figsize=(8, 6))
    plt.plot(freq_range, T_vals, 'k-', label="Transmission Coefficient T")
    plt.plot(freq_range, amp_vals, 'r-', label="Spin-Wave Amplitude ρ")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("T and rho")
    plt.title("Transmission Coefficient and Spin-Wave Amplitude")
    plt.legend()
    plt.grid(True)
    plt.savefig("test_transmission1.png")
    plt.show()

# =============================================================================
# Main Execution
# =============================================================================
def main():
    total_start = time.time()
    print("Starting 1D Model Test Simulation...\n")
    
    # Velocity sweep to obtain velocity data vs. frequency
    velocity_data = run_velocity_sweep()
    
    # Generate displacement curves for 22 and 70 GHz
    plot_displacement_curves()
    
    # Generate velocity vs. frequency plot
    plot_velocity_curves(velocity_data)
    
    # Generate transmission coefficient and amplitude plot
    plot_transmission_and_amplitude()
    
    total_end = time.time()
    print(f"Test simulation completed in {total_end - total_start:.2f} seconds.\n")

if __name__ == "__main__":
    main()
