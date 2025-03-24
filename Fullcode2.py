#!/usr/bin/env python3
"""
Complete Code for Domain Wall Dynamics Simulation

This code integrates:
  • The domain wall dynamics model.
  • Monte Carlo simulation (5000 runs) to obtain robust displacement curves.
  • MPI-parallel frequency sweep for analytical velocity calculation.
  • Plotting functions for displacement, velocity, and transmission/amplitude graphs
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpi4py import MPI

# =============================================================================
# Global Constants and Material Parameters
# =============================================================================
alpha = 0.01             # Gilbert damping constant
Ms = 8.6e5               # Saturation magnetization (A/m)
mu0 = 4 * np.pi * 1e-7   # Magnetic permeability (H/m)
gamma = 2.21e5           # Gyromagnetic ratio (m/As)
Aex = 1.3e-11            # Exchange stiffness (J/m)
K1 = 5.8e5               # Perpendicular anisotropy constant (J/m^3)
delta = np.pi * np.sqrt(Aex / K1)  # Domain wall width (m)
Nx_minus_Ny = 0.05
Kd = 0.5 * mu0 * Ms**2 * Nx_minus_Ny  # Effective anisotropy constant

# =============================================================================
# Domain Wall Dynamics ODE 
# =============================================================================
def domain_wall_dynamics(t, y, alpha, T, u, k, Kd, Ms, gamma, mu0, delta):
    """
    Coupled differential equations for domain wall dynamics.
    
    y[0]: Wall displacement X
    y[1]: Wall tilt angle phi

    dX/dt = [γ*(Kd/(μ0*Ms))*sin(2φ) - T*u + (1-T)*α*δ*u*k] / (1+α²)
    dφ/dt = [-γ*α*(Kd/(μ0*Ms))*sin(2φ) + (1-T)*u*k + T*α*u/δ] / (1+α²)
    """
    X, phi = y
    dXdt = (gamma * (Kd/(mu0*Ms)) * np.sin(2*phi) - T*u + (1-T)*alpha*delta*u*k) / (1 + alpha**2)
    dphidt = (-gamma * alpha * (Kd/(mu0*Ms)) * np.sin(2*phi) + (1-T)*u*k + T*alpha*u/delta) / (1 + alpha**2)
    return [dXdt, dphidt]

# =============================================================================
# Frequency-Dependent Parameter Function
# =============================================================================
def get_frequency_params(frequency):
    """
    Returns frequency-dependent parameters:
      T: Transmission coefficient,
      vg: Group velocity (m/s),
      u: Magnonic spin current parameter,
      k: Spin wave vector = 2π * frequency / vg.

    For 22 GHz and 70 GHz, specific values are provided.
    For other frequencies, placeholder values are used (calibration is required).
    """
    if frequency == 22e9:
        T = 0.4       # Lower transmission implies stronger reflection at 22 GHz
        vg = 1000.0   # Example group velocity (m/s)
        u = 35.0      # Example spin current parameter
    elif frequency == 70e9:
        T = 0.98      # Nearly full transmission at 70 GHz
        vg = 2000.0   # Example group velocity (m/s)
        u = 16.0      # Example spin current parameter
    else:
        T = 0.7       # Placeholder value
        vg = 1500.0   # Placeholder value
        u = 25.0      # Placeholder value
        print(f"Using placeholder T, vg, u for frequency: {frequency/1e9} GHz")
    k = 2 * np.pi * frequency / vg
    return T, vg, u, k

# =============================================================================
# Monte Carlo Simulation for Displacement Curves
# =============================================================================
def run_monte_carlo(frequency, runs=5000):
    """
    Runs Monte Carlo simulations for a given frequency.
    
    For each run, the initial phi is randomized to simulate thermal/structural fluctuations.
    The displacement curve X(t) is computed and then averaged over the specified runs.
    
    Returns:
      t_eval: Time evaluation points
      X_avg: Averaged wall displacement curve
    """
    T, vg, u, k = get_frequency_params(frequency)
    t_span = (0, 50e-9)  # Time span: 0 to 50 ns
    t_eval = np.linspace(t_span[0], t_span[1], 500)  # Increase resolution as needed
    y0 = [0, np.pi/2]    # Initial conditions: X = 0, phi = π/2
    results = np.zeros(len(t_eval))
    
    for _ in range(runs):
        # Add slight Gaussian noise to the initial phi condition
        y0_noise = [0, np.pi/2 + np.random.normal(0, 0.01)]
        sol = solve_ivp(lambda t, y: domain_wall_dynamics(t, y, alpha, T, u, k, Kd, Ms, gamma, mu0, delta),
                        t_span, y0_noise, method='RK45', t_eval=t_eval, rtol=1e-10, atol=1e-12)
        results += sol.y[0]
    
    return t_eval, results / runs

# =============================================================================
# Analytical Velocity Formulas
# =============================================================================
def get_velocity_analytical(T, u, k):
    """
    Computes the analytical initial and steady-state wall velocities:
      vi = (-T/(1+α²))*u + ((1-T)*α*k/(1+α²))*u
      vs = ((1-T)*k/α)*u
    """
    vi = (-T / (1 + alpha**2)) * u + ((1 - T) * alpha * k / (1 + alpha**2)) * u
    vs = ((1 - T) * k / alpha) * u
    return vi, vs

# =============================================================================
# MPI Parallel Frequency Sweep for Velocity Curves
# =============================================================================
def parallel_frequency_sweep():
    """
    Uses MPI to perform a frequency sweep over a specified range.
    
    For each frequency, computes analytical initial (vi) and steady-state (vs) velocities.
    Results from all MPI processes are gathered on the root process.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Frequency range: 20 GHz to 80 GHz (using 5000 points for full resolution)
    frequencies = np.linspace(20e9, 80e9, 5000)
    freq_local = frequencies[rank::size]
    local_results = []
    
    for freq in freq_local:
        T, vg, u, k = get_frequency_params(freq)
        vi, vs = get_velocity_analytical(T, u, k)
        local_results.append((freq, vi, vs))
    
    all_results = comm.gather(local_results, root=0)
    
    if rank == 0:
        # Flatten and sort results by frequency
        all_results = [item for sublist in all_results for item in sublist]
        all_results.sort(key=lambda x: x[0])
        return np.array(all_results)
    else:
        return None

# =============================================================================
# Plotting Functions
# =============================================================================
def transmission_coefficient(frequency):
    """
    Placeholder model for the transmission coefficient.
    For example, T = 0.5 if frequency < 55 GHz, else T = 1.0.
    """
    return 0.5 if frequency < 55e9 else 1.0

def spin_wave_amplitude(frequency):
    """
    Placeholder model for spin-wave amplitude.
    Assumes amplitude decreases with increasing frequency.
    """
    return 1.0 / (1 + frequency/5e9)

def plot_displacement_curves():
    """
    Plots the domain wall displacement vs. time curves for selected frequencies
    using Monte Carlo simulation (5000 runs).
    """
    selected_freqs = [22e9, 70e9]
    plt.figure(figsize=(8, 6))
    for freq in selected_freqs:
        t, X_avg = run_monte_carlo(freq, runs=5000)
        plt.plot(t * 1e9, X_avg * 1e9, label=f"{freq/1e9:.0f} GHz")
    plt.xlabel("Time (ns)")
    plt.ylabel("Displacement (nm)")
    plt.title("Domain Wall Displacement vs. Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("graph_displacement.png", dpi=300)
    plt.show()

def plot_velocity_curves(velocity_data):
    """
    Plots the analytical initial and steady-state velocities vs. frequency.
    """
    frequencies = velocity_data[:, 0] / 1e9  # Convert Hz to GHz for plotting
    initial_vel = velocity_data[:, 1]
    steady_vel = velocity_data[:, 2]
    
    plt.figure(figsize=(8, 6))
    plt.plot(frequencies, initial_vel, 'o-', label="Initial Velocity")
    plt.plot(frequencies, steady_vel, 's-', label="Steady-State Velocity")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Domain Wall Velocity vs. Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig("graph_velocity.png", dpi=300)
    plt.show()

def plot_transmission_and_amplitude():
    """
    Plots the transmission coefficient and spin-wave amplitude vs. frequency.
    """
    frequencies = np.linspace(20e9, 80e9, 5001)
    T_vals = np.array([transmission_coefficient(f) for f in frequencies])
    amp_vals = np.array([spin_wave_amplitude(f) for f in frequencies])
    
    plt.figure(figsize=(8, 6))
    plt.plot(frequencies/1e9, T_vals, 'k-', label="Transmission Coefficient T")
    plt.plot(frequencies/1e9, amp_vals, 'r-', label="Spin-Wave Amplitude ρ")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Value")
    plt.title("Transmission Coefficient and Spin-Wave Amplitude vs. Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig("graph_transmission.png", dpi=300)
    plt.show()

# =============================================================================
# Main Execution
# =============================================================================
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # MPI parallel frequency sweep for velocity curves
    velocity_data = parallel_frequency_sweep()
    
    if rank == 0:
        # Plot displacement curves (Monte Carlo simulation) for selected frequencies
        plot_displacement_curves()
        # Plot velocity curves (analytical) from MPI frequency sweep
        plot_velocity_curves(velocity_data)
        # Plot transmission coefficient and spin-wave amplitude vs. frequency
        plot_transmission_and_amplitude()

if __name__ == "__main__":
    main()
