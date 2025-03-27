#!/usr/bin/env python3

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

# =============================================================================
# Global Parameters
# =============================================================================
alpha = 0.01             
Ms = 8.6e5               
mu0 = 4 * np.pi * 1e-7   
gamma = 2.21e5           
Aex = 1.3e-11            
K1 = 5.8e5               
delta = np.pi * np.sqrt(Aex / K1)  
Nx_minus_Ny = 0.05
Kd = 0.5 * mu0 * Ms**2 * Nx_minus_Ny  

# =============================================================================
# Domain Wall Dynamics ODE
# =============================================================================
def domain_wall_dynamics(t, y, alpha, T, u, k, Kd, Ms, gamma, mu0, delta):
    X, phi = y
    dX/dt = [γ*(Kd/(μ0*Ms))*sin(2φ) - T*u + (1-T)*α*δ*u*k] / (1+α²)
    dφ/dt = [-γ*α*(Kd/(μ0*Ms))*sin(2φ) + (1-T)*u*k + T*α*u/δ] / (1+α²)
    return [dX/dt, dφ/dt]

# =============================================================================
# Frequency Parameters
# =============================================================================
def get_frequency_params(frequency):
    if frequency == 22e9:
        T, vg, u = 0.4, 1000.0, 35.0
    elif frequency == 70e9:
        T, vg, u = 0.98, 2000.0, 16.0
    else:
        T, vg, u = 0.7, 1500.0, 25.0
    k = 2 * np.pi * frequency / vg
    return T, vg, u, k

# =============================================================================
# Monte Carlo Simulation
# =============================================================================
def run_monte_carlo(frequency, runs=10):
    T, vg, u, k = get_frequency_params(frequency)
    t_span = (0, 50e-9)
    t_eval = np.linspace(t_span[0], t_span[1], 250)
    y0 = [0, np.pi/2]
    results = np.zeros(len(t_eval))

    for i in range(runs):
        y0_noise = [0, np.pi/2 + np.random.normal(0, 0.01)]
        sol = solve_ivp(lambda t, y: domain_wall_dynamics(t, y, alpha, T, u, k, Kd, Ms, gamma, mu0, delta),
                        t_span, y0_noise, method='RK45', t_eval=t_eval)
        results += sol.y[0]

        # Progress display for Monte Carlo
        progress = int(((i + 1) / runs) * 10)
        print(f"Monte Carlo Progress: {progress}%", end='\r')

    print("\nMonte Carlo Simulation Completed.")
    return t_eval, results / runs

# =============================================================================
# Analytical Velocity Calculations
# =============================================================================
def get_velocity_analytical(T, u, k):
    vi = (-T / (1 + alpha**2)) * u + ((1 - T) * alpha * k / (1 + alpha**2)) * u
    vs = ((1 - T) * k / alpha) * u
    return vi, vs

# =============================================================================
# Frequency Sweep for Velocity
# =============================================================================
def frequency_sweep():
    frequencies = np.linspace(20e9, 80e9, 100)
    results = []

    for i, freq in enumerate(frequencies):
        T, vg, u, k = get_frequency_params(freq)
        vi, vs = get_velocity_analytical(T, u, k)
        results.append((freq, vi, vs))

        # Progress display for frequency sweep
        progress = int(((i + 1) / len(frequencies)) * 100)
        print(f"Frequency Sweep Progress: {progress}%", end='\r')

    print("\nFrequency Sweep Completed.")
    return np.array(results)

# =============================================================================
# Plotting Functions
# =============================================================================
def plot_displacement_curves():
    freqs = [22e9, 70e9]
    plt.figure()
    for freq in freqs:
        t, X_avg = run_monte_carlo(freq)
        plt.plot(t * 1e9, X_avg * 1e9, label=f"{freq/1e9:.0f} GHz")
    plt.xlabel("Time (ns)")
    plt.ylabel("Displacement (nm)")
    plt.title("Domain Wall Displacement vs. Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("dis.png", dpi=300)
    print("Saved graph: dis.png")

def plot_velocity_curves(velocity_data):
    plt.figure()
    frequencies = velocity_data[:, 0] / 1e9
    plt.plot(frequencies, velocity_data[:, 1], label="Initial Velocity")
    plt.plot(frequencies, velocity_data[:, 2], label="Steady-State Velocity")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Domain Wall Velocity vs. Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig("vel.png", dpi=300)
    print("Saved graph: vel.png")

def plot_transmission_and_amplitude():
    frequencies = np.linspace(20e9, 80e9, 100)
    T_vals = [get_frequency_params(f)[0] for f in frequencies]
    amp_vals = [1.0 / (1 + f / 5e9) for f in frequencies]

    plt.figure()
    plt.plot(frequencies / 1e9, T_vals, label="Transmission Coefficient T")
    plt.plot(frequencies / 1e9, amp_vals, label="Spin-Wave Amplitude ρ")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Value")
    plt.title("Transmission Coefficient and Spin-Wave Amplitude")
    plt.legend()
    plt.grid(True)
    plt.savefig("trans.png", dpi=300)
    print("Saved graph: trans.png")

# =============================================================================
# Main Execution
# =============================================================================
def main():
    print("Starting Domain Wall Dynamics Test...")

    # Generate all graphs
    plot_displacement_curves()

    velocity_data = frequency_sweep()
    plot_velocity_curves(velocity_data)

    plot_transmission_and_amplitude()

    print("All tests and graph generation completed successfully.")

if __name__ == "__main__":
    main()
