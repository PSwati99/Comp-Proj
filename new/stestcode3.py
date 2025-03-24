#!/usr/bin/env python3
"""
Improved Domain Wall Dynamics Simulation Code with Analytical Velocities

This code incorporates the analytical expressions for initial and steady‐state
velocities as given in Phys. Rev. B 86, 054445 (2012) along with improved 
frequency‐dependent functions for the transmission coefficient T(f) and 
spin‐wave amplitude ρ(f) extracted from Fig. 4 of the paper.

Analytical expressions:
  v_i = [-T*u + (1-T)*α*u*k] / (1+α²)
  v_s = ((1-T)*u*k) / α

The plots are saved as PNG files:
  - "dis1.png" for displacement vs. time,
  - "vel1.png" for velocity vs. frequency,
  - "trans1.png" for transmission coefficient and spin-wave amplitude vs. frequency.

Author: [Your Name]
Date: [Today's Date]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =======================
# Global Material Parameters
# =======================
alpha = 0.01                    # Gilbert damping constant
Ms = 8.6e5                      # Saturation magnetization (A/m)
mu0 = 4.0 * np.pi * 1e-7        # Vacuum permeability (H/m)
gamma = 2.21e5                  # Gyromagnetic ratio (m/(A·s))
Nx_minus_Ny = 0.05
Kd = 0.5 * mu0 * Ms**2 * Nx_minus_Ny  # Effective anisotropy (J/m^3)

# Typical domain wall width (taken from paper ~19-20 nm)
delta = 20e-9

# =======================
# Refined Frequency-Dependent Functions (based on Fig. 4)
# =======================

def reflection_coefficient(f_GHz):
    """
    Reflection coefficient R(f) based on Fig. 4.
    Uses a Gaussian peak near 50 GHz to mimic enhanced reflection.
    Below 18 GHz, we assume no propagation.
    """
    if f_GHz < 18.0:
        return 0.0
    center = 50.0      # center frequency (GHz) for the reflection peak
    width  = 5.0       # width (GHz) of the resonance
    peak_amplitude = 0.6  # maximum reflection (60%) at resonance
    R = peak_amplitude * np.exp(-0.5 * ((f_GHz - center)/width)**2)
    return np.clip(R, 0.0, 1.0)

def transmission_coefficient(f_GHz):
    """
    Transmission coefficient T(f) = 1 - R(f)
    """
    T = 1.0 - reflection_coefficient(f_GHz)
    return np.clip(T, 0.0, 1.0)

def spin_wave_amplitude(f_GHz):
    """
    Spin-wave amplitude ρ(f) based on Fig. 4.
    Below 18 GHz, returns 0 (cutoff).
    For f >= 18 GHz, amplitude decays roughly as 1/(1+(f/25)^2)
    with a mild resonance bump near 50 GHz.
    """
    if f_GHz < 18.0:
        return 0.0
    base = 1.0 / (1.0 + (f_GHz/25.0)**2)
    resonance = 1.0 + 0.3 * np.exp(-0.5 * ((f_GHz - 50.0)/3.0)**2)
    return base * resonance

# =======================
# Magnon Velocity Calculation
# =======================

hbar = 1.054571817e-34  # Planck's constant (J·s)

def group_velocity(f_GHz):
    """
    Assumed constant group velocity of spin waves (m/s).
    """
    return 2.0e3  # 2000 m/s

def magnon_velocity_u(f_GHz):
    """
    Compute the effective magnon 'velocity' (flux strength) u.
    u = (γ ℏ n v_g) / (μ0 Ms)
    with n ∝ ρ². An adjustable factor C sets the scale.
    """
    rho = spin_wave_amplitude(f_GHz)
    C = 1e25  # scale factor (adjust as needed)
    n = C * rho**2
    vg = group_velocity(f_GHz)
    return (gamma * hbar * n * vg) / (mu0 * Ms)

# =======================
# Domain Wall ODE System (Incorporating the analytical formulas)
# =======================

def domain_wall_ode(t, y, f_GHz):
    """
    ODE system for domain wall dynamics.
      y[0] = X(t): wall position (m)
      y[1] = φ(t): wall tilt angle (rad)
      
    The ODE is constructed so that, for φ(0)=π/2 (implying sin(2φ)=0),
    the initial wall velocity equals the analytical expression:
      v_i = [-T*u + (1-T)*α*u*k] / (1+α²).
    Here, k = 2πf/v_g (with f in Hz).
    """
    X, phi = y
    T_coef = transmission_coefficient(f_GHz)
    R_coef = 1.0 - T_coef
    u = magnon_velocity_u(f_GHz)
    vg = group_velocity(f_GHz)
    f_Hz = f_GHz * 1e9
    k = 2.0 * np.pi * f_Hz / vg

    factor = gamma * Kd / (mu0 * Ms)
    
    # ODE for X:
    dXdt = ( factor * np.sin(2*phi) 
             - T_coef * u 
             + R_coef * alpha * u * k ) / (1 + alpha**2)
    
    # ODE for φ:
    dphidt = ( -alpha * factor * np.sin(2*phi)
               + R_coef * u * k
               + T_coef * alpha * u ) / (1 + alpha**2)
    
    return [dXdt, dphidt]

# =======================
# Monte Carlo Simulation at a Given Frequency
# =======================

def run_monte_carlo(f_GHz, t_end=50e-9, num_points=5000, runs=20):
    """
    Runs Monte Carlo simulations at a given frequency f_GHz.
    Returns the time array and average wall displacement X(t).
    """
    t_eval = np.linspace(0, t_end, num_points)
    X_total = np.zeros_like(t_eval)
    
    for i in range(runs):
        # Use a small random noise in the initial tilt around π/2
        phi0 = np.pi/2 + np.random.normal(0, 0.01)
        y0 = [0.0, phi0]
        sol = solve_ivp(lambda t, y: domain_wall_ode(t, y, f_GHz),
                        [0, t_end], y0, t_eval=t_eval, rtol=1e-9, atol=1e-12)
        X_total += sol.y[0]
    
    X_avg = X_total / runs
    return t_eval, X_avg

# =======================
# Analytical Velocity Functions (from the paper)
# =======================

def analytical_initial_velocity(f_GHz):
    """
    Calculate the analytical initial velocity v_i at frequency f_GHz.
    v_i = [-T*u + (1-T)*α*u*k] / (1+α²)
    """
    T_coef = transmission_coefficient(f_GHz)
    u = magnon_velocity_u(f_GHz)
    vg = group_velocity(f_GHz)
    f_Hz = f_GHz * 1e9
    k = 2.0 * np.pi * f_Hz / vg
    return (-T_coef*u + (1-T_coef)*alpha*u*k) / (1+alpha**2)

def analytical_steady_velocity(f_GHz):
    """
    Calculate the analytical steady-state velocity v_s.
    v_s = ((1-T)*u*k) / α.
    """
    T_coef = transmission_coefficient(f_GHz)
    u = magnon_velocity_u(f_GHz)
    vg = group_velocity(f_GHz)
    f_Hz = f_GHz * 1e9
    k = 2.0 * np.pi * f_Hz / vg
    return ((1-T_coef)*u*k) / alpha

# =======================
# Frequency Sweep for Velocities
# =======================

def frequency_sweep_velocity(freqs_GHz, t_end=50e-9, num_points=5000, runs=20):
    """
    For each frequency, run the Monte Carlo simulation and compute:
      - the simulated initial velocity (slope from first two time points)
      - the simulated steady-state velocity (slope from the last two time points)
      - the analytical velocities from the paper's expressions.
    
    Returns an array with columns:
    [frequency, sim_v_i, sim_v_s, analytic_v_i, analytic_v_s]
    """
    results = []
    
    for f in freqs_GHz:
        t, X_avg = run_monte_carlo(f, t_end=t_end, num_points=num_points, runs=runs)
        sim_v_i = (X_avg[1] - X_avg[0]) / (t[1] - t[0])
        sim_v_s = (X_avg[-1] - X_avg[-2]) / (t[-1] - t[-2])
        
        analytic_v_i = analytical_initial_velocity(f)
        analytic_v_s = analytical_steady_velocity(f)
        
        results.append((f, sim_v_i, sim_v_s, analytic_v_i, analytic_v_s))
        print(f"f = {f:5.2f} GHz  |  Sim: v_i = {sim_v_i: .2e}, v_s = {sim_v_s: .2e}  |  Analytic: v_i = {analytic_v_i: .2e}, v_s = {analytic_v_s: .2e}")
    
    return np.array(results)

# =======================
# Plotting Functions
# =======================

def plot_displacement_curves():
    """
    Plot displacement vs. time for two example frequencies: 22 GHz and 70 GHz.
    Saves the figure as "dis1.png".
    """
    plt.figure(figsize=(8,6))
    for f in [22.0, 70.0]:
        t, X_avg = run_monte_carlo(f, t_end=50e-9, num_points=5000, runs=20)
        plt.plot(t*1e9, X_avg*1e9, label=f'{f:.0f} GHz')
    plt.xlabel("Time (ns)")
    plt.ylabel("Domain Wall Displacement (nm)")
    plt.title("Domain Wall Displacement vs. Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("dis1.png", dpi=300)
    plt.show()

def plot_velocity_curves(velocity_data):
    """
    Plot simulated and analytical initial and steady-state velocities vs. frequency.
    Saves the figure as "vel1.png".
    """
    freqs = velocity_data[:, 0]
    sim_vi = velocity_data[:, 1]
    sim_vs = velocity_data[:, 2]
    anal_vi = velocity_data[:, 3]
    anal_vs = velocity_data[:, 4]
    
    plt.figure(figsize=(8,6))
    plt.plot(freqs, sim_vi, 'bo-', label="Simulated Initial Velocity")
    plt.plot(freqs, sim_vs, 'ro-', label="Simulated Steady-State Velocity")
    plt.plot(freqs, anal_vi, 'b--', label="Analytical Initial Velocity")
    plt.plot(freqs, anal_vs, 'r--', label="Analytical Steady-State Velocity")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Domain Wall Velocity vs. Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig("vel1.png", dpi=300)
    plt.show()

def plot_transmission_and_amplitude():
    """
    Plot the transmission coefficient and spin-wave amplitude vs. frequency.
    Saves the figure as "trans1.png".
    """
    freqs = np.linspace(18, 80, 200)
    T_vals = np.array([transmission_coefficient(f) for f in freqs])
    amp_vals = np.array([spin_wave_amplitude(f) for f in freqs])
    
    plt.figure(figsize=(8,6))
    plt.plot(freqs, T_vals, 'k-', label="Transmission Coefficient T")
    plt.plot(freqs, amp_vals, 'r-', label="Spin-Wave Amplitude ρ")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Value")
    plt.title("Transmission Coefficient and Spin-Wave Amplitude vs. Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig("trans1.png", dpi=300)
    plt.show()

# =======================
# Main Execution
# =======================

def main():
    print("Plotting displacement curves for 22 GHz and 70 GHz...")
    plot_displacement_curves()
    
    print("Performing frequency sweep for velocities...")
    freq_range = np.linspace(20, 80, 21)  # Frequencies from 20 to 80 GHz
    velocity_data = frequency_sweep_velocity(freq_range, t_end=50e-9, num_points=5000, runs=20)
    
    print("Plotting velocity curves (simulated vs. analytical)...")
    plot_velocity_curves(velocity_data)
    
    print("Plotting transmission coefficient and spin-wave amplitude...")
    plot_transmission_and_amplitude()
    
    # Optionally, save the velocity data to a text file.
    np.savetxt("domain_wall_velocity_data.txt", velocity_data,
               header="Frequency (GHz)   Simulated_v_i (m/s)   Simulated_v_s (m/s)   Analytical_v_i (m/s)   Analytical_v_s (m/s)")
    print("All simulations completed.")

if __name__ == "__main__":
    main()
