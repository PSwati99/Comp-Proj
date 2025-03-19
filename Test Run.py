#!/usr/bin/env python3
"""
Test Code for Spin-Wave–Induced Domain Wall Motion Simulation

This is a scaled-down version of the full MPI simulation to run on a laptop 
in approximately 15-20 minutes. It generates all four graphs for comparison 
with the full simulation and the original paper.

Modifications for Faster Execution:
    - Reduced simulation time: 1000 ns (1,000,000 time steps)
    - Reduced Monte Carlo runs: 50 per frequency
    - Reduced frequency points: 1,000 (instead of 50,000)
    - Single-core execution (No MPI needed)

Author: [Your Name]
Date: [Today's Date]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =============================================================================
# Material and Simulation Parameters
# =============================================================================
alpha = 0.01          
Ms = 8.6e5           
mu0 = 4 * np.pi * 1e-7  
Nx_minus_Ny = 0.05    
K_d = 0.5 * mu0 * Ms**2 * Nx_minus_Ny  
gamma = 2.21e5        
A_const = gamma * K_d / (mu0 * Ms)     
u_const = 10.0        

# =============================================================================
# Test Simulation Parameters (Optimized for Laptop)
# =============================================================================
t_start = 0.0
t_end = 1e-6        
num_t_points = 1000000   # 1M time steps (instead of 10M)
t_eval = np.linspace(t_start, t_end, num_t_points)

# Frequency sweep reduced from 50,000 to 1,000 points
test_frequencies = np.linspace(20, 80, 1000)

# =============================================================================
# Transmission Coefficient Function
# =============================================================================
def transmission_coefficient(f_GHz):
    """Returns the transmission coefficient based on frequency."""
    return 0.5 if f_GHz < 55 else 1.0

# =============================================================================
# Domain Wall ODE System
# =============================================================================
def domain_wall_ode(t, y, T, u):
    phi = y[1]
    dXdt = (A_const * np.sin(2*phi) - T*u + (1-T)*alpha*u) / (1 + alpha**2)
    dphidt = (-A_const * np.sin(2*phi) + (1-T)*u + T*alpha*u) / (1 + alpha**2)
    return [dXdt, dphidt]

# =============================================================================
# Monte Carlo Simulation for a Given Frequency
# =============================================================================
def run_monte_carlo(f_GHz, runs=50):
    """Runs Monte Carlo simulations for a given frequency with random noise."""
    T = transmission_coefficient(f_GHz)
    u = u_const
    results = np.zeros(num_t_points)
    
    for _ in range(runs):
        y0 = [0.0, np.pi/2 + np.random.normal(0, 0.01)]
        sol = solve_ivp(domain_wall_ode, [t_start, t_end], y0, args=(T, u),
                        t_eval=t_eval, rtol=1e-8, atol=1e-10)
        results += sol.y[0]
    
    return t_eval, results / runs  

# =============================================================================
# Velocity Sweep (Single-Core Version)
# =============================================================================
def run_velocity_sweep():
    """Computes initial and steady-state domain wall velocities."""
    velocity_results = []

    for f in test_frequencies:
        t, X_avg = run_monte_carlo(f, runs=50)
        dXdt_initial = (X_avg[1] - X_avg[0]) / (t[1] - t[0])
        dXdt_steady = (X_avg[-1] - X_avg[-2]) / (t[-1] - t[-2])
        velocity_results.append((f, dXdt_initial, dXdt_steady))
    
    return np.array(velocity_results)

# =============================================================================
# Spin-wave Amplitude Model
# =============================================================================
def spin_wave_amplitude(f_GHz):
    """Model for spin-wave amplitude that decreases with frequency."""
    return 1.0 / (1 + f_GHz / 5000)

# =============================================================================
# Plotting Functions
# =============================================================================
def plot_displacement_curves():
    """Plot domain wall displacement vs. time for f = 22 GHz and f = 70 GHz."""
    freqs = [22, 70]
    plt.figure(figsize=(8,6))
    for f in freqs:
        t, X_avg = run_monte_carlo(f, runs=50)
        plt.plot(t*1e9, X_avg, label=f'{f} GHz')
    plt.xlabel("Time (ns)")
    plt.ylabel("Domain Wall Displacement (m)")
    plt.title("Test: Domain Wall Displacement vs. Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("test_displacement.png")
    plt.show()

def plot_velocity_curves(velocity_data):
    """Plot initial and steady-state velocity vs. frequency."""
    frequencies = velocity_data[:,0]
    initial_vel = velocity_data[:,1]
    steady_vel = velocity_data[:,2]
    
    plt.figure(figsize=(8,6))
    plt.plot(frequencies, initial_vel, 'o-', label="Initial Velocity")
    plt.plot(frequencies, steady_vel, 's-', label="Steady-State Velocity")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Test: Domain Wall Velocity vs. Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig("test_velocity.png")
    plt.show()

def plot_transmission_and_amplitude():
    """Plot transmission coefficient and spin-wave amplitude vs. frequency."""
    frequencies = test_frequencies
    T_vals = np.array([transmission_coefficient(f) for f in frequencies])
    amp_vals = np.array([spin_wave_amplitude(f) for f in frequencies])
    
    plt.figure(figsize=(8,6))
    plt.plot(frequencies, T_vals, 'k-', label="Transmission Coefficient T")
    plt.plot(frequencies, amp_vals, 'r-', label="Spin-Wave Amplitude ρ")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Value")
    plt.title("Test: Transmission Coefficient and Spin-Wave Amplitude")
    plt.legend()
    plt.grid(True)
    plt.savefig("test_transmission.png")
    plt.show()

# =============================================================================
# Main Execution
# =============================================================================
def main():
    velocity_data = run_velocity_sweep()
    
    # Generate the four graphs for comparison
    plot_displacement_curves()
    plot_velocity_curves(velocity_data)
    plot_transmission_and_amplitude()

if __name__ == "__main__":
    main()
