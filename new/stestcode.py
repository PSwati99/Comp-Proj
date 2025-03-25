#!/usr/bin/env python3

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpi4py import MPI
import sys

# =============================================================================
# Global Material and Simulation Parameters
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
    ODE system for domain wall dynamics.
    
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
    For other frequencies, placeholder values are used.
    """
    if frequency == 22e9:
        T = 0.4      
        vg = 1000.0   
        u = 35.0      
    elif frequency == 70e9:
        T = 0.98      
        vg = 2000.0   
        u = 16.0      
    else:
        T = 0.7       
        vg = 1500.0   
        u = 25.0      
        print(f"Using placeholder T, vg, u for frequency: {frequency/1e9:.0f} GHz")
    k = 2 * np.pi * frequency / vg
    return T, vg, u, k

# =============================================================================
# Monte Carlo Simulation Function for Displacement Curves
# =============================================================================
def run_monte_carlo(frequency, runs=3):
    """
    Runs Monte Carlo simulations for a given frequency.
    
    Uses a reduced number of runs (default 3) for quick testing.
    Returns:
      t_eval: Time evaluation points
      X_avg: Averaged wall displacement curve
    """
    T, vg, u, k = get_frequency_params(frequency)
    t_span = (0, 50e-9)  
    t_eval = np.linspace(t_span[0], t_span[1], 50)  
    y0 = [0, np.pi/2]    
    results = np.zeros(len(t_eval))
    
    for _ in range(runs):
        
        y0_noise = [0, np.pi/2 + np.random.normal(0, 0.01)]
        sol = solve_ivp(lambda t, y: domain_wall_dynamics(t, y, alpha, T, u, k, Kd, Ms, gamma, mu0, delta),
                        t_span, y0_noise, method='RK45', t_eval=t_eval, rtol=1e-6, atol=1e-8)
        results += sol.y[0]
    
    return t_eval, results / runs

# =============================================================================
# Analytical Velocity Functions
# =============================================================================
def get_velocity_analytical(T, u, k):
    """
    Computes the analytical initial (vi) and steady-state (vs) wall velocities.
      vi = (-T/(1+α²))*u + ((1-T)*α*k/(1+α²))*u
      vs = ((1-T)*k/α)*u
    """
    vi = (-T / (1 + alpha**2)) * u + ((1 - T) * alpha * k / (1 + alpha**2)) * u
    vs = ((1 - T) * k / alpha) * u
    return vi, vs

# =============================================================================
# MPI Parallel Frequency Sweep for Velocity Calculations
# =============================================================================
def parallel_frequency_sweep():
    """
    Uses MPI to perform a frequency sweep.
    For each frequency, computes analytical initial and steady-state velocities.
    Gathers the results on rank 0.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    For testing, use a reduced frequency range (20 GHz to 80 GHz with 100 points)
    frequencies = np.linspace(20e9, 80e9, 100)
    freq_local = frequencies[rank::size]
    local_results = []
    
    for freq in freq_local:
        T, vg, u, k = get_frequency_params(freq)
        vi, vs = get_velocity_analytical(T, u, k)
        local_results.append((freq, vi, vs))
    
    all_results = comm.gather(local_results, root=0)
    
    if rank == 0:
        
        all_results = [item for sublist in all_results for item in sublist]
        all_results.sort(key=lambda x: x[0])
        return np.array(all_results)
    else:
        return None

# =============================================================================
# Test Functions
# =============================================================================
def test_monte_carlo():
    
    frequency = 22e9
    t_eval, X_avg = run_monte_carlo(frequency, runs=3)
    assert t_eval.shape[0] == 50, "Expected 50 time evaluation points."
    assert X_avg.shape[0] == 50, "Expected 50 displacement points."
    print("Monte Carlo simulation test PASSED.")
    print(f"Time range: {t_eval[0]:.2e} to {t_eval[-1]:.2e} s")
    print(f"Sample displacement (nm): {X_avg[0]*1e9:.2f} nm")

def test_velocity_functions():
    
    frequency = 22e9
    T, vg, u, k = get_frequency_params(frequency)
    vi, vs = get_velocity_analytical(T, u, k)
    assert isinstance(vi, float), "Initial velocity (vi) should be a float."
    assert isinstance(vs, float), "Steady-state velocity (vs) should be a float."
    print("Velocity functions test PASSED.")
    print(f"At {frequency/1e9:.0f} GHz: vi = {vi:.2f} m/s, vs = {vs:.2f} m/s")

def test_mpi_sweep():
    
    velocity_data = parallel_frequency_sweep()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        assert velocity_data.ndim == 2 and velocity_data.shape[1] == 3, "Velocity data should have shape (N, 3)."
        print("MPI frequency sweep test PASSED.")
        print(f"Velocity sweep data shape: {velocity_data.shape}")

# =============================================================================
# Main Test Runner
# =============================================================================
def main():
    print("Starting system test for domain wall dynamics code...")
    test_monte_carlo()
    test_velocity_functions()
    test_mpi_sweep()
    print("All tests completed successfully.")
    

if __name__ == "__main__":
    main()
