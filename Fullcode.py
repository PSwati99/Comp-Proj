#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpi4py import MPI

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
# Extended Simulation Parameters
# =============================================================================
t_start = 0.0
t_end = 1e-6        
num_t_points = 10000000   
t_eval = np.linspace(t_start, t_end, num_t_points)

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
def run_monte_carlo(f_GHz, runs=5000):
    """Runs Monte Carlo simulations for a given frequency with random noise."""
    T = transmission_coefficient(f_GHz)
    u = u_const
    results = np.zeros(num_t_points)
    
    for _ in range(runs):
        y0 = [0.0, np.pi/2 + np.random.normal(0, 0.01)]
        sol = solve_ivp(domain_wall_ode, [t_start, t_end], y0, args=(T, u),
                        t_eval=t_eval, rtol=1e-10, atol=1e-12)
        results += sol.y[0]
    
    return t_eval, results / runs  

# =============================================================================
# MPI Parallel Frequency Sweep for Velocity Curves
# =============================================================================
def parallel_frequency_sweep():
    """Uses MPI to compute domain wall velocities over a range of frequencies."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    frequencies = np.linspace(20, 80, 50000)
    freq_local = frequencies[rank::size]  

    local_results = []
    
    for f in freq_local:
        t, X_avg = run_monte_carlo(f, runs=5000)
        dXdt_initial = (X_avg[1] - X_avg[0]) / (t[1] - t[0])
        dXdt_steady = (X_avg[-1] - X_avg[-2]) / (t[-1] - t[-2])
        local_results.append((f, dXdt_initial, dXdt_steady))
    
    all_results = comm.gather(local_results, root=0)
    
    if rank == 0:
        all_results = [item for sublist in all_results for item in sublist]
        all_results.sort(key=lambda x: x[0])
        return np.array(all_results)
    else:
        return None

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
        t, X_avg = run_monte_carlo(f, runs=5000)
        plt.plot(t*1e9, X_avg, label=f'{f} GHz')
    plt.xlabel("Time (ns)")
    plt.ylabel("Domain Wall Displacement (m)")
    plt.title("Domain Wall Displacement vs. Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("graph_displacement.png")
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
    plt.title("Domain Wall Velocity vs. Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig("graph_velocity.png")
    plt.show()

def plot_transmission_and_amplitude():
    """Plot transmission coefficient and spin-wave amplitude vs. frequency."""
    frequencies = np.linspace(20, 80, 5001)
    T_vals = np.array([transmission_coefficient(f) for f in frequencies])
    amp_vals = np.array([spin_wave_amplitude(f) for f in frequencies])
    
    plt.figure(figsize=(8,6))
    plt.plot(frequencies, T_vals, 'k-', label="Transmission Coefficient T")
    plt.plot(frequencies, amp_vals, 'r-', label="Spin-Wave Amplitude ρ")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Value")
    plt.title("Transmission Coefficient and Spin-Wave Amplitude vs. Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig("graph_transmission.png")
    plt.show()

# =============================================================================
# Main Execution
# =============================================================================
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    velocity_data = parallel_frequency_sweep()
    
    if rank == 0:
        plot_displacement_curves()
        plot_velocity_curves(velocity_data)
        plot_transmission_and_amplitude()

if __name__ == "__main__":
    main()
