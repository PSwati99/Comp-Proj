#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
alpha = 0.01
Ms = 8.6e5
mu0 = 4.0*np.pi*1e-7
Nx_minus_Ny = 0.05
K_d = 0.5 * mu0 * (Ms**2) * Nx_minus_Ny
gamma = 2.21e5  # m/(A*s)
A_const = (gamma * K_d) / (mu0 * Ms)

# Reflection parameters
k_factor = 1.0e7  # Reduce this
u0 = 0.1  # Reduce this for reasonable displacement

# Decay length
lambda_decay = 1.0e-7  # Reduce this

# Time
t_start = 0.0
t_end = 1.0e-7  # 100 ns
num_pts = 1000
t_eval = np.linspace(t_start, t_end, num_pts)

def transmission_coefficient(f_GHz):
    """T=0.5 for 22 GHz, T=1.0 for 70 GHz."""
    return 0.5 if f_GHz == 22 else 1.0

def local_amplitude(x):
    """Amplitude decays after a certain distance."""
    return 1.0 if x < 0 else np.exp(-x/lambda_decay)

def domain_wall_ode(t, y, f_GHz):
    """Domain wall motion ODE from paper."""
    X, phi = y

    T_val = transmission_coefficient(f_GHz)
    amp_here = local_amplitude(X)
    u_eff = u0 * amp_here

    Xdot = (
        A_const*np.sin(2.0*phi)
        - T_val*u_eff
        + (1.0 - T_val)*alpha*u_eff*k_factor
    ) / (1.0 + alpha**2)

    phidot = (
        - alpha*A_const*np.sin(2.0*phi)
        + (1.0 - T_val)*u_eff*k_factor
        + T_val*alpha*u_eff
    ) / (1.0 + alpha**2)

    return [Xdot, phidot]

def run_simulation():
    """Run the simulation and plot results."""
    freqs = [22, 70]
    plt.figure(figsize=(8,6))

    for f_GHz in freqs:
        y0 = [0.0, np.pi/4]  # Smaller initial tilt
        sol = solve_ivp(domain_wall_ode, [t_start, t_end], y0, args=(f_GHz,),
                        t_eval=t_eval, rtol=1e-9, atol=1e-11)
        
        X_nm = sol.y[0] * 1e9
        plt.plot(sol.t * 1e9, X_nm, label=f"{f_GHz} GHz")

    plt.xlabel("Time (ns)")
    plt.ylabel("Domain Wall Displacement (nm)")
    plt.title("Domain Wall Displacement vs. Time")
    plt.grid(True)
    plt.legend()
    plt.savefig("3")
    plt.show()

if __name__ == "__main__":
    run_simulation()

