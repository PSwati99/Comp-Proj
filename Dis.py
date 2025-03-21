#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

###############################################################################
# Paper-like Parameters
###############################################################################
alpha = 0.01         # Paper's damping
Ms = 8.6e5
mu0 = 4.0*np.pi*1e-7
Nx_minus_Ny = 0.05
K_d = 0.5 * mu0 * (Ms**2) * Nx_minus_Ny
gamma = 2.21e5       # m/(A*s)
A_const = (gamma*K_d)/(mu0*Ms)  # from eq. in paper

k_factor = 1.0e8     # reflection wave-vector factor
u0 = 10.0            # base spin-wave velocity (m/s)

# amplitude scale factor near the source
A0 = 1.0
# decay length for amplitude (500 nm in this example)
lambda_decay = 5.0e-7

# Simulation time
t_start = 0.0
t_end   = 1.0e-7  # 100 ns
num_pts = 200001
t_eval = np.linspace(t_start, t_end, num_pts)

###############################################################################
# Transmission Coefficient
###############################################################################
def transmission_coefficient(f_GHz):
    """Paper: T=0.5 for low freq (22 GHz), T=1.0 for high freq (70 GHz)."""
    if abs(f_GHz - 22) < 1.0:
        return 0.5
    elif abs(f_GHz - 70) < 1.0:
        return 1.0
    # fallback if you want other frequencies
    return 0.5 if f_GHz < 55 else 1.0

###############################################################################
# Position-Dependent Spin-Wave Amplitude
###############################################################################
def local_amplitude(x):
    """
    If x < 0, amplitude ~ A0 (wall near the wave source).
    If x >= 0, amplitude decays as A0*exp(-x/lambda_decay).
    """
    if x < 0:
        return A0
    else:
        return A0 * np.exp(-x/lambda_decay)

###############################################################################
# ODE System: Eqs. (2) & (3) from Phys. Rev. B 86, 054445
###############################################################################
def domain_wall_ode(t, y, f_GHz):
    """
    y[0] = X(t) in meters
    y[1] = phi(t) in radians

    (1 + alpha^2)*Xdot = A_const*sin(2phi) 
                         - T*(u_eff) 
                         + (1 - T)*alpha*(u_eff)*k_factor

    (1 + alpha^2)*phidot= -alpha*A_const*sin(2phi) 
                          + (1 - T)*(u_eff)*k_factor 
                          + T*alpha*(u_eff)

    where u_eff = u0 * local_amplitude(X).
    """
    X = y[0]
    phi = y[1]

    T_val = transmission_coefficient(f_GHz)
    # local spin-wave velocity
    amp_here = local_amplitude(X)
    u_eff = u0 * amp_here

    # eq. (2)
    Xdot = (
        A_const*np.sin(2.0*phi)
        - T_val*u_eff
        + (1.0 - T_val)*alpha*u_eff*k_factor
    ) / (1.0 + alpha**2)

    # eq. (3)
    phidot = (
        - alpha*A_const*np.sin(2.0*phi)
        + (1.0 - T_val)*u_eff*k_factor
        + T_val*alpha*u_eff
    ) / (1.0 + alpha**2)

    return [Xdot, phidot]

###############################################################################
# Main: Solve for f=22 GHz & f=70 GHz
###############################################################################
def run_simulation():
    freqs = [22, 70]
    plt.figure(figsize=(8,6))

    for f_GHz in freqs:
        y0 = [0.0, np.pi/2]  # initial X=0, phi=pi/2
        sol = solve_ivp(domain_wall_ode, [t_start, t_end], y0, 
                        args=(f_GHz,),
                        t_eval=t_eval, rtol=1e-9, atol=1e-11)
        
        # convert X(t) from m -> nm
        X_nm = sol.y[0] * 1e9
        plt.plot(sol.t*1e9, X_nm, label=f"{f_GHz} GHz")

    plt.xlabel("Time (ns)")
    plt.ylabel("Domain Wall Displacement (nm)")
    plt.title("Domain Wall Displacement vs. Time (Paper-like 1D Model)")
    plt.grid(True)
    plt.legend()
    plt.savefig("2.png")
    plt.show()

if __name__ == "__main__":
    run_simulation()

