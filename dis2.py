#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

###############################################################################
# Paper-like Material & Simulation Parameters
###############################################################################
alpha = 0.01
Ms = 8.6e5
mu0 = 4*np.pi*1e-7
Nx_minus_Ny = 0.05
K_d = 0.5*mu0*(Ms**2)*Nx_minus_Ny
gamma = 2.21e5  # (m/(A*s))
A_const = gamma*K_d/(mu0*Ms)   # from eq. in paper

# Reflection wave-vector factor k from the text discussion (1e8 m^-1)
k_factor = 1.0e8

# Spin-wave base velocity (m/s). We'll multiply by amplitude below.
u0 = 10.0

# For amplitude decay: A(x)=A0*exp(-x/lambda). Tweak as needed.
A0 = 1.0
lambda_decay = 1.0e-6  # 1 micron decay length

# Simulation time
t_start = 0.0
t_end = 1e-7   # 100 ns
num_steps = 200000
t_eval = np.linspace(t_start, t_end, num_steps)

###############################################################################
# Frequency-Dependent Transmission Coefficient (paper: T=0.5 if f<55, else 1.0)
###############################################################################
def transmission_coefficient(f_GHz):
    return 0.5 if f_GHz < 55 else 1.0

###############################################################################
# Position-Dependent Spin-Wave Amplitude
###############################################################################
def local_amplitude(x_m):
    """
    For x<0, assume amplitude is ~A0 (since the wall is near the source).
    For x>0, amplitude decays as exp(-x/lambda_decay).
    """
    if x_m < 0:
        return A0
    else:
        return A0 * np.exp(-x_m / lambda_decay)

###############################################################################
# ODE System: Eqs. (2) & (3) from Phys. Rev. B 86, 054445
###############################################################################
def domain_wall_ode(t, y, f_GHz):
    """
    y[0] = X(t) in meters
    y[1] = phi(t) in radians

    (1+alpha^2)*Xdot = A_const sin(2phi) - T * (u * A(x)) + (1-T)*alpha*(u * A(x))*k_factor
    (1+alpha^2)*phidot= -alpha*A_const sin(2phi) + (1-T)*(u * A(x))*k_factor + T*alpha*(u * A(x))
    """
    X = y[0]
    phi = y[1]

    T_val = transmission_coefficient(f_GHz)
    # local amplitude at domain wall's position
    A_loc = local_amplitude(X)
    # effective spin-wave velocity at this position
    u_eff = u0 * A_loc

    # eq (2)
    Xdot = (
        A_const*np.sin(2*phi)
        - T_val*u_eff
        + (1 - T_val)*alpha*u_eff*k_factor
    ) / (1 + alpha**2)

    # eq (3)
    phidot = (
        - alpha*A_const*np.sin(2*phi)
        + (1 - T_val)*u_eff*k_factor
        + T_val*alpha*u_eff
    ) / (1 + alpha**2)

    return [Xdot, phidot]

###############################################################################
# Solve & Plot for f=22 GHz (low freq) & f=70 GHz (high freq)
###############################################################################
def run_simulation(freq_list):
    plt.figure(figsize=(8,6))

    for freq in freq_list:
        # initial conditions: X(0)=0, phi(0)=pi/2
        y0 = [0.0, np.pi/2]

        sol = solve_ivp(domain_wall_ode, [t_start, t_end], y0,
                        args=(freq,),
                        t_eval=t_eval, rtol=1e-9, atol=1e-11)

        # convert X(t) from m to nm
        X_nm = sol.y[0]*1e9
        plt.plot(sol.t*1e9, X_nm, label=f'{freq} GHz')

    plt.xlabel('Time (ns)')
    plt.ylabel('Domain Wall Displacement (nm)')
    plt.title('Domain Wall Displacement vs. Time (Paper-like 1D Model)')
    plt.legend()
    plt.grid(True)
    plt.savefig("test_displacement5.png")
    plt.show()

if __name__ == '__main__':
    run_simulation([22, 70])
