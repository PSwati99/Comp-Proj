#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Key parameters
alpha = 0.02
u_const = 1.0
A_const = 1e7  # artificially set for demonstration
t_end = 2e-8   # 20 ns
num_points = 200000
t_eval = np.linspace(0, t_end, num_points)

def transmission_coefficient(f_GHz):
    # Simple step function: 0.5 if f<55, else 1.0
    return 0.5 if f_GHz < 55 else 1.0

def domain_wall_ode(t, y, T, u):
    phi = y[1]
    dXdt = (A_const * np.sin(2*phi) - T*u + (1 - T)*alpha*u) / (1 + alpha**2)
    dphidt = (-A_const * np.sin(2*phi) + (1 - T)*u + T*alpha*u) / (1 + alpha**2)
    return [dXdt, dphidt]

# Simulate for f=22 GHz and f=70 GHz
frequencies = [22, 70]
plt.figure()
for f in frequencies:
    T = transmission_coefficient(f)
    y0 = [0.0, np.pi/2]  # no random tilt
    sol = solve_ivp(domain_wall_ode, [0, t_end], y0, args=(T, u_const),
                    t_eval=t_eval, rtol=1e-9, atol=1e-11)
    X = sol.y[0] * 1e9  # convert from m to nm
    plt.plot(sol.t*1e9, X, label=f'{f} GHz')

plt.xlabel('Time (ns)')
plt.ylabel('Displacement (nm)')
plt.title('Domain Wall Displacement vs. Time (Short Window)')
plt.legend()
plt.grid(True)
 plt.savefig("test_displacement3.png")
plt.show()
