import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
Ms = 8.6e5           # Saturation magnetization (A/m)
Aex = 1.3e-11        # Exchange stiffness (J/m)
K1 = 5.8e5           # Perpendicular anisotropy constant (J/m³)
alpha = 0.01         # Gilbert damping constant
mu0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)
gamma = 2.21e5       # Gyromagnetic ratio (m/(A·s))
Delta = 19.1e-9 / np.pi  # Domain wall width parameter (m)
Kd = 0.5 * mu0 * Ms**2 * 0.05  # Effective anisotropy (J/m³)

# Parameters for each frequency
def get_params(frequency):
    if frequency == 22e9:
        T = 0.2      # Transmission coefficient
        k = 1e8      # Wave vector (m⁻¹)
        u = 0.3      # Magnonic current parameter
    elif frequency == 70e9:
        T = 0.95     # Transmission coefficient
        k = 3e8      # Wave vector (m⁻¹)
        u = 0.1      # Magnonic current parameter
    return T, k, u

# Define ODE system
def dw_motion(t, y, T, k, u):
    X, phi = y
    # Corrected line continuation
    dXdt_term1 = gamma * Delta * Kd / (mu0 * Ms) * np.sin(2 * phi)
    dXdt = (dXdt_term1 - T * u + (1 - T) * alpha * Delta * u * k) / (1 + alpha**2)
    
    dphidt = (-gamma * alpha * Kd / (mu0 * Ms) * np.sin(2 * phi) 
              + (1 - T) * u * k + T * alpha * u / Delta) / (1 + alpha**2)
    return [dXdt, dphidt]

# Solve and plot (fixed figure parameters)
plt.figure(figsize=(10, 6), dpi=300)  # Corrected parenthesis
colors = ['#1f77b4', '#d62728']
labels = ['22 GHz', '70 GHz']

for freq, color, label in zip([22e9, 70e9], colors, labels):
    T, k, u = get_params(freq)
    sol = solve_ivp(dw_motion, (0, 10e-9), [0, np.pi/2], 
                    args=(T, k, u), t_eval=np.linspace(0, 10e-9, 1000))
    plt.plot(sol.t * 1e9, sol.y[0] * 1e6, color=color, label=label, lw=2)

plt.xlabel('Time (ns)', fontsize=12)
plt.ylabel('Wall Displacement (nm)', fontsize=12)
plt.title('Domain Wall Motion Induced by Spin Waves', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.5)
plt.tight_layout()

plt.savefig('domain_wall_motion.png', bbox_inches='tight', dpi=300)
plt.close()
