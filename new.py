import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define physical parameters (SI units)
mu0 = 4 * np.pi * 1e-7      # Vacuum permeability, H/m
Ms = 8.6e5                  # Saturation magnetization, A/m
alpha = 0.01                # Gilbert damping constant
gamma = 2.21e5              # Gyromagnetic ratio, m/(A·s) (approximate)
Nx_minus_Ny = 0.05          # Difference in demagnetizing factors

# Effective anisotropy for the Bloch wall:
Kd = (Nx_minus_Ny) / (2 * mu0 * Ms**2)

# Effective spin-wave parameter u.
# u is chosen such that for T = 0 the wall is pushed forward (positive X)
# and for T = 1 the wall is pulled backward (negative X).
u = 0.4  # effective spin velocity in m/s (adjust as needed)

# Precompute denominator (1+alpha^2)
denom = 1 + alpha**2
A = gamma * (Kd / (mu0 * Ms))  # factor multiplying sin(2phi)

def wall_dynamics(t, y, T):
    """
    y[0] = X (wall position)
    y[1] = phi (tilt angle in radians)
    
    T: transmission coefficient (T ~ 0: nearly full reflection; T ~ 1: nearly full transmission)
    """
    X, phi = y
    dX_dt = (1/denom) * ( A * np.sin(2 * phi) - T * u + (1 - T) * alpha * u )
    dphi_dt = (1/denom) * ( -A * np.sin(2 * phi) + (1 - T) * u + T * alpha * u )
    return [dX_dt, dphi_dt]

# Time span for simulation: 0 to 50 ns
t_start = 0
t_end = 50e-9  # 50 ns
t_eval = np.linspace(t_start, t_end, 1000)

# Initial conditions: wall at X = 0 and phi = pi/2
y0 = [0, np.pi/2]

# Case 1: Low-frequency (20 GHz) with nearly full reflection (T ~ 0).
T_low = 0.0  
sol_low = solve_ivp(wall_dynamics, [t_start, t_end], y0, args=(T_low,), t_eval=t_eval)

# Case 2: High-frequency (70 GHz) with nearly full transmission (T ~ 1).
T_high = 1.0  
sol_high = solve_ivp(wall_dynamics, [t_start, t_end], y0, args=(T_high,), t_eval=t_eval)

# Plot the wall displacement X(t) for both cases.
# Convert time to nanoseconds and displacement to nanometers.
plt.figure(figsize=(8, 5))
plt.plot(sol_low.t * 1e9, sol_low.y[0] * 1e9, label='f ≈ 20 GHz (T = 0)', color='blue')
plt.plot(sol_high.t * 1e9, sol_high.y[0] * 1e9, label='f = 70 GHz (T = 1)', color='red', linestyle='--')
plt.xlabel('Time (ns)')
plt.ylabel('Wall Displacement X (nm)')
plt.title('Domain Wall Displacement vs. Time')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure in PNG format
plt.savefig("domain_wall_displacement.png", dpi=300)
plt.show()

