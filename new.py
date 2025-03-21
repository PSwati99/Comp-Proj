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

# Precompute common denominator and constant A
denom = 1 + alpha**2
A = gamma * (Kd / (mu0 * Ms))  # multiplies sin(2phi)

def wall_dynamics(t, y, T, u):
    """
    y[0] = X (wall position)
    y[1] = phi (tilt angle in radians)
    
    T: transmission coefficient (T ~ 0 for nearly full reflection; T ~ 1 for nearly full transmission)
    u: effective spin-wave parameter (different for each frequency case)
    """
    X, phi = y
    dX_dt = (1/denom) * ( A * np.sin(2 * phi) - T * u + (1 - T) * alpha * u )
    dphi_dt = (1/denom) * ( -A * np.sin(2 * phi) + (1 - T) * u + T * alpha * u )
    return [dX_dt, dphi_dt]

# Set simulation time: 0 to 500 ns
t_start = 0
t_end = 500e-9  # 500 ns
t_eval = np.linspace(t_start, t_end, 1000)

# Initial conditions: X = 0 and phi = pi/2
y0 = [0, np.pi/2]

# Case 1: Low-frequency (~20 GHz) with nearly full reflection, T ~ 0.
# We choose u_low so that the steady state velocity ~ (alpha*u_low) gives ~1200 nm over 500 ns.
#  Desired displacement: 1200e-9 m over 500e-9 s -> velocity ~ 2.4 m/s.
# Since for T = 0, steady state v = (alpha*u_low) (with alpha small), choose:
u_low = 240.0  # m/s, so alpha*u_low = 2.4 m/s
T_low = 0.0
sol_low = solve_ivp(wall_dynamics, [t_start, t_end], y0, args=(T_low, u_low), t_eval=t_eval)

# Case 2: High-frequency (70 GHz) with nearly full transmission, T ~ 1.
# For T = 1, the initial (and steady-state) velocity is about -u_high (since -u_high + alpha*u_high ~ -u_high)
# To have a displacement near -60 nm over 500 ns, we need v ≈ -60e-9/500e-9 = -0.12 m/s.
# So choose u_high ≈ 0.12 m/s.
u_high = 0.12  # m/s
T_high = 1.0
sol_high = solve_ivp(wall_dynamics, [t_start, t_end], y0, args=(T_high, u_high), t_eval=t_eval)

# Create the plot
plt.figure(figsize=(8, 5))
# Convert time to ns and displacement to nm.
plt.plot(sol_low.t * 1e9, sol_low.y[0] * 1e9, label='f ≈ 20 GHz (T = 0)', color='blue')
plt.plot(sol_high.t * 1e9, sol_high.y[0] * 1e9, label='f = 70 GHz (T = 1)', color='red', linestyle='--')

plt.xlabel('Time (ns)')
plt.ylabel('Wall Displacement X (nm)')
plt.title('Domain Wall Displacement vs. Time')
plt.legend()
plt.grid(True)

# Set y-axis limits: from -60 nm to 1200 nm
plt.ylim(-60, 1200)
plt.tight_layout()

# Save the figure as a PNG file
plt.savefig("domain_wall_displacement.png", dpi=300)
plt.show()

