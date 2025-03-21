import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define constants
alpha = 0.01  # Damping coefficient
gamma = 1.76e11  # Gyromagnetic ratio (rad/(s.T))
Kd = 1e3  # Anisotropy constant (J/m^3)
mu0 = 4 * np.pi * 1e-7  # Permeability of free space (T.m/A)
Ms = 8e5  # Saturation magnetization (A/m)

# Driving strengths for different frequencies
u_22GHz = 50   # Positive displacement effect
u_70GHz = -30  # Negative displacement effect
T = 0.5  # Spin polarization factor

# Define the system of equations
def model_equations(t, y, u):
    X, phi = y
    sin_2phi = np.sin(2 * phi)

    # Model equations based on the given formulas
    dX_dt = (gamma * Kd * mu0 / Ms) * sin_2phi - T * u + (1 - T) * alpha * u
    dPhi_dt = (-gamma * alpha * Kd * mu0 / Ms) * sin_2phi + (1 - T) * u + (T * alpha * u)

    return [dX_dt / (1 + alpha**2), dPhi_dt / (1 + alpha**2)]

# Time range
t_start, t_end = 0, 50  # in nanoseconds
t_eval = np.linspace(t_start, t_end, 500)

# Initial conditions
y0 = [0, np.pi / 4]  # Initial displacement and angle

# Solve ODEs for 22 GHz (positive displacement)
sol_22GHz = solve_ivp(model_equations, [t_start, t_end], y0, args=(u_22GHz,), t_eval=t_eval)

# Solve ODEs for 70 GHz (negative displacement)
sol_70GHz = solve_ivp(model_equations, [t_start, t_end], y0, args=(u_70GHz,), t_eval=t_eval)

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(sol_22GHz.t, sol_22GHz.y[0], label='Model Calculation (22 GHz)', color='red', linewidth=2)
plt.plot(sol_70GHz.t, sol_70GHz.y[0], label='Model Calculation (70 GHz)', color='green', linestyle="dashed", linewidth=2)

# Labels and Title
plt.xlabel("Time (ns)")
plt.ylabel("Domain Wall Displacement (nm)")
plt.title("Domain Wall Displacement vs. Time")
plt.axhline(0, color='black', linewidth=1)  # Reference line at X=0
plt.legend()
plt.grid()

# Save the figure as a PNG file
plt.savefig("domain_wall_displacement.png", dpi=300)

# Show the plot
plt.show()

