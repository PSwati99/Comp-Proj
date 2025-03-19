import numpy as np

# =============================================================================
# Material Constants (from Phys. Rev. B 86, 054445)
# =============================================================================
alpha = 0.01          # Gilbert damping
Ms = 8.6e5           # Saturation magnetization (A/m)
mu0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)
Nx_minus_Ny = 0.05   # Demagnetization factor difference
K_d = 0.5 * mu0 * Ms**2 * Nx_minus_Ny  # Effective anisotropy (J/m^3)
gamma = 2.21e5       # Gyromagnetic ratio (m/(A·s))
A_const = gamma * K_d / (mu0 * Ms)     # Effective constant (s^-1)
u_const = 10.0       # Effective magnon velocity (m/s)

# =============================================================================
# Simulation Time & Resolution
# =============================================================================
t_start = 0.0
t_end = 1e-6           # 1000 ns (Extended simulation)
num_t_points = 1000000  # 1,000,000 time steps
t_eval = np.linspace(t_start, t_end, num_t_points)

# =============================================================================
# Monte Carlo Sampling Parameters
# =============================================================================
monte_carlo_runs = 50

# =============================================================================
# Frequency Range
# =============================================================================
freq_min = 20
freq_max = 80
num_freq_points = 5001
