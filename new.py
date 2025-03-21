import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------------
# 1) Physical and model parameters
# -----------------------------
mu0 = 4.0 * np.pi * 1e-7    # Vacuum permeability (H/m)
Ms = 8.6e5                  # Saturation magnetization (A/m)
alpha = 0.01                # Gilbert damping
gamma = 2.21e5              # Gyromagnetic ratio (m / (A·s)) approx
Nx_minus_Ny = 0.05          # Difference in demagnetizing factors
# Effective anisotropy for the Bloch wall
Kd = (Nx_minus_Ny) / (2.0 * mu0 * Ms**2)

# For simplicity, we use the same "u" for both frequencies,
# but we vary the transmission T to control the sign of the velocity.
u_value = 1000.0  # (m/s) "effective" spin-wave velocity parameter

# Precompute helpful factors
denom = 1.0 + alpha**2
A = gamma * (Kd / (mu0 * Ms))  # multiplies sin(2phi)

# -----------------------------
# 2) Define the ODEs
# -----------------------------
def wall_equations(t, y, T, u):
    """
    y[0] = X(t)   (domain wall position)
    y[1] = phi(t) (tilt angle in radians)
    T     = transmission coefficient
    u     = spin-wave velocity parameter
    """
    X, phi = y
    
    # dX/dt  = [ A sin(2 phi ) - T*u + (1 - T)*alpha*u ] / (1 + alpha^2)
    # dphi/dt= [ -A sin(2 phi ) + (1 - T)*u + T*alpha*u ] / (1 + alpha^2)
    dX_dt = ( A * np.sin(2*phi) - T*u + (1.0 - T)*alpha*u ) / denom
    dphi_dt = ( -A * np.sin(2*phi) + (1.0 - T)*u + T*alpha*u ) / denom
    return [dX_dt, dphi_dt]

# -----------------------------
# 3) Simulation setup
# -----------------------------
# We simulate from t=0 to t=5 ns so we can see the transient, curved behavior.
t_start = 0.0
t_end   = 5e-9  # 5 ns
t_eval  = np.linspace(t_start, t_end, 1000)

# Initial conditions: X=0, phi=pi/2
y0 = [0.0, np.pi/2]

# -----------------------------
# 4) Solve for two "frequencies"
#    We mimic "f=20 GHz" by picking T=0.3 (mostly reflection => positive motion)
#    We mimic "f=70 GHz" by picking T=0.8 (mostly transmission => negative motion)
# -----------------------------
T_20GHz = 0.3   # partial reflection, net positive velocity
T_70GHz = 0.8   # mostly transmission, net negative velocity

sol_20 = solve_ivp(
    wall_equations, [t_start, t_end], y0, 
    args=(T_20GHz, u_value), t_eval=t_eval
)

sol_70 = solve_ivp(
    wall_equations, [t_start, t_end], y0, 
    args=(T_70GHz, u_value), t_eval=t_eval
)

# Convert time (s -> ns) and position (m -> nm)
time_20 = sol_20.t * 1e9
X_20    = sol_20.y[0] * 1e9
time_70 = sol_70.t * 1e9
X_70    = sol_70.y[0] * 1e9

# -----------------------------
# 5) Plot results
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(time_20, X_20, 'b', label='f ~ 20 GHz (T=0.3)')
plt.plot(time_70, X_70, 'r--', label='f ~ 70 GHz (T=0.8)')

plt.xlabel('Time (ns)')
plt.ylabel('Wall Displacement, X (nm)')
plt.title('Domain Wall Displacement vs Time')
plt.grid(True)
plt.legend()

# Force y-axis from -60 to +1200 nm
plt.ylim([-60, 1200])

plt.tight_layout()
plt.savefig('domain_wall_displacement.png', dpi=300)
plt.show()

