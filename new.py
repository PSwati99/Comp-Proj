import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Physical constants
alpha = 0.01  # Gilbert damping
gamma = 1.76e11  # Gyromagnetic ratio (rad/s/T)
Ms = 8.0e5  # Saturation magnetization (A/m)
Hk = 50e3  # Anisotropy field (A/m)
u0 = 0.2  # Initial velocity factor

# Time settings
t_start, t_end = 0, 50e-9  # Simulate up to 50 ns
t_eval = np.linspace(t_start, t_end, 500)

def transmission_coefficient(f_GHz):
    """Returns the transmission coefficient T based on frequency."""
    return 1.0 if f_GHz == 22 else 0.5  # Higher for 22 GHz

def domain_wall_ode(t, y, f_GHz):
    """ODE function for domain wall motion."""
    x, phi = y
    Hx = 10  # External field
    k_factor = 5.0e6  # Reflection factor

    T_val = transmission_coefficient(f_GHz)
    velocity = u0 * T_val * np.cos(phi)

    dx_dt = velocity  # Displacement velocity
    dphi_dt = (-gamma * Hx * np.sin(2*phi) - alpha * velocity) / (1 + alpha**2)

    return [dx_dt, dphi_dt]

def run_simulation():
    """Run the simulation and plot results."""
    freqs = [22, 70]  # GHz
    plt.figure(figsize=(8,6))

    for f_GHz in freqs:
        y0 = [0.0, np.pi/4]
        sol = solve_ivp(domain_wall_ode, [t_start, t_end], y0, args=(f_GHz,),
                        t_eval=t_eval, rtol=1e-9, atol=1e-11)

        if len(sol.y[0]) == 0:
            print(f"Error: No solution for {f_GHz} GHz")
            continue  # Skip plotting if there's no data

        X_nm = sol.y[0] * 1e9  # Convert meters to nm
        print(f"Max displacement for {f_GHz} GHz: {max(X_nm):.2f} nm")

        plt.plot(sol.t * 1e9, X_nm, label=f"{f_GHz} GHz")

    plt.xlabel("Time (ns)")
    plt.ylabel("Domain Wall Displacement (nm)")
    plt.title("Domain Wall Displacement vs. Time (Paper-like 1D Model)")
    plt.grid(True)
    plt.legend()
    plt.savefig("4")
    plt.show()

# Run the simulation
run_simulation()

