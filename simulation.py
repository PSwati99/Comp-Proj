import numpy as np
from scipy.integrate import solve_ivp
from config import A_const, u_const, alpha, num_t_points, t_eval, monte_carlo_runs
from utils import transmission_coefficient

# =============================================================================
# Domain Wall ODE System
# =============================================================================
def domain_wall_ode(t, y, T, u):
    """
    ODE system for domain wall dynamics.
    y[0]: X(t) - Domain wall position (m)
    y[1]: φ(t) - Tilt angle (rad)
    """
    phi = y[1]
    dXdt = (A_const * np.sin(2*phi) - T*u + (1-T)*alpha*u) / (1 + alpha**2)
    dphidt = (-A_const * np.sin(2*phi) + (1-T)*u + T*alpha*u) / (1 + alpha**2)
    return [dXdt, dphidt]

# =============================================================================
# Monte Carlo Simulation for Stochastic Effects
# =============================================================================
def run_monte_carlo(f_GHz, runs=monte_carlo_runs):
    """
    Runs Monte Carlo simulations for a given frequency with random noise.
    Returns averaged displacement vs. time.
    """
    T = transmission_coefficient(f_GHz)
    u = u_const
    results = np.zeros(num_t_points)
    
    for _ in range(runs):
        y0 = [0.0, np.pi/2 + np.random.normal(0, 0.01)]  # Add small randomness
        sol = solve_ivp(domain_wall_ode, [t_eval[0], t_eval[-1]], y0, args=(T, u),
                        t_eval=t_eval, rtol=1e-10, atol=1e-12)
        results += sol.y[0]
    
    return t_eval, results / runs  # Return averaged displacement over runs
