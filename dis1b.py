import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

alpha = 0.01
beta  = 150.0   # Reflection boost factor (tune as needed)
u_const = 5.0
A_const = 1e7

t_end = 2e-8
t_eval = np.linspace(0, t_end, 200000)

def transmission_coefficient(f_GHz):
    # 0.5 if f<55, else 1.0
    return 0.5 if f_GHz < 55 else 1.0

def domain_wall_ode(t, y, T, u):
    phi = y[1]
    # Reflection term uses alpha*beta in Xdot, and beta in phidot
    dXdt = (A_const*np.sin(2*phi) - T*u + (1-T)*alpha*beta*u)/(1+alpha**2)
    dphidt= (-A_const*np.sin(2*phi) + (1-T)*beta*u + T*alpha*u)/(1+alpha**2)
    return [dXdt, dphidt]

plt.figure()
for freq in [22, 70]:
    T = transmission_coefficient(freq)
    sol = solve_ivp(domain_wall_ode, [0, t_end], [0.0, np.pi/2],
                    args=(T, u_const), t_eval=t_eval)
    X_nm = sol.y[0]*1e9
    plt.plot(sol.t*1e9, X_nm, label=f'{freq} GHz')

plt.xlabel('Time (ns)')
plt.ylabel('Displacement (nm)')
plt.title('Domain Wall Displacement vs. Time (with Reflection Boost)')
plt.legend()
plt.grid(True)
plt.savefig("test_displacement4.png")
plt.show()
