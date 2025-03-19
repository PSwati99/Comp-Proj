import numpy as np
from mpi4py import MPI
from config import freq_min, freq_max, num_freq_points
from simulation import run_monte_carlo

def parallel_frequency_sweep():
    """
    Uses MPI to compute domain wall velocities over a range of frequencies.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    frequencies = np.linspace(freq_min, freq_max, num_freq_points)
    freq_local = frequencies[rank::size]  # Distribute frequencies among MPI processes

    local_results = []
    
    for f in freq_local:
        t, X_avg = run_monte_carlo(f)
        dXdt_initial = (X_avg[1] - X_avg[0]) / (t[1] - t[0])
        dXdt_steady = (X_avg[-1] - X_avg[-2]) / (t[-1] - t[-2])
        local_results.append((f, dXdt_initial, dXdt_steady))
    
    all_results = comm.gather(local_results, root=0)
    
    if rank == 0:
        all_results = [item for sublist in all_results for item in sublist]
        all_results.sort(key=lambda x: x[0])
        return np.array(all_results)
    else:
        return None
