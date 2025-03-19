from parallel_processing import parallel_frequency_sweep
from visualization import plot_displacement, plot_velocity, plot_transmission

def main():
    velocity_data = parallel_frequency_sweep()
    plot_displacement()
    plot_velocity(velocity_data)
    plot_transmission()

if __name__ == "__main__":
    main()
