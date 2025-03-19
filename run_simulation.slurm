#!/bin/bash
#SBATCH --job-name=domain_wall_sim       # Job name
#SBATCH --account=ISAAC-UTK0350          # Project account
#SBATCH --partition=condo-phys573        # Partition name
#SBATCH --qos=condo                      # Quality of Service
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=56                      # Total MPI tasks (56 cores per node)
#SBATCH --cpus-per-task=1                 # CPUs per MPI task
#SBATCH --time=48:00:00                   # Max wall time (48 hours)
#SBATCH --output=output_%j.log            # Standard output log file
#SBATCH --error=error_%j.log              # Standard error log file

# Load required modules
module purge
module load python
module load mpi4py

# Navigate to project directory
cd /path/to/your/project_directory

# Run the Python script with MPI
mpirun -np 112 python main.py
