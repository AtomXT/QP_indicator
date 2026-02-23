#!/bin/bash
#SBATCH --account=p32811  ## YOUR ACCOUNT pXXXX or bXXXX
#SBATCH --partition=normal  ### PARTITION (buyin, short, normal, etc)
#SBATCH --nodes=1 ## how many computers do you need
#SBATCH --ntasks-per-node=8 ## how many cpus or processors do you need on each >
#SBATCH --time=10:00:00 ## how long does this need to run (remember different p>
#SBATCH --mem=8G ## how much RAM do you need per CPU, also see --mem=<XX>G for >
#SBATCH --job-name=job5  ## When you run squeue -u NETID this is how you ca>
#SBATCH --output=quest_jobs/outlog/Q_sparsity_job5_log ## standard out and standar>
#SBATCH --mail-type=ALL ## you can receive e-mail alerts from SLURM when your j>
#SBATCH --mail-user=tongxu2027@u.northwestern.edu ## your email

module purge all
module load python-miniconda3
source activate python39
module load gurobi

python3 -m experiments.Run_sparse_to_dense_Q \
  --n_list "100,500,1000" \
  --delta_list "0.01,0.1,0.5" \
  --tau_list "0.05,0.1,0.2" \
  --timelimit "600" \
  --rep_list "4" \
  --job_name "job5"

