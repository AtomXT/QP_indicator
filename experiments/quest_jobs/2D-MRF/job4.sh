#!/bin/bash
#SBATCH --account=p32811  ## YOUR ACCOUNT pXXXX or bXXXX
#SBATCH --partition=short  ### PARTITION (buyin, short, normal, etc)
#SBATCH --nodes=1 ## how many computers do you need
#SBATCH --ntasks-per-node=8 ## how many cpus or processors do you need on each >
#SBATCH --time=2:00:00 ## how long does this need to run (remember different p>
#SBATCH --mem=8G ## how much RAM do you need per CPU, also see --mem=<XX>G for >
#SBATCH --job-name=mrf_4  ## When you run squeue -u NETID this is how you ca>
#SBATCH --output=quest_jobs/outlog/2d_mrf_4 ## standard out and standar>
#SBATCH --mail-type=ALL ## you can receive e-mail alerts from SLURM when your j>
#SBATCH --mail-user=tongxu2027@u.northwestern.edu ## your email

module purge all
module load python-miniconda3
source activate python39
module load gurobi

python3 -m experiments.Run_2D_MRF \
  --grid_size_list "10,40,100" \
  --sigma2_list "0.1" \
  --rep_list "104" \
  --timelimit "3600" \
  --job_name "mrf_4"

