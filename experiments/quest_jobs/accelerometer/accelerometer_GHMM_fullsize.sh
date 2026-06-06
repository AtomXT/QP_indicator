#!/bin/bash
#SBATCH --account=p32811  ## YOUR ACCOUNT pXXXX or bXXXX
#SBATCH --partition=short  ### PARTITION (buyin, short, normal, etc)
#SBATCH --nodes=1 ## how many computers do you need
#SBATCH --ntasks-per-node=8 ## how many cpus or processors do you need on each >
#SBATCH --time=03:00:00 ## how long does this need to run (remember different p>
#SBATCH --mem=12G ## how much RAM do you need per CPU, also see --mem=<XX>G for >
#SBATCH --job-name=accelerometer_GHMM_fullsize  ## When you run squeue -u NETID this is how you ca>
#SBATCH --output=quest_jobs/outlog/accelerometer_GHMM_fullsize_log ## standard out and standar>
#SBATCH --mail-type=ALL ## you can receive e-mail alerts from SLURM when your j>
#SBATCH --mail-user=tongxu2027@u.northwestern.edu ## your email

module purge all
module load python-miniconda3
source activate python39
module load gurobi

python3 -m experiments.Run_accelerometer_GHMM \
  --gamma "400" \
  --lambda_outlier "100" \
  --sigma2 "2" \
  --nu "1" \
  --formulations "opt,tree" \
  --inference_modes "robust,nonrobust" \
  --timelimit "3600" \
  --threads "8" \
  --job_name "accelerometer_fullsize"
