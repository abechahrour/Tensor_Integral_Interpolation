#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=example_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1gb
#SBATCH --time=8:00:00
#SBATCH --account=lsa1
#SBATCH --partition=standard
#SBATCH --output=/home/chahrour/golem_NN/I640/output/slurm_%A.out
# The application(s) to execute along with its input arguments and options:

# Load modules
module load tensorflow/2.3.2
/bin/hostname

python3 I640.py -nd 32 -lyr 8 -l mse -d 2 -e 200 -b 32 -load 0
#python3 PVC_Regression_TwoNN_test.py
