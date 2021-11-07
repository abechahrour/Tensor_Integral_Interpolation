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
#SBATCH --output=/home/chahrour/golem_NN/ti3r1/output/slurm_%A.out
# The application(s) to execute along with its input arguments and options:

# Load modules
module load tensorflow/2.3.2
/bin/hostname

python3 train.py -nd 64 -lyr 8 -l mae -d 3 -nout 2 -e 200 -b 100 -load 0
#python3 PVC_Regression_TwoNN_test.py
