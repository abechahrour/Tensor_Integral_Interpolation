#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=example_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=3:00:00
#SBATCH --account=lsa1
#SBATCH --partition=standard
#SBATCH --output=./output/slurm_%A.out
# The application(s) to execute along with its input arguments and options:

# Load modules
module load tensorflow/2.3.2
/bin/hostname

python3 train.py -nd 32 -lyr 8 -l mae -d 2 -nout 7 -e 100 -b 100 -real 1 -load 0
#python3 PVC_Regression_TwoNN_test.py
