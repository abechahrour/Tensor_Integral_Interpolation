#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=example_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5gb
#SBATCH --time=00:30:00
#SBATCH --account=lsa1
#SBATCH --partition=standard
#SBATCH --output=./output/slurm_%A.out
# The application(s) to execute along with its input arguments and options:

# Load modules
module load tensorflow/2.3.2
/bin/hostname

python3 train.py -nd 32 -lyr 8 -l mae -d 2 -nout 3 -e 200 -b 1000 -load 1
#python3 PVC_Regression_TwoNN_test.py
