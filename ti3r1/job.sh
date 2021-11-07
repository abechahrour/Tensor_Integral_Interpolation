#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=tf_test
##BATCH --account=eecs545f20_class
#SBATCH --account=lsa2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3gb
#SBATCH --output=/home/chahrour/golem_NN/ti3r1/output/slurm_%A.out
# The application(s) to execute along with its input arguments and options:

# Load modules
module load tensorflow/2.3.2
/bin/hostname

python3 train.py -nd 128 -lyr 8 -l mae -d 3 -nout 2 -e 5000 -b 100000 -load 0
#python3 PVC_Regression_TwoNN_test.py
