#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=tf_test
##BATCH --account=eecs545f20_class
#SBATCH --account=lsa2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6gb
#SBATCH --output=./output/slurm_%A.out
# The application(s) to execute along with its input arguments and options:

# Load modules
module load tensorflow/2.3.2
/bin/hostname

string=''

while getopts ":a:b:r:" opt; do
  case $opt in
    a) echo $OPTARG
      comp=$OPTARG
      ;;
    b)
      string=$OPTARG
      ;;
    r) im=$OPTARG
       echo $im
      python3 train_comp.py -nd 64 -lyr 6 -l mae -d 2 -nout 1 -e 2000 -b 100000 -real $im -load 0 -dir $string -n_train 5000000 -n_test 1000000 -comp $comp
      ;;

    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done
#python3 train.py -nd 256 -lyr 8 -l mae -d 2 -nout 22 -e 4000 -b 100000 -real 0 -load 0 -dir ti4r4s -n_train 5000000 -n_test 1000000
#python3 train.py -nd 128 -lyr 8 -l mae -d 2 -nout 22 -e 4000 -b 100000 -real 0 -load 0
#python3 PVC_Regression_TwoNN_test.py
