#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=example_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=8:00:00
#SBATCH --account=lsa1
#SBATCH --partition=standard
#SBATCH --output=./output/slurm_%A.out
# The application(s) to execute along with its input arguments and options:

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
      python3 train_comp.py -nd 64 -lyr 6 -l mae -d 2 -nout 1 -e 100 -b 100 -real $im -load 0 -dir $string -n_train 5000000 -n_test 1000000 -comp $comp
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
echo $OPTARG



#python3 train_disc.py -nd 64 -lyr 3 -l mae -d 2 -nout 7 -e 100 -b 1000 -real 0 -load 0 -dir ti4r2s -n_train 5000000 -n_test 1000000


#python3 train.py -nd 32 -lyr 8 -l mae -d 2 -nout 22 -e 100 -b 100 -real 0 -load 0 -dir ti4r4s -n_train 5000000 -n_test 1000000




#python3 PVC_Regression_TwoNN_test.py
