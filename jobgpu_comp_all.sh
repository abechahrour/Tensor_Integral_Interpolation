#!/bin/bash


for (( i = 0; i < 3; i++ ))
do
    sbatch job_comp.sh -a $i -b ti4r1s -r 0
    sbatch job_comp.sh -a $i -b ti4r1s -r 1
done


for (( i = 0; i < 7; i++ ))
do
    sbatch job_comp.sh -a $i -b ti4r2s -r 0
    sbatch job_comp.sh -a $i -b ti4r2s -r 1
done

for (( i = 0; i < 13; i++ ))
do
    sbatch job_comp.sh -a $i -b ti4r3s -r 0
    sbatch job_comp.sh -a $i -b ti4r3s -r 1
done

for (( i = 0; i < 22; i++ ))
do
    sbatch job_comp.sh -a $i -b ti4r4s -r 0
    sbatch job_comp.sh -a $i -b ti4r4s -r 1
done

