#!/bin/bash

cd jobs

num=20
echo $num

for (( i = 0; i <= $num; i++ ))
do
    sbatch jobcpu_ti4r4s.sh
    sleep 1
done
for (( i = 0; i <= $num; i++ ))
do
    sbatch jobcpu_ti4r4u.sh
    sleep 1
done
for (( i = 0; i <= $num; i++ ))
do
    sbatch jobcpu_ti4r4t.sh
    sleep 1
done
for (( i = 0; i <= $num; i++ ))
do
    sbatch jobcpu_ti4r3s.sh
    sleep 1
done
for (( i = 0; i <= $num; i++ ))
do
    sbatch jobcpu_ti4r3u.sh
    sleep 1
done
for (( i = 0; i <= $num; i++ ))
do
    sbatch jobcpu_ti4r3t.sh
    sleep 1
done
for (( i = 0; i <= $num; i++ ))
do
    sbatch jobcpu_ti4r2s.sh
    sleep 1
done
for (( i = 0; i <= $num; i++ ))
do
    sbatch jobcpu_ti4r2u.sh
    sleep 1
done
for (( i = 0; i <= $num; i++ ))
do
    sbatch jobcpu_ti4r2t.sh
    sleep 1
done
