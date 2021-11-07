#!/bin/bash


cd ti4r4s
sbatch jobcpu.sh
sbatch jobcpuimag.sh

cd ..
cd ti4r4u
sbatch jobcpu.sh
sbatch jobcpuimag.sh

cd ..
cd ti4r4t
sbatch jobcpu.sh
sbatch jobcpuimag.sh
