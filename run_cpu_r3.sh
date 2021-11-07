#!/bin/bash


cd ti4r3s
sbatch jobcpu.sh
sbatch jobcpuimag.sh

cd ..
cd ti4r3u
sbatch jobcpu.sh
sbatch jobcpuimag.sh

cd ..
cd ti4r3t
sbatch jobcpu.sh
sbatch jobcpuimag.sh
