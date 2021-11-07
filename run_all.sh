#!/bin/bash

cd ti4r1s
sbatch jobcpu.sh
cd ..
cd ti4r1u
sbatch jobcpu.sh
cd ..
cd ti4r1t
sbatch jobcpu.sh
cd ..
cd ti4r2s
sbatch jobcpu.sh
cd ..
cd ti4r2t
sbatch jobcpu.sh
cd ..
cd ti4r2u
sbatch jobcpu.sh

cd ..

cd ti4r1s
sbatch job.sh
cd ..
cd ti4r1u
sbatch job.sh
cd ..
cd ti4r1t
sbatch job.sh
cd ..
cd ti4r2s
sbatch job.sh
cd ..
cd ti4r2t
sbatch job.sh
cd ..
cd ti4r2u
sbatch job.sh

