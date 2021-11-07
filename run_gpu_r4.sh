#!/bin/bash


cd ti4r4s
sbatch job.sh
sbatch jobimag.sh

cd ..
cd ti4r4u
sbatch job.sh
sbatch jobimag.sh

cd ..
cd ti4r4t
sbatch job.sh
sbatch jobimag.sh
