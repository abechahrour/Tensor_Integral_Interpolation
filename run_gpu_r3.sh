#!/bin/bash


cd ti4r3s
sbatch job.sh
sbatch jobimag.sh

cd ..
cd ti4r3u
sbatch job.sh
sbatch jobimag.sh

cd ..
cd ti4r3t
sbatch job.sh
sbatch jobimag.sh
