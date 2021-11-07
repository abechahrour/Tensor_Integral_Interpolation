#!/bin/bash


cd ti4r1s
sbatch jobimag.sh
cd ..
cd ti4r1u
sbatch jobimag.sh
cd ..
cd ti4r1t
sbatch jobimag.sh
cd ..
cd ti4r2s
sbatch jobimag.sh
cd ..
cd ti4r2t
sbatch jobimag.sh
cd ..
cd ti4r2u
sbatch jobimag.sh

