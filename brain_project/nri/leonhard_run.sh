#!/bin/bash

# BSUB -W 24:00
# BSUB -n 6
# BSUB -R "rusage[mem=8192]"
# BSUB -R "rusage[ngpus_excl_p=1]"
# BSUB -o ./log_nri.%J.out
# BSUB -e ./log_nri.%J.err
# BSUB -J nri

module load python_gpu/3.6.4

python train.py
