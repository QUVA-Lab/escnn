#!/usr/bin/env bash

#SBATCH --ntasks 20
#SBATCH --time 02:00:00
#SBATCH --partition short
#SBATCH --mem 8GB

set -eu

OUT_DIR=$(realpath ${1:-out})

mkdir $OUT_DIR

cd ~/sandbox/escnn

. venv/bin/activate

python test/slurm/run_tests.py $OUT_DIR
