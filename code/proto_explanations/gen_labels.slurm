#!/bin/bash

#SBATCH --job-name=definition_labels
#SBATCH --account=ec30
#SBATCH --partition=accel    # To use the accelerator nodes
#SBATCH --gpus=1
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8

source ${HOME}/.bashrc

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-transformers/4.24.0-foss-2021a-Python-3.9.5
module load nlpl-sentencepiece/0.1.96-foss-2021a-Python-3.9.5
module load nlpl-scikit-bundle/1.1.1-foss-2021a-Python-3.9.5


MODEL=${1}  # sentence-transformers/distiluse-base-multilingual-cased-v1 will do for most languages
DATA=${2}  # tsv file with definitions, usages and clusters
OUT=${3}
echo ${MODEL}
echo ${DATA}

python3 sense_label.py --model ${MODEL} --data ${DATA} --bsize 16 --save text --output ${OUT}
