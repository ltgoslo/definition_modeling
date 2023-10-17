#!/bin/bash

#SBATCH --job-name=t5_definition
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
module load nlpl-accelerate/0.13.2-foss-2021a-Python-3.9.5
module load nlpl-scikit-bundle/1.1.1-foss-2021a-Python-3.9.5

MODEL=${1}  # Definition generation model (e.g. mt0-xl_english/)
SENT_MODEL=${2}  # Sentence embedding model sentence-transformers/distiluse-base-multilingual-cased-v1 will do for most languages
TEST=${3}   # Input file with examples and target words
DATA=${4}  # where to save an interim file with generated definitions
PROMPT=${5}  # What prompt to use? (see the list in generate_t5.py)
OUT=${6}  # Final file with cluster labels

echo ${MODEL}
echo ${SENT_MODEL}
echo ${TEST}
echo ${PROMPT}

echo "Start generating definitions..."
python3 modeling/generate_t5.py --model ${MODEL} --test ${TEST} --save ${DATA} --prompt ${PROMPT} --bsize 16 --maxl 256 --filter 1
echo "Generating definitions finished..."
echo ${DATA}
echo "Generating sense labels..."
python3 proto_explanations/sense_label.py --model ${SENT_MODEL} --data ${DATA} --bsize 16 --save text --output ${OUT}
echo "Generating sense labels finished"
echo ${OUT}