#!/bin/bash

#SBATCH --job-name=flan_t5_finetuning
#SBATCH --account=ec30
#SBATCH --partition=accel    # To use the accelerator nodes
#SBATCH --gpus=2
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8

source ${HOME}/.bashrc

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-datasets/2.21.0-foss-2022b-Python-3.10.8
module load nlpl-accelerate/0.33.0-foss-2022b-Python-3.10.8
module load nlpl-sentencepiece/0.1.99-foss-2022b-Python-3.10.8
module load nlpl-nlptools/03-foss-2022b-Python-3.10.8
module load nlpl-transformers/4.43.4-foss-2022b-Python-3.10.8

MODEL=${1}
TRAIN_DATASET=${2}
VAL_DATASET=${3}
SAVE=${4}

echo ${MODEL}
echo ${TRAIN_DATASET}

export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1 
export TORCH_DISTRIBUTED_DEBUG=DETAIL

srun python3 finetune_model.py \
    --model_name_or_path ${MODEL} \
    --do_train \
    --do_eval \
    --train_file ${TRAIN_DATASET} \
    --validation_file ${VAL_DATASET} \
    --output_dir ${SAVE} \
    --overwrite_output_dir \
    --evaluation_strategy=epoch \
    --logging_strategy=epoch \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --save_total_limit=5 \
    --max_source_length=192 \
    --max_target_length=128 \
    --val_max_target_length=128 \
    --bf16=False \
    --num_train_epochs=2 \
    --save_strategy=epoch \
    --load_best_model_at_end=True \
    --metric_for_best_model=eval_rouge1 \
    --ddp_find_unused_parameters=False \
    --optim=adafactor \
    --report_to=none \
#    --source_prefix "summarize: "
