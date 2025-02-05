#!/bin/bash

#SBATCH --job-name=defgen_finetuning
#SBATCH --account=project_465001386
#SBATCH --partition=standard-g    # 8 gpus at each node
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=8


source ${HOME}/.bashrc

export EBU_USER_PREFIX=/projappl/project_465001384/software/
# the important bit: unload all current modules (just in case) and load only the necessary ones
module --quiet purge
module load LUMI PyTorch/2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617


TRAIN_DATASET=${1}
VAL_DATASET=${2}
SAVE=${3}
BATCH_SIZE=${4}
MAX_TARGET_LENGTH=${5}
MODEL=${6}

echo ${TRAIN_DATASET}

srun singularity exec $SIF python3 finetune_model.py \
    --model_name_or_path ${MODEL} \  # /scratch/project_465001386/models/mt0-xl
    --do_train \
    --do_eval \
    --train_file ${TRAIN_DATASET} \
    --validation_file ${VAL_DATASET} \
    --output_dir ${SAVE} \
    --overwrite_output_dir \
    --eval_strategy=epoch \
    --logging_strategy=epoch \
    --per_device_train_batch_size=${BATCH_SIZE} \
    --per_device_eval_batch_size=${BATCH_SIZE} \
    --predict_with_generate \
    --save_total_limit=2 \
    --max_source_length=192 \
    --max_target_length=${MAX_TARGET_LENGTH} \
    --bf16=False \
    --num_train_epochs=20 \
    --save_strategy=epoch \
    --load_best_model_at_end=True \
    --ddp_find_unused_parameters=False \
    --optim=adafactor \
    --report_to=none \
    --metric_for_best_model=eval_rouge1 \
    --save_only_model=True \
    --generation_max_length=24 \
    --val_max_target_length=24
