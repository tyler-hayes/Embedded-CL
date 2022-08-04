#!/usr/bin/env bash
PROJ_ROOT=/media/tyler/Data/codes/Embedded-CL
export PYTHONPATH=${PROJ_ROOT}
source activate embedded_cl
cd ${PROJ_ROOT}/openloris

SAVE_DIR=/media/tyler/Data/codes/Embedded-CL/results/
IMAGES_DIR=/media/tyler/Data/datasets/OpenLORIS
DATASET=openloris
LR=0.001
DATA_ORDER=instance_small
POOL='avg'
STEP=1
MODELS=('mobilenet_v3_small' 'mobilenet_v3_large' 'efficientnet_b0' 'efficientnet_b1' 'resnet18')

for SEED in 10 20 30; do
  for CL_MODEL in ncm nb slda replay fine_tune ovr perceptron; do
    for ((i = 0; i < ${#MODELS[@]}; ++i)); do
      MODEL="${MODELS[i]}"

      echo "Model: ${MODEL}; CL Model: ${CL_MODEL}; Dataset: ${DATASET}; "
      EXPT_NAME=streaming_${CL_MODEL}_LR_${LR}_${MODEL}_${DATASET}_${POOL}_${DATA_ORDER}_seed_${SEED}
      LOG_FILE=${SAVE_DIR}/logs/${EXPT_NAME}.log

      python -u full_streaming_experiment.py \
        --arch ${MODEL} \
        --pooling_type ${POOL} \
        --images_dir ${IMAGES_DIR} \
        --model ${CL_MODEL} \
        --dataset ${DATASET} \
        --step ${STEP} \
        --lr ${LR} \
        --batch_size 64 \
        --save_dir ${SAVE_DIR}${EXPT_NAME} \
        --expt_name ${EXPT_NAME} \
        --order ${DATA_ORDER} \
        --seed ${SEED} >${LOG_FILE}
    done
  done

  for CL_MODEL in replay; do
    for ((i = 0; i < ${#MODELS[@]}; ++i)); do
      MODEL="${MODELS[i]}"

      echo "Model: ${MODEL}; CL Model: ${CL_MODEL}; Dataset: ${DATASET}; "
      EXPT_NAME=streaming_${CL_MODEL}_2percls_LR_${LR}_${MODEL}_${DATASET}_${POOL}_${DATA_ORDER}_seed_${SEED}
      LOG_FILE=${SAVE_DIR}/logs/${EXPT_NAME}.log

      python -u full_streaming_experiment.py \
        --arch ${MODEL} \
        --pooling_type ${POOL} \
        --images_dir ${IMAGES_DIR} \
        --model ${CL_MODEL} \
        --dataset ${DATASET} \
        --step ${STEP} \
        --lr ${LR} \
        --buffer_size 80 \
        --batch_size 64 \
        --save_dir ${SAVE_DIR}${EXPT_NAME} \
        --expt_name ${EXPT_NAME} \
        --order ${DATA_ORDER} \
        --seed ${SEED} >${LOG_FILE}
    done
  done
done
