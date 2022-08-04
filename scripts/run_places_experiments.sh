#!/usr/bin/env bash
PROJ_ROOT=/media/tyler/Data/codes/Embedded-CL
export PYTHONPATH=${PROJ_ROOT}
source activate embedded_cl
cd ${PROJ_ROOT}

SAVE_DIR=/media/tyler/Data/codes/Embedded-CL/results/
FEATURES_DIR=/media/tyler/Data/codes/edge-cl/features
IN_MEMORY=0
NUM_WORKERS=8
DATASET=places
PERMUTATION_SEED=0
LR=0.0001
POOL='avg'
MODELS=('mobilenet_v3_small' 'mobilenet_v3_large' 'efficientnet_b0' 'efficientnet_b1' 'resnet18')

for CL_MODEL in ncm nb slda replay fine_tune ovr perceptron; do
  for DATA_ORDER in iid class_iid; do
    for ((i = 0; i < ${#MODELS[@]}; ++i)); do
      MODEL="${MODELS[i]}"

      echo "Model: ${MODEL}; Dataset: ${DATASET}; Pool: ${POOL}"
      CACHE=${FEATURES_DIR}/${DATASET}/supervised_${MODEL}_${DATASET}_${POOL}
      EXPT_NAME=streaming_${CL_MODEL}_LR_${LR}_${MODEL}_${DATASET}_${POOL}_${DATA_ORDER}_seed_${PERMUTATION_SEED}
      LOG_FILE=${SAVE_DIR}/logs/${EXPT_NAME}.log

      python -u streaming_places_experiment.py \
        --arch ${MODEL} \
        --cl_model ${CL_MODEL} \
        --dataset ${DATASET} \
        --h5_features_dir ${CACHE} \
        --expt_name ${EXPT_NAME} \
        --save_dir ${SAVE_DIR}${EXPT_NAME} \
        --data_ordering ${DATA_ORDER} \
        --dataset_in_memory ${IN_MEMORY} \
        --num_workers ${NUM_WORKERS} \
        --permutation_seed ${PERMUTATION_SEED} >${LOG_FILE}
    done
  done
done

for CL_MODEL in replay; do
  for DATA_ORDER in iid class_iid; do
    for ((i = 0; i < ${#MODELS[@]}; ++i)); do
      MODEL="${MODELS[i]}"

      echo "Model: ${MODEL}; Dataset: ${DATASET}; Pool: ${POOL}"
      CACHE=${FEATURES_DIR}/${DATASET}/supervised_${MODEL}_${DATASET}_${POOL}
      EXPT_NAME=streaming_${CL_MODEL}_2percls_LR_${LR}_${MODEL}_${DATASET}_${POOL}_${DATA_ORDER}_seed_${PERMUTATION_SEED}
      LOG_FILE=${SAVE_DIR}/logs/${EXPT_NAME}.log

      python -u streaming_places_experiment.py \
        --arch ${MODEL} \
        --cl_model ${CL_MODEL} \
        --dataset ${DATASET} \
        --lr ${LR} \
        --h5_features_dir ${CACHE} \
        --buffer_size 730 \
        --expt_name ${EXPT_NAME} \
        --save_dir ${SAVE_DIR}${EXPT_NAME} \
        --data_ordering ${DATA_ORDER} \
        --dataset_in_memory ${IN_MEMORY} \
        --num_workers ${NUM_WORKERS} \
        --permutation_seed ${PERMUTATION_SEED} >${LOG_FILE}
    done
  done
done
