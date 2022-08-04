#!/usr/bin/env bash
PROJ_ROOT=/media/tyler/Data/codes/Embedded-CL
export PYTHONPATH=${PROJ_ROOT}
source activate embedded_cl
cd ${PROJ_ROOT}

DATASET_PATH=/media/tyler/Data/datasets/Places-365/places365_standard
CACHE_PATH=/media/tyler/Data/codes/edge-cl/features
BATCH_SIZE=1024
POOL='avg'

MODELS=('resnet18' 'mobilenet_v3_small' 'mobilenet_v3_large' 'efficientnet_b0' 'efficientnet_b1')

for ((i = 0; i < ${#MODELS[@]}; ++i)); do
  MODEL="${MODELS[i]}"
  for DATASET in 'places_lt'; do

    echo "Model: ${MODEL}; Dataset: ${DATASET}; Pool: ${POOL}"
    CACHE=${CACHE_PATH}/${DATASET}/supervised_${MODEL}_${DATASET}_${POOL}
    EXPT_NAME=cache_supervised_${MODEL}_${DATASET}_${POOL}
    LOG_FILE=logs/${EXPT_NAME}.log

    exec > >(tee ${LOG_FILE}) 2>&1
    python -u cache_features.py \
      --arch ${MODEL} \
      --dataset ${DATASET} \
      --cache_h5_dir ${CACHE} \
      --images_dir ${DATASET_PATH} \
      --pooling_type ${POOL} \
      --batch_size ${BATCH_SIZE}
  done
done

DATASET_PATH=/media/tyler/Data/datasets/Places-365/places365_standard
CACHE_PATH=/media/tyler/Data/codes/edge-cl/features

MODELS=('resnet18' 'mobilenet_v3_small' 'mobilenet_v3_large' 'efficientnet_b0' 'efficientnet_b1')

for ((i = 0; i < ${#MODELS[@]}; ++i)); do
  MODEL="${MODELS[i]}"
  for DATASET in 'places'; do

    echo "Model: ${MODEL}; Dataset: ${DATASET}; Pool: ${POOL}"
    CACHE=${CACHE_PATH}/${DATASET}/supervised_${MODEL}_${DATASET}_${POOL}
    EXPT_NAME=cache_supervised_${MODEL}_${DATASET}_${POOL}
    LOG_FILE=logs/${EXPT_NAME}.log

    exec > >(tee ${LOG_FILE}) 2>&1
    python -u cache_features.py \
      --arch ${MODEL} \
      --dataset ${DATASET} \
      --cache_h5_dir ${CACHE} \
      --images_dir ${DATASET_PATH} \
      --pooling_type ${POOL} \
      --batch_size ${BATCH_SIZE}
  done
done
