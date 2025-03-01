#!/bin/bash -ex
AUG=dnodes
DATASET=MUTAG
aug_ratio=2
seed=0
CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DATASET --lr 1e-4 --local --num-gc-layers 5 --aug $AUG --seed $seed --aug_ratio $aug_ratio --batch-size 188
