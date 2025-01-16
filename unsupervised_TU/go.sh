#!/bin/bash -ex
AUG=dnodes
DATASET=MUTAG
for aug_ratio in 1 2 3 4 5 6 7 8 9
do
  for seed in 0 
  do
    CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DATASET --lr 0.01 --local --num-gc-layers 3 --aug $AUG --seed $seed --aug_ratio $aug_ratio --or_loss

  done
done
