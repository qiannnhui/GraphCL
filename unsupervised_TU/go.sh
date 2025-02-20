#!/bin/bash -ex
AUG=dnodes
for DATASET in MUTAG PROTEINS ENZYMES MSRC_21
do
  for mode in normal cheated rm_FN
  do
    for aug_ratio in 2
    do
      for seed in 0
      do
        # CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DATASET --lr 0.01 --local --num-gc-layers 5 --aug $AUG --seed $seed --aug_ratio $aug_ratio
        CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DATASET --lr 0.01 --local --num-gc-layers 5 --aug $AUG --seed $seed --aug_ratio $aug_ratio --mode $mode
      done
    done
  done
done
