#!/bin/bash -ex
# for DATASET in MUTAG PROTEINS ENZYMES MSRC_21
# CUDA_VISIBLE_DEVICES=0
for seed in 3
for seed in 3
do
  # for DATASET in COLLAB DD REDDIT-BINARY IMDB-BINARY NCI1
  # for DATASET in IMDB-BINARY REDDIT-BINARY REDDIT-MULTI-5K
  for DATASET in PROTEINS
  do
    # for mode in single_other_pos rm_FP rm_FN cheated pull_negative_rm_FN pull_negative
    # for mode in rm_FP
    for mode in pull_negative
    # for mode in single_other_pos
    do
      for AUG in none
      do
        # for lr in 1e-2 1e-3 1e-4 1e-5
        for lr in 1e-2
        do
          # CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DATASET --lr 0.01 --local --num-gc-layers 5 --aug $AUG --seed $seed --aug_ratio $aug_ratio
          CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DATASET --lr $lr --local --num-gc-layers 5 --aug $AUG --mode $mode --log_interval 10 --epochs 30 --seed $seed
          # CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DATASET --lr 0.01 --local --num-gc-layers 5 --aug $AUG --seed $seed --aug_ratio $aug_ratio --mode $mode --or_loss
        done
      done
    done
  done
done
# AUG=none
# for DATASET in MUTAG PROTEINS ENZYMES MSRC_21
# # for DATASET in MUTAG
# do
#   for mode in normal cheated rm_FN rm_FP
#   do
#     for aug_ratio in 2
#     do
#       for seed in 0
#       do
#         # CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DATASET --lr 0.01 --local --num-gc-layers 5 --aug $AUG --seed $seed --aug_ratio $aug_ratio
#         CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DATASET --lr 0.01 --local --num-gc-layers 5 --aug $AUG --seed $seed --aug_ratio $aug_ratio --mode $mode
#         # CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DATASET --lr 0.01 --local --num-gc-layers 5 --aug $AUG --seed $seed --aug_ratio $aug_ratio --mode $mode --or_loss
#       done
#     done
#   done
# done