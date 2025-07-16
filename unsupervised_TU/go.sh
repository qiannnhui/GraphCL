#!/bin/bash -ex
# # gpu = 0
# for DS in PROTEINS DD COLLAB REDDIT-BINARY
# do
#     for SIM_MEASURE in l2 cosine cosine+l2
#     do
#         CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DS --lr 1e-4 --local --num-gc-layers 5 --aug none --mode cheated --log_interval 10 --epochs 500 --plot_theta_l2 --similarity_measure $SIM_MEASURE
#     done
# done
# gpu = 1
for DS in MUTAG IMDB-BINARY REDDIT-MULTI-5K NCI1
do
    for SIM_MEASURE in l2 cosine cosine+l2
    do
        CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DS --lr 1e-4 --local --num-gc-layers 5 --aug none --mode cheated --log_interval 10 --epochs 500 --plot_theta_l2 --similarity_measure $SIM_MEASURE
    done
done

# for DATASET in PROTEINS
# for DATASET in MUTAG PROTEINS ENZYMES MSRC_21 DD COLLAB
# for DATASET in ENZYMES MSRC_21

# for DATASET in DD
# # for DATASET in MSRC_21 PROTEINS DD COLLAB REDDIT-BINARY IMDB-BINARY NCI1
# # for DATASET in IMDB-BINARY NCI1 COLLAB REDDIT-BINARY
# do
#   for mode in normal cheated rm_FN rm_FP
#   # for mode in normal rm_FN rm_FP cheated
#   do
#     for AUG in none
#     # for AUG in none dnodes random3
#     do
#       # for lr in 1e-2 1e-3 1e-4 1e-5
#       for lr in 1e-4 5e-5
#       do
#         # CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DATASET --lr 0.01 --local --num-gc-layers 5 --aug $AUG --seed $seed --aug_ratio $aug_ratio
#         CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DATASET --lr $lr --local --num-gc-layers 5 --aug $AUG --mode $mode --log_interval 10 --epochs 500
#         # CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DATASET --lr 0.01 --local --num-gc-layers 5 --aug $AUG --seed $seed --aug_ratio $aug_ratio --mode $mode --or_loss
#       done
#     done
#   done
# done


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