#!/bin/bash -x
# for DS in MUTAG DD PROTEINS IMDB-BINARY NCI1
# do
#   for AUG in none dnodes random3
#   do
#     for MODE in normal cheated rm_FN rm_FP
#     do
#         CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DS --lr 1e-4 --local --num-gc-layers 5 --aug $AUG --mode $MODE --log_interval 10 --epochs 200 --plot_kde
#     done
#   done
# done
for DS in MUTAG DD REDDIT-BINARY IMDB-BINARY NCI1 PROTEINS COLLAB REDDIT-MULTI-5K
do
  for AUG in dnodes
  do
    # for MODE in cheated normal reweighted rm_FN rm_FP reweighted_l2 rm_FNs_by_ENs reweight_FNs_by_ENs reweight_FNs_by_RBO
    for MODE in reweight_FNs_by_RBO
    # for MODE in normal_rm_HNs normal_rm_ENs normal_HNs normal_ENs
    # for MODE in normal TP1_normal TP1_Nnormal TPs_TNs normal_TNs TPs_normal FP1_FNs normal_FNs FP1_normal FPs_FNs FPs_normal TP1_TN2
    do
      for RBO_p in 0.980 0.981 0.982 0.983
      do
          # CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DS --lr 1e-4 --local --num-gc-layers 5 --aug $AUG --mode $MODE --log_interval 1 --epochs 20 --plot_kde --plot_theta_l2 --neg_include_self --plot_anchor_aug_pair_theta_per_epoch --do_hn_analysis --reweight_strategy 1-coverage --renormalization --RBO_p $RBO_p
          CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DS --lr 1e-4 --local --num-gc-layers 5 --aug $AUG --mode $MODE --log_interval 1 --epochs 20 --plot_kde --plot_theta_l2 --neg_include_self --plot_anchor_aug_pair_theta_per_epoch --do_hn_analysis --reweight_strategy 1-coverage --renormalization --RBO_anchor --RBO_p $RBO_p
          # CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DS --lr 1e-4 --local --num-gc-layers 5 --aug $AUG --mode $MODE --log_interval 1 --epochs 20 --plot_kde --plot_theta_l2 --plot_anchor_aug_pair_theta_per_epoch --do_hn_analysis --reweight_strategy 1-coverage --renormalization --RBO_anchor --RBO_p $RBO_p
          # CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DS --lr 1e-4 --local --num-gc-layers 5 --aug $AUG --mode $MODE --log_interval 1 --epochs 20 --plot_kde --plot_theta_l2 --neg_include_self --plot_anchor_aug_pair_theta_per_epoch --do_hn_analysis --reweight_strategy 1-coverage --renormalization --denominator_anchor --RBO_p $RBO_p
          # CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DS --lr 1e-4 --local --num-gc-layers 5 --aug $AUG --mode $MODE --log_interval 1 --epochs 20 --plot_kde --plot_theta_l2 --neg_include_self --plot_anchor_aug_pair_theta_per_epoch --do_hn_analysis --reweight_strategy 1-coverage --renormalization --RBO_anchor --denominator_anchor --RBO_p $RBO_p
      done
    done
          # CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DS --lr 1e-4 --local --num-gc-layers 5 --aug $AUG --mode normal --log_interval 1 --epochs 20 --plot_kde --plot_theta_l2 --neg_include_self --plot_anchor_aug_pair_theta_per_epoch --do_hn_analysis --denominator_anchor
          # CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DS --lr 1e-4 --local --num-gc-layers 5 --aug $AUG --mode normal --log_interval 1 --epochs 20 --plot_kde --plot_theta_l2 --neg_include_self --plot_anchor_aug_pair_theta_per_epoch --do_hn_analysis
  done
done
# # gpu = 0
# for DS in PROTEINS DD REDDIT-BINARY COLLAB
# do
#     for SIM_MEASURE in l2 cosine cosine+l2
#     do
#         CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DS --lr 1e-4 --local --num-gc-layers 5 --aug none --mode cheated --log_interval 10 --epochs 500 --plot_theta_l2 --similarity_measure $SIM_MEASURE
#     done
# done
# gpu = 1
# for DS in IMDB-BINARY REDDIT-MULTI-5K NCI1
# do
#     for SIM_MEASURE in l2 cosine cosine+l2
#     do
#         CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DS --lr 1e-4 --local --num-gc-layers 5 --aug none --mode cheated --log_interval 10 --epochs 500 --plot_theta_l2 --similarity_measure $SIM_MEASURE
#     done
# done

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