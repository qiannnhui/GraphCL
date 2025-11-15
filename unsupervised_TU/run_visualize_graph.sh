#!/bin/bash -x
# # gpu: 1, tmux: 8
for DS in MUTAG DD IMDB-BINARY NCI1 REDDIT-BINARY PROTEINS REDDIT-MULTI-5K COLLAB
do
  for AUG in none dnodes random3
  do
    for MODE in cheated normal reweighted rm_FN rm_FP reweighted_l2
    do
        CUDA_VISIBLE_DEVICES=$1 python visualize_graph.py --DS $DS --lr 1e-4 --local --num-gc-layers 5 --aug $AUG --mode $MODE --log_interval 10 --epochs 200
    done
  done
done

# gpu: 0, tmux: 9
# for DS in REDDIT-BINARY REDDIT-MULTI-5K COLLAB
# do
#     CUDA_VISIBLE_DEVICES=1 python dagcl.py --DS $DS --num_epochs 200 --plot_kde --plot_2d_scatter # DAGCL
#     CUDA_VISIBLE_DEVICES=1 python dagcl.py --DS $DS --num_epochs 200 --shuffle_DBN --neg_aug --plot_kde --plot_2d_scatter # 看看 neg_aug 有没有效果
#     CUDA_VISIBLE_DEVICES=1 python dagcl.py --DS $DS --num_epochs 200 --shuffle_DBN --loss l2_reweighted --plot_kde --plot_2d_scatter # DAGCL
#     CUDA_VISIBLE_DEVICES=0 python dagcl.py --DS $DS --num_epochs 200 --shuffle_DBN --loss l2_reweighted --neg_aug --plot_kde --plot_2d_scatter # 看看 neg_aug 有没有效果
#     CUDA_VISIBLE_DEVICES=0 python dagcl.py --DS $DS --num_epochs 200 --shuffle_DBN --loss l2_reweighted --theta_loss --lambda_theta_loss -1.5 --plot_kde --plot_2d_scatter # DAGCL
#     CUDA_VISIBLE_DEVICES=0 python dagcl.py --DS $DS --num_epochs 200 --shuffle_DBN --loss l2_reweighted --theta_loss --lambda_theta_loss -1.5 --neg_aug --plot_kde --plot_2d_scatter # 看看 neg_aug 有没有效果
# done

