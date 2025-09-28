# #!/bin/bash -ex
# AUG=dnodes
# DATASET=PROTEINS
# for aug_ratio in 1 2 3 4 5 6 7 8 9
# do
#   for seed in 0 1 2 3 4 
#   do
#     CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DATASET --lr 0.01 --local --num-gc-layers 3 --aug $AUG --seed $seed --aug_ratio $aug_ratio

#   done
# done

#!/bin/bash -ex
AUG=dnodes
aug_ratio=2
for DATASET in NCI1 PROTEINS REDDIT-BINARY REDDIT-MULTI-5K COLLAB
do
  for seed in {0..4}
  do
    CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DATASET --lr 0.001 --local --num-gc-layers 5 --aug $AUG --seed $seed --aug_ratio $aug_ratio --epoch 100

  done
done
