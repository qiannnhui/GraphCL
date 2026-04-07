#!/bin/bash -x
for DS in PROTEINS COLLAB REDDIT-MULTI-5K
do
    for MODE in PPR_infonce
    do
        CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $DS --lr 1e-4 --local --num-gc-layers 5 --aug dnodes --mode $MODE --log_interval 1 --epochs 200
    done
done
