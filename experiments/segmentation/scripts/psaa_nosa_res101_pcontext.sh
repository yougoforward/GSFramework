# !/usr/bin/env bash
 train
 python -m experiments.segmentation.train --dataset pcontext \
     --model psaa_nosa --aux --dilated --base-size 520 --crop-size 520 \
     --backbone resnet101 --checkname psaa_nosa_res101_pcontext

#test [single-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model psaa_nosa --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/psaa_nosa/psaa_nosa_res101_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model psaa_nosa --aux --dilated --base-size 576 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/psaa_nosa/psaa_nosa_res101_pcontext/model_best.pth.tar --split val --mode testval --ms