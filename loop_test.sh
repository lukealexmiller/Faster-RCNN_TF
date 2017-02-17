#!/bin/bash

# loop through checkpoints
for iter in {1..20}
do
  echo Calculting mAP for checkpoint: $[iter*1000]
  python ./tools/test_net.py --device GPU --device_id 0 --weights /root/faster_rcnn/output/BSKT_VGG16/voc_2007_trainval/VGGnet_fast_rcnn_iter_$[iter*1000].ckpt --imdb voc_2007_test --cfg experiments/cfgs/faster_rcnn_end2end.yml --network VGGnet_test --set EXP_DIR BSKT_VGG16 RNG_SEED 42 TRAIN.SCALES '[600]'
done
