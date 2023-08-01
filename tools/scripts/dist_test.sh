#!/usr/bin/env bash
cd ..
set -x
NGPUS=$1
PY_ARGS=${@:2}

python3 -m torch.distributed.launch --nproc_per_node=${NGPUS} test.py --launcher pytorch  --cfg_file cfgs/waymo_models/centerpoint_pillar_car_base_b2.yaml --ckpt /home/u1120220265/zzy/OpenPCDet-master/output/waymo_models/centerpoint_pillar_car_base_b2/default/ckpt/latest_model.pth


