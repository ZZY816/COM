



#!/usr/bin/env bash


cd ..
set -x
NGPUS=$1
PY_ARGS=${@:2}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

python3 -m torch.distributed.launch --nproc_per_node=${NGPUS} --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$PORT train.py --launcher pytorch --cfg_file cfgs/waymo_models/com/centercurriculum_pillar_ped_b2_com2.yaml

#


#set -x
#NGPUS=$1
#PY_ARGS=${@:2}
#
#while true
#do
#    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
#    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
#    if [ "${status}" != "0" ]; then
#        break;
#    fi
#done
#echo $PORT
#
#python -m torch.distributed.launch --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} train.py --cfg_file cfgs/waymo_models/centercurriculum_pillar_car_b2_com1.yaml
