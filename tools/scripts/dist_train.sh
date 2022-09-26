



#!/usr/bin/env bash


cd ..
set -x
NGPUS=$1
PY_ARGS=${@:2}

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

python3 -m torch.distributed.launch --nproc_per_node=${NGPUS} --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$PORT train.py --launcher pytorch ${PY_ARGS}

