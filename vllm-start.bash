#!/bin/bash

set -e
set -x

MODEL_NAME=$1
N_GPUS=$2
N_NODES=$3
MASTER_IP=$4
MOE=$5

env

rank=$OMPI_COMM_WORLD_RANK
if [[ -z "${rank}" ]] ; then
	rank=0
fi

OTHER_OPTIONS=""
if [[ $MODEL_NAME == MiniMaxAI* ]] ; then
#    OTHER_OPTIONS="  --tool-call-parser minimax_m2  --reasoning-parser minimax_m2   --compilation-config '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}'  --enable-auto-tool-choice "
    OTHER_OPTIONS="  --tool-call-parser minimax_m2  --reasoning-parser minimax_m2 --enable-auto-tool-choice "
fi
HEADLESS=""
if [ "$rank" -ne "0" ]; then
   HEADLESS="--headless"
fi
export NCCL_DEBUG=WARN
echo vllm serve $1 --tensor-parallel-size $N_GPUS --pipeline-parallel-size $N_NODES --nnodes $N_NODES --node-rank $rank --master-addr $MASTER_IP  --trust-remote-code --distributed-executor-backend "mp" --moe-backend $MOE $OTHER_OPTIONS $HEADLESS
NCCL_SOCKET_IFNAME=eno8303 NCCL_DEBUG=WARN vllm serve $1 --tensor-parallel-size $N_GPUS --pipeline-parallel-size $N_NODES --nnodes $N_NODES --node-rank $rank --master-addr $MASTER_IP  --trust-remote-code --distributed-executor-backend "mp" --moe-backend $MOE $OTHER_OPTIONS $HEADLESS



