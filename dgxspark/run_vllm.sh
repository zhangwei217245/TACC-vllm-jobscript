#!/bin/bash



IFACE=enp1s0f1np1
LOCAL_IP=$(ip -4 -o addr show dev ${IFACE} | head -n 1 | awk '{split($4,a,"/"); print a[1]}')
HEAD_IP=192.168.1.1

PROJECT=/opt/share/gits/Agentic/vllm

source $PROJECT/.venv/bin/activate

cd "$PROJECT"

export VLLM_HOST_IP=$LOCAL_IP
export NCCL_SOCKET_IFNAME="$IFACE"
export GLOO_SOCKET_IFNAME="$IFACE"
export NCCL_SOCKET_IFNAME='=enp1s0f1np1'
export NCCL_IB_HCA='=rocep1s0f1:1'
export NCCL_IP_PORT=8041
export NCCL_NET=Socket
export NCCL_NET=IB
export NCCL_IB_DISABLE=0

export NCCL_DEBUG=WARN
#export NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET,GRAPH


export NODE_RANK=$(hostname | sed 's/\./-/g'| awk -F '-' '{print int($3)-1}')

NUM_NODES=4
NUM_GPUS=$(nvidia-smi -L | wc -l)


#export MODEL_NAME=Inferact-Qwen3.8-Flash-Next-NVFP4
export MODEL_NAME=Qwen3-Coder-Next-FP8
export MODEL_REPO=$PROJECT/models
export MODEL_PATH=$MODEL_REPO/$MODEL_NAME
export MAX_MODEL_LEN=262144
export MAX_MODEL_LEN=32768
export GPU_MEM_UTILIZATION="0.6"
export MAX_NUM_SEQS=8
export LOAD_FORMAT=instanttensor
export MOE_BACKEND=b12x
export MOE_BACKEND=auto

export SERVICE_PORT=8040


if [ "$NODE_RANK" -eq 0 ]; then

	echo "Running Master on $(hostname)"
	nohup "$PROJECT/.venv/bin/vllm" serve "$MODEL_PATH" \
	    --served-model-name $MODEL_NAME \
	    --host 0.0.0.0 \
	    --port $SERVICE_PORT \
	    --tensor-parallel-size $NUM_GPUS \
	    --pipeline-parallel-size $NUM_NODES \
	    --distributed-executor-backend mp \
	    --nnodes $NUM_NODES \
	    --node-rank $NODE_RANK \
	    --master-addr "$HEAD_IP" \
	    --master-port $NCCL_IP_PORT \
	    --max-model-len $MAX_MODEL_LEN \
	    --gpu-memory-utilization "$GPU_MEM_UTILIZATION" \
	    --max-num-seqs $MAX_NUM_SEQS \
	    --load-format $LOAD_FORMAT \
	    --enable-prefix-caching \
	    --enable-auto-tool-choice \
	    --tool-call-parser qwen3_coder  > vllm_$NODE_RANK.txt 2>&1  &
else
	echo "Running Worker on $(hostname)"
	"$PROJECT/.venv/bin/vllm" serve "$MODEL_PATH" \
	    	--tensor-parallel-size $NUM_GPUS \
	    	--pipeline-parallel-size $NUM_NODES \
    		--max-model-len $MAX_MODEL_LEN \
    		--max-num-seqs $MAX_NUM_SEQS \
    		--enable-prefix-caching \
    		--distributed-executor-backend mp \
    		--nnodes $NUM_NODES \
    		--node-rank $NODE_RANK \
    		--master-addr "$HEAD_IP" \
    		--master-port=$NCCL_IP_PORT \
    		--gpu-memory-utilization $GPU_MEM_UTILIZATION \
    		--load-format=$LOAD_FORMAT \
    		--headless  > vllm_$NODE_RANK.txt 2>&1   &
fi



		#--moe-backend $MOE_BACKEND \
	        #--enforce-eager \
