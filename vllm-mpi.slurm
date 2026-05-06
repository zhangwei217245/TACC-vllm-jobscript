#!/bin/bash
#SBATCH --job-name=vllm_multinode
#SBATCH --nodes=2
#SBATCH --partition=gh-dev
#SBATCH --time=1:00:00
#SBATCH --output=slurm_jobs/vllm_%j.log


set -x
set -e


# Validate input
MODEL_NAME=$1
if [ -z "$MODEL_NAME" ]; then
  echo "Usage: sbatch vllm_multinode.sh <huggingface_model_name>"
  exit 1
fi

# Setup environment
# WEI's comment: we should use apptainer for running this so we don't have to install a venv
# mkdir -p $SCRATCH/.uv
# export UV_PYTHON_BIN_DIR="$SCRATCH/.uv/python_bin_dir"
# export UV_PYTHON_CACHE_DIR="$SCRATCH/.uv/python_cache_dir"
# export UV_CACHE_DIR="$SCRATCH/.uv/cache"
# export UV_PYTHON_INSTALL_DIR="$SCRATCH/.uv/python_install_dir"
# export UV_TOOL_BIN_DIR="$SCRATCH/.uv/tool_bin_dir"
# export UV_TOOL_DIR="$SCRATCH/.uv/tool_dir"
# export UV_LINK_MODE="copy"

# Wei's comments: This is definitely needed, but at the same time, for FlexServ, we provide better model management
# and we use external dir for HF models
export HF_HOME="$SCRATCH/huggingface/"
mkdir -p $HF_HOME
export HF_DATASETS_CACHE="$SCRATCH/huggingface/"
export HF_TOKEN="<YOUR HUGGING FACE TOKEN>"

# Wei's comment: this is very good that we have the VLLM-particular configuration
export VLLM_CACHE_ROOT="$SCRATCH/huggingface/"
export VLLM_USE_PRECOMPILED=True



# Wei's comment: I believe this is more TACC-specific 
cluster=`hostname | awk -F . '{print $2}'`
# Wei's comment: this is not necessary for running vLLM in apptainer
# if [ "$cluster" -eq "vista" ] ; then
#        	module load nvidia/24.9  cuda/12.8  openmpi/5.0.7
# elif [ "$cluster" -eq "stampede3" ] ; then
# 	module load cuda/13.1 nvidia/26.1  openmpi/5.0.7
# fi
# source .venv/bin/activate

module load tacc-apptainer




# Wei's comment: useful for apptainer
export nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
export N_NODES=${#nodes[@]}
export N_GPUS=`nvidia-smi --list-gpus | wc -l`

# Configure distributed settings
export MASTER_ADDR=${nodes[0]}

export ib_dev=`ip -4 addr | grep ib | head -1 | awk -F : '{print $2}'`

export head_node_ip=`ip -4 addr show dev $ib_dev | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n1`

env

# Check for GPU on head node
if [ -e /dev/nvidia0 ]; then
  echo "GPU found on head node $MASTER_ADDR"
else
  echo "ERROR: No GPU detected on head node $MASTER_ADDR. Exiting."
  exit 1
fi

export NVLINK=`nvidia-smi nvlink --status | grep  Link  | wc -l`


echo $MASTER_ADDR
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=ib_dev
export NCCL_IB_HCA=mlx5_0
export TRITON_CACHE_DIR=/$SCRATCH/.triton/cache
export VLLM_CACHE_ROOT=/tmp/vllm-cache
#export NCCL_DEBUG=INFO



MOE=triton

if [[ ! -n "$MODEL_NAME" ]] ; then
       echo "Model name not given"
       exit 1;       
fi

if [[ ! -n "$N_GPUS" ]] ; then
       echo "nvidia-smi did not report GPUs"
       exit 1;       
fi

if [[ ! -n "$N_NODES" ]] ; then
       echo "N_NODES not found using scontrol"
       exit 1;       
fi

if [[ ! -n "$head_node_io" ]] ; then
       echo "Cannot find head node IB IP"
       exit 1;       
fi

if [[ ! -n "$MOE" ]] ; then
       echo "MOE not defined"
       exit 1;       
fi

echo ibrun ./vllm-start.bash $MODEL_NAME $N_GPUS $N_NODES  $head_node_ip  $MOE
ibrun ./vllm-start.bash $MODEL_NAME $N_GPUS $N_NODES  $head_node_ip  $MOE

#echo ./vllm-start.bash $MODEL_NAME $N_GPUS $N_NODES  $head_node_ip  $MOE
#./vllm-start.bash $MODEL_NAME $N_GPUS $N_NODES  $head_node_ip  $MOE
