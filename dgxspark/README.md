 

### Preparation

chmod +x *.sh

installer.sh  -> this will configure a local venv
download_model.sh -> this will download the required model to a local `model` directory

### Slurm-based launcher series

```
slurm-vllm.sbatch
      |
      V
launch-vllm.sh
      |
      V
inference-network.sh

```

```
mkdir -p "$PWD/logs"

sbatch --time=0 --chdir="$PWD/logs" \
    "$PWD/dgxspark/slurm-vllm.sbatch"
```

Logs will shown in vllm-{JOBID} directory. 


### Manual launcher series
```
bash run_vllm.sh
```

logs will shown in vllm_{NODE_RANK}.txt

```
./stop_vllm.sh
```

### MPI with Slurm test

mpi-smoke.c
mpi-smoke.sbatch

