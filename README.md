# TACC-vllm-jobscript
The slurm jobscript to launch VLLM on CUDA GPU enabled systems

This first version works on Vista.  Will expand it to Stampede3 next.

## Collect a four-node DGX inventory

`collect_dgx_info.py` requires Python 3.8+ on the collecting machine and each
target node, with **no pip dependencies**. Run it on Linux for a complete inventory.
It uses existing system utilities where available; missing commands and permission
errors are recorded in the report. It does not install software, change network
settings, start services, or run GPU workloads.

Inside a Slurm allocation containing your four DGX nodes:

```bash
python3 collect_dgx_info.py --transport slurm \
  --path '/path/to/models' --path '$SCRATCH' --output dgx-inventory.json
```

The script expands `SLURM_JOB_NODELIST` and launches one independent `srun` step
per node. Run it once from the batch script or allocation shell, not underneath
another multi-task `srun`. Request the GPUs you intend to inspect in your site's
allocation options. Exclusive steps may wait if other steps hold those resources;
the node timeout records this as a failure. Site-specific partitions, accounts,
and GPU requests are intentionally left to your allocation script.

Alternatively, use existing noninteractive SSH access:

```bash
python3 collect_dgx_info.py --transport ssh \
  --nodes dgx01 dgx02 dgx03 dgx04 \
  --path /path/to/models --output dgx-inventory.json
```

SSH uses your normal config and known-host verification with batch mode enabled.
Use SSH config aliases to select a username or jump host. The Python source is
sent over standard input; no remote script installation or shared directory is
required. The remote interpreter defaults to `python3`; select a deployment
virtual environment with `--python /path/to/venv/bin/python`. SSH sessions may
have a different environment from your interactive shell; load required modules
before Slurm collection or configure the remote environment as appropriate.

For a single-node inspection (including inside the intended container):

```bash
python3 collect_dgx_info.py --transport local --expected-nodes 1 --torch-check
```

`--torch-check` optionally imports installed PyTorch and queries CUDA availability,
device count, and its NCCL version. The default only reads package metadata.
Host Python package versions do not describe software inside an Apptainer/Docker
image. Run local collection inside that image as well if that is your deployment
environment. `nvidia-smi` visibility can differ from the CUDA-visible GPU count.

The output consists of `dgx-inventory.json` and `dgx-inventory.json.txt`:

- GPU models, UUIDs, driver versions, total/free memory in MiB, compute mode,
  MIG listing, GPU/NIC topology, and NVLink status.
- CPU/NUMA details, architecture, RAM, memory-lock and file limits, kernel modules.
- IP addresses/routes, NIC MTU/speed, RDMA firmware/port state, GIDs and netdev
  mapping, DNS resolution of every target from every node, and TCP listeners.
- CUDA compiler, container utility and Python package versions, library inventory,
  selected distributed-runtime environment variables, mounts, `/dev/shm`, and
  capacity/access checks for each `--path` (quote `$VARIABLE` for remote expansion).
- Cross-node GPU/driver, architecture and package differences, missing GPUs/RDMA,
  missing vLLM, duplicate hostnames, and incomplete node collection.

Raw command output, errors, and exit status are retained in JSON. The environment
allowlist excludes tokens and credentials; reports still contain hostnames,
addresses, paths, and hardware identifiers. Review them before sharing.

The default expected count is four. Exit status is `1` if a node fails collection
or the count differs, `2` for invalid arguments, and `0` for complete collection.
Missing optional utilities and deployment findings do not change exit status:
**successful collection is not a deployment readiness verdict**. DNS resolution
does not establish peer TCP/RDMA reachability. This script does not test NCCL
collectives, bandwidth, shared-file visibility, or whether a particular model
fits. Use the reported interfaces/topology to choose networking settings and
validate those separately before launching distributed inference.

Each probe defaults to a 15-second timeout and each remote node to 600 seconds;
adjust with `--command-timeout` and `--node-timeout`. Nodes are collected with at
most four concurrent workers. Results describe what the current user/allocation
can see at collection time.

Run the standard-library tests with:

```bash
python3 -m unittest discover -s tests -v
```
