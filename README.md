# TACC-vllm-jobscript
The slurm jobscript to launch VLLM on CUDA GPU enabled systems

This first version works on Vista.  Will expand it to Stampede3 next.

## Collect information on each DGX node

`collect_dgx_info.py` collects information from **only the node where it runs**.
Run the same script once on each of the four DGX nodes, then gather the reports
for deployment planning. It requires Python 3.8+ with **no pip dependencies**.
It does not use SSH or launch Slurm jobs/steps.

On each node:

```bash
python3 collect_dgx_info.py
```

Each run writes two files to the current directory:

- `dgx-inventory-<hostname>-<UTC timestamp>.json`: full structured inventory.
- `dgx-inventory-<hostname>-<UTC timestamp>.json.txt`: readable summary.

Hostnames and timestamps keep reports from different nodes and runs separate,
even when writing to a shared directory. Optionally inspect model/cache paths,
query installed PyTorch's CUDA/NCCL runtime, or choose an output filename:

```bash
python3 collect_dgx_info.py --path /path/to/models --path "$SCRATCH" --torch-check
python3 collect_dgx_info.py --output dgx01.json
```

An explicit `--output` overwrites that JSON file and its `.txt` companion; use a
different name per node when sharing an output directory. The output directory
must already exist.

Run in the environment intended for deployment, with required modules loaded and
GPUs allocated/visible. To inspect a virtual environment, invoke its Python:

```bash
/path/to/venv/bin/python collect_dgx_info.py
```

Host Python package versions do not describe software inside an Apptainer/Docker
image. Run the script inside the intended image too if needed. `--torch-check`
optionally imports PyTorch and queries CUDA availability, device count, and NCCL
version; the default reads package metadata without importing those packages.
`nvidia-smi` visibility can differ from the CUDA-visible GPU count.

The inventory includes:

- GPU models, UUIDs, driver versions, total/free memory in MiB, compute mode,
  MIG listing, GPU/NIC topology, and NVLink status.
- CPU/NUMA details, architecture, RAM, memory-lock and file limits, kernel modules.
- IP addresses/routes, NIC MTU/speed, RDMA firmware/port state, GIDs and netdev
  mapping, and TCP listeners.
- CUDA compiler, container utility and Python package versions, library inventory,
  selected distributed-runtime environment variables, mounts, `/dev/shm`, and
  capacity/access checks for each `--path`.
- Findings for missing GPUs, RDMA devices, or vLLM in the current Python environment.

Run on Linux for a complete inventory. Existing system utilities are optional;
missing commands, permission errors, and timeouts are recorded with raw command
output in JSON. Each command has a 15-second timeout, configurable with
`--command-timeout`. The script does not install software, change network
settings, start services, or run GPU workloads.

Exit status is `0` when reports are written, `1` if report writing fails, and `2`
for invalid arguments. Missing optional utilities and inventory findings do not
change exit status. Successful collection is not a deployment readiness verdict:
peer connectivity, NCCL collectives/bandwidth, shared-file visibility, cross-node
consistency, and model fit must be validated separately.

The environment allowlist excludes tokens and credentials. Reports contain
hostnames, addresses, paths, and hardware identifiers; review before sharing.
The JSON format is now schema version 2, containing a single `inventory` object
instead of the earlier multi-node `nodes` array.

Run the standard-library tests with:

```bash
python3 -m unittest discover -s tests -v
```
