#!/usr/bin/env python3
"""Read-only per-node vLLM deployment inventory; Python 3.8+, no pip dependencies."""

import argparse
import csv
import datetime
import glob
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import signal
import socket
import subprocess
import sys


GPU_FIELDS = ["index", "name", "uuid", "pci.bus_id", "driver_version",
              "memory.total", "memory.free", "compute_mode"]
ENV_KEYS = """CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES CUDA_HOME
NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_DISABLE NCCL_NET_GDR_LEVEL
GLOO_SOCKET_IFNAME VLLM_HOST_IP MASTER_ADDR MASTER_PORT
HF_HOME HF_HUB_CACHE VLLM_CACHE_ROOT TRITON_CACHE_DIR
SLURM_JOB_ID SLURM_JOB_NODELIST SLURM_JOB_NUM_NODES SLURM_JOB_GPUS
SLURM_STEP_GPUS LOADEDMODULES""".split()


def run(argv, timeout=15):
    """Bound command duration, including descendants of local launchers."""
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
    except OSError as exc:
        return {"status": "unavailable", "error": str(exc), "command": argv}
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        status = "ok" if proc.returncode == 0 else "error"
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        status = "timeout"
    return {"status": status, "returncode": proc.returncode,
            "stdout": stdout, "stderr": stderr, "command": argv}


def read(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def sysfs_inventory():
    interfaces = {}
    for path in sorted(glob.glob('/sys/class/net/*')):
        p = Path(path)
        interfaces[p.name] = {key: read(p / key) for key in
                              ('operstate', 'mtu', 'speed', 'address', 'device/numa_node')}
    rdma = {}
    for path in sorted(glob.glob('/sys/class/infiniband/*')):
        p = Path(path)
        ports = {}
        for port in sorted(p.glob('ports/*')):
            ports[port.name] = {key: read(port / key) for key in
                                ('state', 'phys_state', 'rate', 'link_layer')}
            ports[port.name]['gids'] = {
                gid.name: {"gid": read(gid),
                           "type": read(port / 'gid_attrs/types' / gid.name),
                           "netdev": read(port / 'gid_attrs/ndevs' / gid.name)}
                for gid in sorted((port / 'gids').glob('*'))}
        rdma[p.name] = {"firmware": read(p / 'fw_ver'),
                        "numa_node": read(p / 'device/numa_node'), "ports": ports}
    return {"interfaces": interfaces, "rdma": rdma}


def probe(config):
    commands = {
        'gpus': ['nvidia-smi', '--query-gpu=' + ','.join(GPU_FIELDS),
                 '--format=csv,noheader,nounits'],
        'gpu_status': ['nvidia-smi'],
        'gpu_topology': ['nvidia-smi', 'topo', '-m'],
        'nvlink': ['nvidia-smi', 'nvlink', '--status'],
        'mig': ['nvidia-smi', '-L'],
        'cpu': ['lscpu'],
        'numa': ['numactl', '--hardware'],
        'addresses': ['ip', '-j', 'address', 'show'],
        'routes': ['ip', '-j', 'route', 'show', 'table', 'all'],
        'rdma_links': ['rdma', 'link', 'show'],
        'ib_devices': ['ibv_devinfo'],
        'ib_netdev': ['ibdev2netdev'],
        'cuda_compiler': ['nvcc', '--version'],
        'apptainer': ['apptainer', '--version'],
        'docker': ['docker', '--version'],
        'libraries': ['ldconfig', '-p'],
        'listeners': ['ss', '-ltn'],
    }
    if config.get('torch_check'):
        commands['torch_runtime'] = [sys.executable, '-c',
            'import json, torch; print(json.dumps({"torch":torch.__version__, '
            '"cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available(),'
            '"gpu_count":torch.cuda.device_count(),'
            '"nccl":torch.cuda.nccl.version() if torch.cuda.is_available() else None}))']
    results = {name: run(cmd, config['command_timeout']) for name, cmd in commands.items()}
    gpus = []
    if results['gpus']['status'] == 'ok':
        for row in csv.reader(io.StringIO(results['gpus']['stdout'])):
            if len(row) == len(GPU_FIELDS):
                gpus.append(dict(zip(GPU_FIELDS, [v.strip() for v in row])))
    packages = {}
    for name in ('torch', 'vllm', 'ray', 'transformers', 'triton',
                 'nvidia-nccl-cu12', 'nvidia-nccl-cu13'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    storage = {}
    for raw in ['/dev/shm', '/tmp'] + config.get('paths', []):
        path = os.path.expandvars(os.path.expanduser(raw))
        try:
            usage = shutil.disk_usage(path)
            storage[raw] = {"resolved_path": path, "total_bytes": usage.total,
                            "free_bytes": usage.free, "readable": os.access(path, os.R_OK),
                            "writable": os.access(path, os.W_OK)}
        except OSError as exc:
            storage[raw] = {"resolved_path": path, "error": str(exc)}
    return {"hostname": socket.gethostname(), "platform": platform.platform(),
            "architecture": platform.machine(), "python": sys.version,
            "python_executable": sys.executable, "os_release": read('/etc/os-release'),
            "memory": read('/proc/meminfo'), "mounts": read('/proc/mounts'),
            "kernel_modules": read('/proc/modules'),
            "limits": {"memlock": resource.getrlimit(resource.RLIMIT_MEMLOCK),
                       "open_files": resource.getrlimit(resource.RLIMIT_NOFILE)},
            "environment": {k: os.environ[k] for k in ENV_KEYS if k in os.environ},
            "gpus": gpus, "network": sysfs_inventory(), "packages": packages,
            "storage": storage, "commands": results}


def summarize(inv):
    warnings = []
    if not inv['gpus']:
        warnings.append('No GPUs inventoried; inspect nvidia-smi results.')
    if not inv['network']['rdma']:
        warnings.append('No RDMA devices visible in sysfs.')
    if not inv['packages']['vllm']:
        warnings.append('vLLM absent from probed Python (may exist in a container).')
    return warnings


def render(report):
    inv = report['inventory']
    lines = ['DGX per-node vLLM inventory', 'Collected: ' + report['collected_at'],
             'Host: %s | %s | GPUs: %d' %
             (inv['hostname'], inv['architecture'], len(inv['gpus']))]
    for gpu in inv['gpus']:
        lines.append('  GPU %s: %s, total/free MiB %s/%s, driver %s' %
                     (gpu['index'], gpu['name'], gpu['memory.total'],
                      gpu['memory.free'], gpu['driver_version']))
    lines.append('RDMA devices: ' + (', '.join(inv['network']['rdma']) or 'none visible'))
    lines.append('Packages: ' + ', '.join('%s=%s' % (k, v or 'absent')
                                          for k, v in inv['packages'].items()))
    missing = [k for k, v in inv['commands'].items() if v['status'] != 'ok']
    lines.append('Unavailable/failed probes: ' + (', '.join(missing) or 'none'))
    lines += ['', 'Review findings:'] + (['- ' + w for w in report['warnings']]
                                         or ['- No inventory findings.'])
    lines += ['', 'Inventory only: peer connectivity, NCCL performance, shared-file',
              'visibility, cross-node consistency, and model fit are not validated.']
    return '\n'.join(lines) + '\n'


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', dest='paths', action='append', default=[],
                        help='Local model/cache/container path to inspect; repeatable')
    parser.add_argument('--torch-check', action='store_true', help='Also import torch and query CUDA/NCCL')
    parser.add_argument('--command-timeout', type=positive, default=15)
    parser.add_argument('--output', help='JSON path; default: dgx-inventory-<hostname>-<UTC timestamp>.json')
    args = parser.parse_args()
    inventory = probe({'paths': args.paths, 'torch_check': args.torch_check,
                       'command_timeout': args.command_timeout})
    now = datetime.datetime.now(datetime.timezone.utc)
    report = {'schema_version': 2, 'collected_at': now.isoformat(),
              'inventory': inventory, 'warnings': summarize(inventory)}
    hostname = re.sub(r'[^A-Za-z0-9_.-]', '_', inventory['hostname'])
    output = Path(args.output or 'dgx-inventory-%s-%s.json' %
                  (hostname, now.strftime('%Y%m%dT%H%M%S%fZ')))
    summary = render(report)
    try:
        output.write_text(json.dumps(report, indent=2) + '\n')
        output.with_suffix(output.suffix + '.txt').write_text(summary)
    except OSError as exc:
        print('Could not write report: ' + str(exc), file=sys.stderr)
        return 1
    print(summary, end='')
    print('JSON: ' + str(output))
    return 0


if __name__ == '__main__':
    sys.exit(main())
