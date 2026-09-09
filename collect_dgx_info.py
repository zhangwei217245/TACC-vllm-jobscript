#!/usr/bin/env python3
"""Read-only distributed deployment inventory; Python 3.8+, no pip dependencies."""

import argparse
import concurrent.futures
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
import shlex
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


def run(argv, timeout=15, input_text=None):
    """Bound command duration, including descendants of local launchers."""
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
    except OSError as exc:
        return {"status": "unavailable", "error": str(exc), "command": argv}
    try:
        stdout, stderr = proc.communicate(input_text, timeout=timeout)
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
    peers = {}
    for host in config.get('peers', []):
        # Separate process keeps DNS lookup bounded even with a broken resolver.
        peers[host] = run([sys.executable, '-c',
            'import socket,json,sys; print(json.dumps(sorted(set('
            'x[4][0] for x in socket.getaddrinfo(sys.argv[1],None)))))', host],
            config['command_timeout'])
    return {"hostname": socket.gethostname(), "platform": platform.platform(),
            "architecture": platform.machine(), "python": sys.version,
            "python_executable": sys.executable, "os_release": read('/etc/os-release'),
            "memory": read('/proc/meminfo'), "mounts": read('/proc/mounts'),
            "kernel_modules": read('/proc/modules'),
            "limits": {"memlock": resource.getrlimit(resource.RLIMIT_MEMLOCK),
                       "open_files": resource.getrlimit(resource.RLIMIT_NOFILE)},
            "environment": {k: os.environ[k] for k in ENV_KEYS if k in os.environ},
            "gpus": gpus, "network": sysfs_inventory(), "packages": packages,
            "storage": storage, "peer_dns": peers, "commands": results}


def collect_node(host, args, source, config):
    payload = source + '\n' + 'print(json.dumps(probe(' + repr(config) + ')))\n'
    if args.transport == 'ssh':
        argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
                shlex.join([args.python, '-'])]
    else:
        argv = ['srun', '--nodes=1', '--ntasks=1', '--nodelist=' + host,
                '--exclusive', args.python, '-']
    result = run(argv, args.node_timeout, payload)
    if result['status'] != 'ok':
        return {"target": host, "status": "failed", "transport": result}
    try:
        # Ignore startup banners, but never silently select an arbitrary JSON object.
        lines = [line for line in result['stdout'].splitlines() if line.startswith('{')]
        if len(lines) != 1:
            raise ValueError('Expected one JSON report from remote probe')
        report = json.loads(lines[0])
        if not isinstance(report.get('commands'), dict):
            raise ValueError('Invalid probe report')
        return {"target": host, "status": "collected", "inventory": report}
    except (ValueError, AttributeError) as exc:
        return {"target": host, "status": "failed", "error": str(exc), "transport": result}


def summarize(nodes, expected):
    warnings = []
    good = [n for n in nodes if n['status'] == 'collected']
    if len(good) != expected:
        warnings.append('Expected %d nodes; collected %d.' % (expected, len(good)))
    names = [n['inventory']['hostname'] for n in good]
    if len(set(names)) != len(names):
        warnings.append('Multiple targets returned the same hostname; check node aliases.')
    signatures = {}
    for node in good:
        inv, host = node['inventory'], node['target']
        if not inv['gpus']:
            warnings.append(host + ': no GPUs inventoried; inspect nvidia-smi results.')
        if not inv['network']['rdma']:
            warnings.append(host + ': no RDMA devices visible in sysfs.')
        if not inv['packages']['vllm']:
            warnings.append(host + ': vLLM absent from probed Python (may exist in a container).')
        signatures[host] = {
            'GPU models/memory/driver': sorted((g['name'], g['memory.total'], g['driver_version'])
                                               for g in inv['gpus']),
            'architecture': inv['architecture'], 'Python packages': inv['packages']}
    for key in ('GPU models/memory/driver', 'architecture', 'Python packages'):
        if len({json.dumps(s[key], sort_keys=True) for s in signatures.values()}) > 1:
            warnings.append(key + ' differ across nodes.')
    return warnings


def render(report):
    lines = ['DGX distributed vLLM inventory', 'Collected: ' + report['collected_at'], '']
    for node in report['nodes']:
        lines.append(node['target'] + ': ' + node['status'])
        if node['status'] != 'collected':
            lines.append('  See JSON for transport failure details.')
            continue
        inv = node['inventory']
        lines.append('  Host: %s | %s | GPUs: %d' %
                     (inv['hostname'], inv['architecture'], len(inv['gpus'])))
        for gpu in inv['gpus']:
            lines.append('  GPU %s: %s, total/free MiB %s/%s, driver %s' %
                         (gpu['index'], gpu['name'], gpu['memory.total'],
                          gpu['memory.free'], gpu['driver_version']))
        lines.append('  RDMA devices: ' + (', '.join(inv['network']['rdma']) or 'none visible'))
        lines.append('  Packages: ' + ', '.join('%s=%s' % (k, v or 'absent')
                                              for k, v in inv['packages'].items()))
        missing = [k for k, v in inv['commands'].items() if v['status'] != 'ok']
        lines.append('  Unavailable/failed probes: ' + (', '.join(missing) or 'none'))
    lines += ['', 'Review findings:'] + ['- ' + w for w in report['warnings']]
    lines += ['', 'Inventory only: DNS is checked; peer TCP/RDMA connectivity, NCCL',
              'performance, shared-file visibility, and model fit are not validated.']
    return '\n'.join(lines) + '\n'


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport', choices=['ssh', 'slurm', 'local'], default='slurm')
    parser.add_argument('--nodes', nargs='+', help='Explicit hostnames; Slurm defaults to allocation')
    parser.add_argument('--expected-nodes', type=positive, default=4)
    parser.add_argument('--python', default='python3', help='Python executable on each remote node')
    parser.add_argument('--path', dest='paths', action='append', default=[],
                        help='Model/cache/container path to inspect on every node; repeatable')
    parser.add_argument('--torch-check', action='store_true', help='Also import torch and query CUDA/NCCL')
    parser.add_argument('--command-timeout', type=positive, default=15)
    parser.add_argument('--node-timeout', type=positive, default=600)
    parser.add_argument('--output', default='dgx-inventory.json')
    args = parser.parse_args()
    if args.transport == 'local' and args.nodes:
        parser.error('--nodes is not applicable to local mode')
    hosts = args.nodes or []
    if args.transport == 'slurm':
        if not os.environ.get('SLURM_JOB_ID'):
            parser.error('Slurm mode requires an active allocation; use --transport ssh otherwise')
        if not hosts:
            expanded = run(['scontrol', 'show', 'hostnames', os.environ.get('SLURM_JOB_NODELIST', '')])
            if expanded['status'] != 'ok':
                parser.error('Could not expand SLURM_JOB_NODELIST: ' + str(expanded))
            hosts = expanded['stdout'].split()
    if args.transport != 'local' and not hosts:
        parser.error('--nodes is required for SSH')
    if len(set(hosts)) != len(hosts) or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', h) for h in hosts):
        parser.error('Use unique plain hostnames or SSH config aliases (no user@host)')
    config = {'paths': args.paths, 'torch_check': args.torch_check,
              'command_timeout': args.command_timeout, 'peers': hosts}
    if args.transport == 'local':
        nodes = [{"target": socket.gethostname(), "status": "collected", "inventory": probe(config)}]
    else:
        source = Path(__file__).read_text().split('\nif __name__ ==')[0]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(hosts), 4)) as pool:
            nodes = list(pool.map(lambda host: collect_node(host, args, source, config), hosts))
    report = {'schema_version': 1,
              'collected_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'transport': args.transport, 'expected_nodes': args.expected_nodes,
              'nodes': nodes, 'warnings': summarize(nodes, args.expected_nodes)}
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2) + '\n')
    summary = render(report)
    output.with_suffix(output.suffix + '.txt').write_text(summary)
    print(summary, end='')
    print('JSON: ' + str(output))
    return 0 if len(nodes) == args.expected_nodes and all(n['status'] == 'collected' for n in nodes) else 1


if __name__ == '__main__':
    sys.exit(main())
