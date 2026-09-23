#!/usr/bin/env bash
# Linux IPv4 environment setup for vLLM / SGLang using NCCL >= 2.21.
# No root, package installs, interface changes, firewall changes or MTU changes.
# Run in the same host/container/network namespace as the inference engine.
#
# Auto-select a local NIC, or optionally select using a route to the master:
#   source ./inference-network.sh --transport rdma
#   bash ./inference-network.sh --list
# Optional, separate HTTP NIC and available-port selection (Python 3 required):
#   source ./inference-network.sh --http --http-ports 8040-8050 --http-reserved-ports 8041
# In an automated launcher, stop on failure: source ... || exit 1
# Or wrap a command (arguments after -- are passed without eval):
#   bash ./inference-network.sh --master 192.168.1.1 -- vllm serve ...
#
# Discovery is local only. It does not prove peer reachability, RDMA bandwidth,
# GPU memory registration, driver/library compatibility, or model parallelism.
# Configure engine master address/port, node rank, TP/PP and API port separately.
# This does not configure Ray/MPI, UCX, NIXL, Mooncake, DeepEP or NVSHMEM.
# Reference: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

_inference_network_ports_for_iface() {
    local iface=$1 p f ndev device port
    for p in /sys/class/infiniband/*/ports/*; do
        [[ -d "$p" ]] || continue
        for f in "$p"/gid_attrs/ndevs/*; do
            [[ -r "$f" ]] || continue
            ndev=$(< "$f")
            if [[ "$ndev" == "$iface" ]]; then
                device=${p%/ports/*}; device=${device##*/}; port=${p##*/}
                printf '%s:%s\n' "$device" "$port"
                break
            fi
        done
    done
    if command -v ibdev2netdev >/dev/null; then
        ibdev2netdev 2>/dev/null | awk -v iface="$iface" '$5==iface {print $1 ":" $3}' || true
    fi
}

_inference_network_fabric() {
    local p=$1 layer='' f types=''
    [[ ! -r "$p/link_layer" ]] || layer=$(< "$p/link_layer")
    if [[ "$layer" == InfiniBand ]]; then printf 'InfiniBand'; return; fi
    for f in "$p"/gid_attrs/types/*; do
        [[ ! -r "$f" ]] || types+="$(< "$f") "
    done
    # Ethernet alone is not proof of RoCE: iWARP is also Ethernet RDMA.
    if [[ "$types" == *'RoCE v2'* ]]; then printf 'RoCEv2'
    elif [[ "$types" == *RoCE* ]]; then printf 'RoCE'
    elif [[ "$layer" == Ethernet ]]; then printf 'Ethernet-RDMA'
    else printf 'RDMA-unknown'; fi
}

_inference_network_inventory() {
    # TSV: NIC, IPv4(s), HCA:port, fabric, reported Mbps (0=unknown),
    #      netdev state, RDMA state, selection tier (0=ineligible).
    local n nic ips ports h p state active fabric speed netstate tier
    for n in /sys/class/net/*; do
        [[ -d "$n" ]] || continue
        nic=${n##*/}
        [[ "$nic" != lo ]] || continue
        ips=$(ip -o -4 addr show dev "$nic" scope global 2>/dev/null |
            awk '{split($4,a,"/"); print a[1]}' | sort -u | paste -sd, -)
        netstate=unknown
        [[ ! -r "$n/operstate" ]] || netstate=$(< "$n/operstate")
        ports=$(_inference_network_ports_for_iface "$nic" | sort -u)
        [[ -n "$ports" ]] || ports=-
        while IFS= read -r h; do
            speed=$(cat "$n/speed" 2>/dev/null) || speed=0
            [[ "$speed" =~ ^[0-9]+$ ]] || speed=0
            state=-; active=0; tier=0; fabric=no-RDMA-exposed
            if [[ "$h" != - ]]; then
                p=/sys/class/infiniband/${h%:*}/ports/${h##*:}
                [[ -r "$p/state" ]] || continue
                state=$(< "$p/state"); state=${state#*: }; state=${state// /_}
                [[ "$state" != ACTIVE ]] || active=1
                fabric=$(_inference_network_fabric "$p")
                if (( speed == 0 )) && [[ -r "$p/rate" ]]; then
                    speed=$(awk '{printf "%.0f",$1*1000}' "$p/rate")
                fi
            fi
            if [[ -n "$ips" && "$netstate" == up ]]; then
                if (( active )); then tier=3
                elif [[ "$h" == - && -e "$n/device" && ! -d "$n/wireless" ]]; then tier=2
                elif [[ "$h" == - && -d "$n/wireless" ]]; then tier=1; fi
            fi
            # Virtual netdevs without RDMA mappings require an explicit --iface.
            # A DOWN RDMA port is not silently reclassified as a TCP candidate.
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$nic" "${ips:--}" "$h" "$fabric" "$speed" "$netstate" "$state" "$tier"
        done <<< "$ports"
    done
}

_inference_network_http() {
    # Prints one TSV result; diagnostics/rule snapshots go to stderr.
    # Port checks briefly bind and close. No HTTP listener or firewall mutation.
    python3 - "$@" <<'PYHTTP'
import errno, ipaddress, json, shutil, socket, subprocess, sys
nic, client, bind_ip, port_spec, reserved_spec, audit, kctx, kns = sys.argv[1:]

def fail(message):
    print('HTTP selection: ' + message, file=sys.stderr)
    raise SystemExit(1)

def query(args):
    try:
        return json.loads(subprocess.check_output(['ip', '-j', '-4'] + args,
                          stderr=subprocess.PIPE, text=True, timeout=5))
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        fail('cannot query local routes/addresses: ' + str(exc))

def ports(text):
    result = []
    for part in text.split(','):
        ends = part.split('-')
        if len(ends) not in (1, 2) or any(not x.isdigit() for x in ends):
            fail('ports must be comma-separated numbers or ranges, e.g. 8040,8050-8060')
        lo, hi = int(ends[0]), int(ends[-1])
        if not 1 <= lo <= hi <= 65535:
            fail('invalid TCP port range')
        if len(result) + hi - lo + 1 > 1024:
            fail('limit each port list to 1024 entries')
        result.extend(range(lo, hi + 1))
    return list(dict.fromkeys(result))

candidates = ports(port_spec)
reserved = set(ports(reserved_spec)) if reserved_spec else set()
source = ''
reason = 'explicit-interface'
if client:
    try:
        ipaddress.IPv4Address(client)
    except ValueError:
        fail('--http-client requires a single IPv4 address')
    args = ['route', 'get', client]
    if nic:
        args += ['oif', nic]
    routes = query(args)
    if len(routes) != 1 or not routes[0].get('dev') or routes[0].get('type') in ('local', 'blackhole', 'unreachable', 'prohibit'):
        fail('cannot identify a route to the external client')
    route = routes[0]
    nic = nic or route['dev']
    source = route.get('prefsrc', route.get('src', ''))
    reason = 'route-to-client'
elif not nic:
    routes = query(['route', 'show', 'default'])
    routes = [r for r in routes if r.get('type', 'unicast') == 'unicast']
    if not routes:
        fail('no default route; specify --http-iface or --http-client')
    metric = min(int(r.get('metric', 0)) for r in routes)
    best = [r for r in routes if int(r.get('metric', 0)) == metric]
    if len(best) != 1 or best[0].get('nexthops') or not best[0].get('dev'):
        fail('ambiguous default route; specify --http-iface or --http-client')
    nic = best[0]['dev']
    source = best[0].get('prefsrc', best[0].get('src', ''))
    reason = 'default-route-heuristic'
if not nic or nic == 'lo':
    fail('need a non-loopback HTTP interface')
records = query(['addr', 'show', 'dev', nic])
if not records or 'UP' not in records[0].get('flags', []):
    fail('selected HTTP interface is not administratively UP')
addresses = sorted({a['local'] for rec in records for a in rec.get('addr_info', [])
                    if a.get('family') == 'inet' and a.get('scope') == 'global'})
if not bind_ip:
    if source in addresses:
        bind_ip = source
    elif len(addresses) == 1:
        bind_ip = addresses[0]
    else:
        fail('HTTP interface has zero or multiple IPv4 addresses; specify --http-ip')
if bind_ip not in addresses:
    fail('--http-ip must be a local IPv4 address on the HTTP interface')
chosen = None
for port in candidates:
    if port in reserved:
        print(f'HTTP port {port}: excluded as reserved', file=sys.stderr)
        continue
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((bind_ip, port))
        chosen = port
        break
    except OSError as exc:
        if exc.errno not in (errno.EADDRINUSE, errno.EACCES, errno.EPERM):
            fail(f'cannot bind {bind_ip}:{port}: {exc}')
        print(f'HTTP port {port}: skipped ({exc.strerror})', file=sys.stderr)
if chosen is None:
    fail('no bindable non-reserved port in the supplied pool')

# Firewall evaluation depends on complete packet context, chain order, NAT,
# source addresses and sometimes cluster/cloud policies. Collect evidence;
# never equate an allow rule, empty output or denied inspection with reachability.
def snapshot(args):
    print('\nRead-only policy snapshot: ' + ' '.join(args), file=sys.stderr)
    if not shutil.which(args[0]):
        print('Tool unavailable; policy remains unknown.', file=sys.stderr)
        return
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=5)
        print(result.stdout, end='', file=sys.stderr)
        print(result.stderr, end='', file=sys.stderr)
        if result.returncode:
            print(f'Inspection failed (exit {result.returncode}); policy remains unknown.', file=sys.stderr)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f'Inspection unavailable: {exc}', file=sys.stderr)

if audit == '1':
    for args in [['ip', '-4', 'rule', 'show'], ['nft', 'list', 'ruleset'],
                 ['iptables-save'], ['ufw', 'status', 'verbose'],
                 ['firewall-cmd', '--list-all-zones']]:
        snapshot(args)
    print('Host snapshots are not an effective-policy verdict. No sudo escalation or rules changed.', file=sys.stderr)
if kctx:
    snapshot(['kubectl', '--context', kctx, '--namespace', kns, '--request-timeout=4s',
              'get', 'networkpolicies', '-o', 'yaml'])
    snapshot(['kubectl', '--context', kctx, '--namespace', kns, '--request-timeout=4s',
              'get', 'services,ingresses', '-o', 'yaml'])
    print('Kubernetes snapshot is incomplete without pod/namespace labels, CNI policy and client path. Reachability remains unknown.', file=sys.stderr)
print(f'HTTP candidate: {nic} {bind_ip}:{chosen} ({reason})', file=sys.stderr)
print('Port was bindable locally but is not held open. Firewall permission and external reachability are unverified.', file=sys.stderr)
print(f'After launch, run FROM THE INTENDED CLIENT: curl --noproxy "*" --connect-timeout 3 --max-time 10 -i http://{bind_ip}:{chosen}/health', file=sys.stderr)
print('\t'.join([nic, bind_ip, str(chosen), reason]))
PYHTTP
}

_inference_network_main() {
    local master='' iface='' local_ip='' hca='' transport=auto debug=0
    local master_ip='' route='' route_iface='' route_src='' addresses=''
    local p f ndev state device port layer matches='' candidate=''
    local sourced=0 list=0 inventory='' ranked='' selected='' best='' count=''
    local http=0 http_iface='' http_client='' http_ip='' http_ports=8000-8099
    local http_reserved=29500,29501 http_audit=0 http_result='' http_port='' http_reason=''
    local http_kctx='' http_kns=''
    [[ "${BASH_SOURCE[0]}" != "$0" ]] && sourced=1
    while (( $# )); do
        case "$1" in
            --master|--iface|--local-ip|--hca|--transport|--http-iface|--http-client|--http-ip|--http-port|--http-ports|--http-reserved-ports|--http-k8s-context|--http-k8s-namespace)
                if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
                    printf 'Missing value for %s\n' "$1" >&2; return 2
                fi
                case "$1" in
                    --master) master=$2;;
                    --iface) iface=$2;;
                    --local-ip) local_ip=$2;;
                    --hca) hca=$2;;
                    --transport) transport=$2;;
                    --http-iface) http=1; http_iface=$2;;
                    --http-client) http=1; http_client=$2;;
                    --http-ip) http=1; http_ip=$2;;
                    --http-port)
                        [[ "$2" =~ ^[0-9]+$ ]] || { printf 'Use one numeric --http-port, or --http-ports for a pool.\n' >&2; return 2; }
                        http=1; http_ports=$2;;
                    --http-ports) http=1; http_ports=$2;;
                    --http-reserved-ports) http=1; http_reserved=$2;;
                    --http-k8s-context) http=1; http_kctx=$2;;
                    --http-k8s-namespace) http=1; http_kns=$2;;
                esac
                shift 2;;
            --debug) debug=1; shift;;
            --list) list=1; shift;;
            --http) http=1; shift;;
            --http-audit) http=1; http_audit=1; shift;;
            --) shift; break;;
            -h|--help)
                cat <<'HELP'
Usage: source ./inference-network.sh [options]
       bash ./inference-network.sh --list
       bash ./inference-network.sh [options] -- COMMAND [ARGS...]

  --master IP|HOST    OPTIONAL: use routing to the head to select a local NIC.
                     Multiple DNS IPv4 results require an explicit IP.
  --iface NAME        Override the automatically selected bootstrap/Gloo NIC.
  --local-ip IP       Select a local IPv4 address when the interface has several.
  --hca DEVICE:PORT   Select one active RDMA port explicitly, e.g. mlx5_0:1.
                     May differ from the bootstrap interface (separate fabrics).
  --transport MODE   auto (default), rdma, or socket; use the same mode on peers.
                     auto lets NCCL choose transport; it does NOT test RDMA.
                     rdma requires an active port and forces NCCL_NET=IB.
                     socket forces TCP and disables NCCL's IB transport.
  --debug            Enable NCCL INFO logging for INIT,BOOTSTRAP,NET,GRAPH.
  --list             Print local NIC and RDMA port inventory; change no exports.

Optional HTTP selection (independent of NCCL NIC; Python 3 + iproute2 JSON):
  --http                   Select HTTP NIC and locally bindable port.
  --http-client IPV4        Prefer route back to an intended external client.
  --http-iface NAME         Explicit HTTP NIC; otherwise use the default route.
  --http-ip IPV4            Bind address on that NIC if it has multiple IPs.
  --http-port PORT          Require this exact port (fail if unavailable).
  --http-ports POOL         Comma-separated ports/ranges; default 8000-8099.
                           Supply an administrator-approved pool when known.
  --http-reserved-ports POOL Exclusions; default 29500,29501. REPLACES defaults.
                           Include your engine rendezvous ports, e.g. 8041.
  --http-audit              Read-only host policy snapshots, no sudo escalation.
  --http-k8s-context NAME   Optional explicit Kubernetes context for snapshots.
  --http-k8s-namespace NAME Required with context; never uses an implicit cluster.

HTTP selection exports INFER_HTTP_IFACE, INFER_HTTP_HOST, INFER_HTTP_PORT,
INFER_HTTP_URL, INFER_HTTP_SELECTION, INFER_HTTP_PORT_STATUS,
INFER_HTTP_POLICY_STATUS and INFER_HTTP_REACHABILITY, and prints their values.
Without --http these variables are cleared to avoid retaining old selections.
Pass --host "$INFER_HTTP_HOST" --port "$INFER_HTTP_PORT" to the API server.
The port check only binds/closes a socket: the port is NOT reserved and could
be taken before engine startup. It is NOT a firewall or reachability test.
Default-route detection is only an outbound-route heuristic. NAT, cloud ACLs,
container port mappings, Kubernetes Services/Ingress/CNI policies require
deployment-specific configuration and verification from the intended client.
Rule snapshots are diagnostic evidence; arbitrary policies are not evaluated
or used to label ports allowed/blocked. Choose ports within the approved pool.

Without --master/--iface, rank local candidates: active RDMA > physical wired
Ethernet > Wi-Fi, then reported link speed within the tier. Ties fail and show
the candidates. Unknown speeds on multiple candidates in the best tier also
fail. This is a local heuristic, not a connectivity or throughput benchmark.
Candidates need an UP netdev and global-scope IPv4 (private IPs qualify).
RDMA ports without an IP netdev are listed but require separate bootstrap NIC
selection with --iface and --hca. Virtual NICs need --iface unless mapped to RDMA.
With --transport rdma, only active RDMA candidates qualify. Socket mode ranks
physical interfaces by reported speed without an RDMA preference.
INFER_MASTER_ADDR is empty when --master is omitted: this does NOT elect a head.

Exports INFER_IFACE, INFER_LOCAL_IP, INFER_HCA, INFER_MASTER_ADDR,
INFER_TRANSPORT and VLLM_HOST_IP, plus NCCL/Gloo network settings.
Does not export generic IFACE/HCA variables or change engine launch arguments.
RDMA discovery uses Linux sysfs, with ibdev2netdev as an optional fallback.

Owns/replaces NCCL_NET, NCCL_IB_DISABLE, NCCL_IB_HCA, NCCL_SOCKET_IFNAME,
NCCL_SOCKET_FAMILY, GLOO_SOCKET_IFNAME, VLLM_HOST_IP and dynamic GID settings
(NCCL_IB_GID_INDEX, NCCL_IB_ADDR_RANGE, NCCL_IB_ADDR_FAMILY,
NCCL_IB_ROCE_VERSION_NUM). GID index/range overrides are cleared intentionally.
Other tuning variables, shell options, working directory and ulimits stay intact.
Use a fresh shell if troubleshooting inherited NCCL tuning or system nccl.conf.

Linux/Bash/IPv4 only. Single-port selection; advanced multi-rail/routed fabrics
need explicit topology-specific configuration. Run inside your engine container
if applicable. RDMA devices and permissions must already be exposed there.
HELP
                return 0;;
            *) printf 'Unknown option: %s (see --help)\n' "$1" >&2; return 2;;
        esac
    done
    if (( sourced && $# )); then
        printf 'Use bash, rather than source, when wrapping a command.\n' >&2; return 2
    fi
    case "$transport" in auto|rdma|socket) ;; *) printf 'Invalid transport.\n' >&2; return 2;; esac
    if [[ "$transport" == socket && -n "$hca" ]]; then
        printf '%s\n' '--hca conflicts with --transport socket.' >&2; return 2
    fi
    [[ "$(uname -s)" == Linux ]] || { printf 'Linux is required.\n' >&2; return 1; }
    command -v ip >/dev/null || { printf 'iproute2 (ip command) is required.\n' >&2; return 1; }
    if [[ -n "$http_kctx" && -z "$http_kns" || -z "$http_kctx" && -n "$http_kns" ]]; then
        printf 'Provide both --http-k8s-context and --http-k8s-namespace.\n' >&2; return 2
    fi
    if (( list )); then
        printf 'NIC\tIPv4\tHCA:PORT\tFABRIC\tMbps(0=unknown)\tLINK\tRDMA_STATE\tTIER\n'
        _inference_network_inventory
        printf '\nAll RDMA ports (including unmapped/no-IP ports):\n'
        for p in /sys/class/infiniband/*/ports/*; do
            [[ -r "$p/state" ]] || continue
            device=${p%/ports/*}; device=${device##*/}; port=${p##*/}
            printf '%s:%s\t%s\t%s\n' "$device" "$port" "$(< "$p/state")" "$(_inference_network_fabric "$p")"
        done
        printf '\nNo RDMA mapping means no RDMA device exposed by the current driver/namespace; it is not proof that the hardware cannot support RDMA.\n'
        return 0
    fi
    if [[ -z "$master" && -z "$iface" ]]; then
        inventory=$(_inference_network_inventory)
        ranked=$(awk -F '\t' -v mode="$transport" -v h="$hca" -v addr="$local_ip" '
            $8>0 && (mode!="rdma" || $8==3) && (h=="" || $3==h) {
                if(addr!="") { n=split($2,ips,","); ok=0; for(i=1;i<=n;i++) if(ips[i]==addr) ok=1; if(!ok) next }
                tier=$8; if(mode=="socket" && tier==3) tier=2;
                print tier "\t" $5 "\t" $0
            }' <<< "$inventory" | sort -t $'\t' -k1,1nr -k2,2nr)
        [[ -n "$ranked" ]] || { printf 'No eligible local NIC. Run --list; use --iface/--hca for a separate or virtual fabric.\n' >&2; return 1; }
        selected=${ranked%%$'\n'*}
        best=$(cut -f1,2 <<< "$selected")
        count=$(awk -F '\t' -v best="$best" '$1 "\t" $2==best {n++} END {print n+0}' <<< "$ranked")
        # Unknown speed cannot safely be ranked below a known speed in the top tier.
        if (( count != 1 )) || awk -F '\t' -v tier="${selected%%$'\t'*}" '
            $1==tier {n++; if($2==0) unknown=1} END {exit !(n>1 && unknown)}' <<< "$ranked"; then
            printf 'NIC selection is ambiguous (tie or unknown speed). Use --iface or --hca after checking --list:\n%s\n' "$ranked" >&2; return 1
        fi
        iface=$(cut -f3 <<< "$selected")
        if [[ "$transport" != socket ]]; then
            hca=$(cut -f5 <<< "$selected"); [[ "$hca" != - ]] || hca=''
        fi
        printf 'Selected %s by local capability/link-speed heuristic; peer connectivity is unverified.\n' "$iface" >&2
    fi
    if [[ -n "$master" ]]; then
    if [[ "$master" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        master_ip=$master
    else
        command -v getent >/dev/null || { printf 'Use an IPv4 literal or install getent.\n' >&2; return 1; }
        master_ip=$(getent ahostsv4 "$master" | awk '{print $1}' | sort -u)
        if [[ -z "$master_ip" || "$master_ip" == *$'\n'* ]]; then
            printf 'Master DNS is missing or ambiguous; supply a single IPv4 address.\n' >&2; return 1
        fi
    fi
    [[ "$master_ip" != 127.* && "$master_ip" != 0.0.0.0 ]] || {
        printf 'Use a peer-reachable master address, not loopback/wildcard.\n' >&2; return 1;
    }
    # On the head, route-to-self normally says "dev lo". Find the interface
    # owning the master IP before consulting the routing table.
    if [[ -z "$iface" ]]; then
        iface=$(ip -o -4 addr show | awk -v target="$master_ip" '
            {split($4,a,"/"); if(a[1]==target) {sub(/@.*/,"",$2); print $2}}' | sort -u)
        [[ "$iface" != *$'\n'* ]] || { printf 'Master IP is assigned to multiple interfaces; use --iface.\n' >&2; return 1; }
    fi
    if [[ -n "$iface" ]]; then
        route=$(ip -4 route get "$master_ip" oif "$iface" 2>/dev/null) || {
            printf 'No route to %s using %s.\n' "$master_ip" "$iface" >&2; return 1;
        }
    else
        route=$(ip -4 route get "$master_ip" 2>/dev/null) || {
            printf 'No route to master %s.\n' "$master_ip" >&2; return 1;
        }
    fi
    route_iface=$(awk '{for(i=1;i<NF;i++) if($i=="dev") {print $(i+1); exit}}' <<< "$route")
    route_src=$(awk '{for(i=1;i<NF;i++) if($i=="src") {print $(i+1); exit}}' <<< "$route")
    iface=${iface:-$route_iface}
    fi
    [[ -n "$iface" && "$iface" != lo ]] || { printf 'No usable interface; supply --iface.\n' >&2; return 1; }
    ip link show dev "$iface" >/dev/null 2>&1 || { printf 'Interface %s does not exist.\n' "$iface" >&2; return 1; }
    addresses=$(ip -o -4 addr show dev "$iface" scope global | awk '{split($4,a,"/"); print a[1]}')
    if [[ -z "$local_ip" ]]; then
        if [[ -n "$route_src" ]] && awk -v x="$route_src" '$0==x {found=1} END {exit !found}' <<< "$addresses"; then
            local_ip=$route_src
        elif [[ -n "$addresses" && "$addresses" != *$'\n'* ]]; then
            local_ip=$addresses
        else
            printf 'Cannot choose local IPv4 address; supply --local-ip.\n' >&2; return 1
        fi
    fi
    awk -v x="$local_ip" '$0==x {found=1} END {exit !found}' <<< "$addresses" || {
        printf '%s is not a global IPv4 address on %s.\n' "$local_ip" "$iface" >&2; return 1;
    }
    if [[ "$transport" != socket ]]; then
        if [[ -z "$hca" ]]; then
            for p in /sys/class/infiniband/*/ports/*; do
                [[ -r "$p/state" ]] || continue
                state=$(< "$p/state")
                [[ "$state" == *ACTIVE* ]] || continue
                for f in "$p"/gid_attrs/ndevs/*; do
                    [[ -r "$f" ]] || continue
                    ndev=$(< "$f")
                    if [[ "$ndev" == "$iface" ]]; then
                        device=${p%/ports/*}; device=${device##*/}; port=${p##*/}
                        matches+="$device:$port"$'\n'
                        break
                    fi
                done
            done
            if [[ -z "$matches" ]] && command -v ibdev2netdev >/dev/null; then
                matches=$(ibdev2netdev 2>/dev/null | awk -v iface="$iface" '$5==iface && $6=="(Up)" {print $1 ":" $3}')
            fi
            candidate=$(printf '%s\n' "$matches" | awk 'NF' | sort -u)
            if [[ "$candidate" == *$'\n'* ]]; then
                printf 'Multiple RDMA ports match %s; select --hca DEVICE:PORT:\n%s\n' "$iface" "$candidate" >&2; return 1
            fi
            hca=$candidate
        fi
        if [[ -n "$hca" ]]; then
            [[ "$hca" =~ ^[[:alnum:]_.-]+:[1-9][0-9]*$ ]] || { printf 'Use --hca DEVICE:PORT (one port).\n' >&2; return 2; }
            device=${hca%:*}; port=${hca##*:}; p=/sys/class/infiniband/$device/ports/$port
            [[ -r "$p/state" ]] || { printf 'RDMA port %s is not visible in sysfs.\n' "$hca" >&2; return 1; }
            state=$(< "$p/state")
            [[ "$state" == *ACTIVE* ]] || { printf 'RDMA port %s is not ACTIVE.\n' "$hca" >&2; return 1; }
            layer=$(< "$p/link_layer")
            printf 'RDMA candidate: %s (%s); peer connectivity is not yet verified.\n' "$hca" "$layer" >&2
        elif [[ "$transport" == rdma ]]; then
            printf 'No active RDMA port matches %s. Check rdma link/ibdev2netdev, or specify --hca for a separate fabric.\n' "$iface" >&2; return 1
        else
            printf 'No RDMA port mapped to %s; NCCL auto-selection remains enabled. Use --hca for a separate fabric or --transport socket for TCP only.\n' "$iface" >&2
        fi
    fi
    if (( http )); then
        command -v python3 >/dev/null || { printf 'HTTP detection requires Python 3.\n' >&2; return 1; }
        http_result=$(_inference_network_http "$http_iface" "$http_client" "$http_ip" \
            "$http_ports" "$http_reserved" "$http_audit" "$http_kctx" "$http_kns") || return 1
        IFS=$'\t' read -r http_iface http_ip http_port http_reason <<< "$http_result"
    fi
    # Commit exports only after validating the complete local selection.
    export INFER_IFACE="$iface" INFER_LOCAL_IP="$local_ip" INFER_HCA="$hca"
    export INFER_MASTER_ADDR="$master_ip" INFER_TRANSPORT="$transport"
    export VLLM_HOST_IP="$local_ip" GLOO_SOCKET_IFNAME="$iface"
    export NCCL_SOCKET_IFNAME="=$iface" NCCL_SOCKET_FAMILY=AF_INET
    unset NCCL_IB_GID_INDEX NCCL_IB_ADDR_RANGE
    export NCCL_IB_ADDR_FAMILY=AF_INET NCCL_IB_ROCE_VERSION_NUM=2
    if [[ -n "$hca" ]]; then export NCCL_IB_HCA="=$hca"; else unset NCCL_IB_HCA; fi
    case "$transport" in
        auto) unset NCCL_NET; export NCCL_IB_DISABLE=0;;
        rdma) export NCCL_NET=IB NCCL_IB_DISABLE=0;;
        socket) export NCCL_NET=Socket NCCL_IB_DISABLE=1;;
    esac
    if (( debug )); then
        export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET,GRAPH
    fi
    if (( http )); then
        export INFER_HTTP_IFACE="$http_iface" INFER_HTTP_HOST="$http_ip" INFER_HTTP_PORT="$http_port"
        export INFER_HTTP_URL="http://$http_ip:$http_port" INFER_HTTP_SELECTION="$http_reason"
        export INFER_HTTP_PORT_STATUS=bindable-at-check INFER_HTTP_POLICY_STATUS=unverified
        export INFER_HTTP_REACHABILITY=unverified
    else
        unset INFER_HTTP_IFACE INFER_HTTP_HOST INFER_HTTP_PORT INFER_HTTP_URL INFER_HTTP_SELECTION
        unset INFER_HTTP_PORT_STATUS INFER_HTTP_POLICY_STATUS INFER_HTTP_REACHABILITY
    fi
    printf 'Network: iface=%s local=%s master=%s mode=%s hca=%s\n' \
        "$iface" "$local_ip" "$master_ip" "$transport" "${hca:-auto/none}" >&2
    printf 'memlock limit: %s KiB (or unlimited)\n' "$(ulimit -l)" >&2
    # Fixed allowlist: print only settings owned by this script, not the full environment.
    local env_name
    printf 'Managed environment variables (current values):\n' >&2
    for env_name in \
        INFER_IFACE INFER_LOCAL_IP INFER_HCA INFER_MASTER_ADDR INFER_TRANSPORT \
        VLLM_HOST_IP GLOO_SOCKET_IFNAME \
        NCCL_SOCKET_IFNAME NCCL_SOCKET_FAMILY NCCL_NET NCCL_IB_DISABLE NCCL_IB_HCA \
        NCCL_IB_GID_INDEX NCCL_IB_ADDR_RANGE NCCL_IB_ADDR_FAMILY NCCL_IB_ROCE_VERSION_NUM \
        NCCL_DEBUG NCCL_DEBUG_SUBSYS \
        INFER_HTTP_IFACE INFER_HTTP_HOST INFER_HTTP_PORT INFER_HTTP_URL INFER_HTTP_SELECTION \
        INFER_HTTP_PORT_STATUS INFER_HTTP_POLICY_STATUS INFER_HTTP_REACHABILITY; do
        if [[ -v "$env_name" ]]; then
            printf '  %s=%q\n' "$env_name" "${!env_name}" >&2
        else
            printf '  %s=<unset>\n' "$env_name" >&2
        fi
    done
    if (( $# )); then
        exec "$@"
    elif (( ! sourced )); then
        printf 'Inspection only: use source to retain exports, or append -- COMMAND.\n' >&2
    fi
    return 0
}

_inference_network_main "$@"

