"""Check the real batch/launcher handoff without Slurm, GPUs or model weights.

Run: BASH_BIN=/path/to/bash python3 tests/check_slurm_handoff.py
External srun/scontrol, GPU discovery, port binding and the final vLLM process
are simulated. Embedded model/context/speculation configuration runs unchanged.
"""
from pathlib import Path
import atexit
import os, sys, json, subprocess, tempfile, textwrap, shutil

repo = Path(__file__).resolve().parents[1]
bash = os.environ.get('BASH_BIN') or shutil.which('bash')
if not bash or subprocess.run([bash, '-c', '(( BASH_VERSINFO[0] >= 4 ))']).returncode:
    raise SystemExit('Bash 4+ is required; set BASH_BIN to its absolute path.')
bash = str(Path(bash).absolute())
fixture = Path(tempfile.mkdtemp(prefix='tacc-slurm-contract-')).resolve()
atexit.register(shutil.rmtree, fixture)
project = fixture/'project with spaces'
mockbin = fixture/'bin'
for p in [mockbin, project/'.venv/bin', project/'models/Qwen--Qwen3-Coder-Next-FP8', project/'models/z-lab--Qwen3-Coder-Next-DFlash', project/'frontend', project/'middleware', project/'support']:
    p.mkdir(parents=True, exist_ok=True)

def write(path, content, executable=False):
    path.write_text(textwrap.dedent(content))
    if executable: path.chmod(0o755)

write(project/'models/models.txt', '# model roles\n\nQwen/Qwen3-Coder-Next-FP8\nz-lab/Qwen3-Coder-Next-DFlash\n')
write(project/'models/one.txt', '# primary only\r\nQwen/Qwen3-Coder-Next-FP8\r\n')
config=json.dumps(dict(model_type='qwen3_next',max_position_embeddings=262144))
write(project/'models/Qwen--Qwen3-Coder-Next-FP8/config.json',config)
write(project/'models/z-lab--Qwen3-Coder-Next-DFlash/config.json','{}')
write(project/'frontend/chat.html','<html>fixture</html>')
write(project/'support/torch.py','class cuda:\n    @staticmethod\n    def device_count(): return 1\n')
write(project/'middleware/static_ui.py','''
    import os
    from pathlib import Path
    class StaticUIMiddleware:
        def __init__(self,app):
            assert Path(os.environ['TACC_UI_DIR']).joinpath(os.environ['TACC_UI_PAGE']).is_file()
            assert os.environ['TACC_UI_MODEL']
''')
write(project/'.venv/bin/activate','export VIRTUAL_ENV="$PROJECT/.venv"\n')
# Execute all embedded configuration Python for real, stubbing only GPU and bind checks.
write(project/'.venv/bin/python', '#!'+sys.executable+'\n'+textwrap.dedent('''
    import sys, subprocess
    if sys.argv[1] == '-':
        code=sys.stdin.read()
        if 'sock.bind((host, port))' in code: sys.exit(0)
        p=subprocess.run([sys.executable,*sys.argv[1:]],input=code,text=True)
    else:
        p=subprocess.run([sys.executable,*sys.argv[1:]])
    sys.exit(p.returncode)
'''), True)
write(project/'.venv/bin/vllm', '#!'+sys.executable+'\n'+textwrap.dedent('''
    import os,sys,json
    from pathlib import Path
    keys=['TP_SIZE','PP_SIZE','SPEC_METHOD','SPEC_MODEL','SPEC_TOKENS','MODEL_NAME','MODEL_PATH','HEAD_IP','MASTER_PORT','HOSTFILE','NUM_NODES','TACC_UI_DIR','TACC_UI_MODEL','TACC_PUBLIC_BASE_URL','VLLM_UI_ENABLE','VLLM_UI_DIR','VLLM_MIDDLEWARE_DIR','SLURM_EXPORT_ENV','VLLM_USE_V2_MODEL_RUNNER','TACC_QWEN3NEXT_PP_DFLASH']
    Path(os.environ['FAKE_CAPTURE_DIR'],os.environ['SLURMD_NODENAME']+'.json').write_text(json.dumps({'args':sys.argv[1:],'env':{k:os.environ.get(k) for k in keys},'cwd':os.getcwd()}))
'''),True)
write(project/'network.sh','''
    export INFER_LOCAL_IP=192.168.1.10
    export INFER_HTTP_HOST=192.168.1.10 INFER_HTTP_PORT=${SERVICE_PORT:-8040}
    export INFER_HTTP_URL=http://192.168.1.10:$INFER_HTTP_PORT
''')
write(mockbin/'scontrol','#!'+sys.executable+'\nimport os\nprint("\\n".join("dgx-"+str(i) for i in range(int(os.environ["SLURM_JOB_NUM_NODES"])) ))\n',True)
write(mockbin/'srun', '#!'+sys.executable+'\n'+textwrap.dedent('''
    import sys,os,subprocess,json
    from pathlib import Path
    args=sys.argv[1:]; split=args.index('bash'); opts=args[:split]; cmd=args[split:]
    assert '--export=ALL' in opts and '--mpi=none' in opts
    def option(name): return next((x.split('=',1)[1] for x in opts if x.startswith(name+'=')),None)
    out=option('--output'); cwd=option('--chdir') or os.getcwd()
    with open(Path(os.environ['FAKE_CAPTURE_DIR'],'srun.jsonl'),'a') as f: f.write(json.dumps({'opts':opts,'cwd':cwd})+'\\n')
    nodes=[option('--nodelist')] if option('--nodelist') else Path(os.environ['HOSTFILE']).read_text().splitlines()
    for node in nodes:
        env=os.environ.copy();env['SLURMD_NODENAME']=node
        if out:
            with open(out.replace('%N',node),'w') as f:
                p=subprocess.run(cmd,env=env,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
        else: p=subprocess.run(cmd,env=env,cwd=cwd)
        if p.returncode: sys.exit(p.returncode)
'''),True)
base_env = {k:v for k,v in os.environ.items() if not k.startswith(('MODEL','SPEC','TP_','PP_','VLLM_','TACC_','SLURM_','DEPLOY','PROJECT','NETWORK','HEAD_','CONTEXT_','MAX_','LOAD_','HTTP_','SERVICE_','NET_'))}
base_env.update(PATH=os.pathsep.join([str(mockbin), str(Path(bash).parent), os.defpath]),PROJECT=str(project),NETWORK_SCRIPT=str(project/'network.sh'),VLLM_UI_DIR=str(project/'frontend'),VLLM_MIDDLEWARE_DIR=str(project/'middleware'),PYTHONPATH=str(project/'support'),SLURM_JOB_NODELIST='dgx-[0-3]',SLURM_JOB_NUM_NODES='4',VLLM_API_KEY='test-key-never-log',LOCAL_NODE_NAME='stale',NODE_RANK='99')
count=0

def run(name, overrides=None, dry=False, submit=None, fail=None):
    global count
    count+=1
    work=fixture/name;work.mkdir()
    env=base_env.copy();env.update(SLURM_JOB_ID=str(count),SLURM_SUBMIT_DIR=str(submit or repo),FAKE_CAPTURE_DIR=str(work),LOG_DIR=str(work/'logs'))
    for k,v in (overrides or {}).items():
        if v is None: env.pop(k,None)
        else: env[k]=str(v)
    p=subprocess.run([bash,str(repo/'dgxspark/slurm-vllm.sbatch'),*(['--dry-run'] if dry else [])],cwd=work,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    logs='\n'.join(f.read_text() for f in (work/'logs').glob('vllm-run-*/node-*.log'))
    if fail:
        assert p.returncode!=0, (name,'unexpected success')
        assert fail in p.stdout+logs,(name,p.stdout,logs)
        assert not list(work.glob('dgx-*.json')),(name,'launched unexpectedly')
    else:
        assert p.returncode==0,(name,p.stdout,logs)
        assert 'test-key-never-log' not in p.stdout+logs,(name,'key logged')
        assert len(list((work/'logs').glob('vllm-run-*/node-*.log')))==int(env['SLURM_JOB_NUM_NODES'])
        if dry: assert not list(work.glob('dgx-*.json'))
    records=[json.loads(f.read_text()) for f in sorted(work.glob('dgx-*.json'))]
    print('PASS',name)
    return records,work,p.stdout+logs

def arg(record,key):
    args=record['args'];return args[args.index(key)+1]

records,work,logs=run('default')
assert len(records)==4
for i,r in enumerate(records):
    assert arg(r,'--tensor-parallel-size')=='2' and arg(r,'--pipeline-parallel-size')=='2'
    assert arg(r,'--node-rank')==str(i) and arg(r,'--nnodes')=='4'
    assert arg(r,'--distributed-executor-backend')=='mp'
    assert arg(r,'--max-num-seqs')=='8' and arg(r,'--max-num-batched-tokens')=='8192'
    assert '--speculative-config' not in r['args'] and r['env']['SPEC_MODEL'] is None
    assert r['env']['VLLM_UI_ENABLE'] is None and r['env']['VLLM_UI_DIR'] is None
    assert r['env']['SLURM_EXPORT_ENV']=='ALL'
    assert ('--headless' in r['args']) == (i!=0)
    assert ('--middleware' in r['args']) == (i==0)
    if i==0:
        assert arg(r,'--served-model-name')=='Qwen3-Coder-Next-FP8'
        assert r['env']['TACC_UI_DIR']==str(project/'frontend')
    assert r['cwd']==str(project)
steps=[json.loads(x) for x in (work/'srun.jsonl').read_text().splitlines()]
assert len(steps)==2 and steps[-1]['cwd']==str(work/'logs')
run('dry-from-dgxspark',dry=True,submit=repo/'dgxspark')
records,_,_=run('relative-paths',{'LAUNCHER':str(repo/'dgxspark/launch-vllm.sh'),'NETWORK_SCRIPT':'network.sh','VLLM_UI_DIR':'frontend','VLLM_MIDDLEWARE_DIR':'middleware','MODEL_LIST':'models/one.txt','MODEL_REPO':'models'},submit=project)
assert records[0]['env']['TACC_UI_DIR']==str(project/'frontend')
records,_,_=run('pipeline-override',{'TP_SIZE':'1','PP_SIZE':'4','CONTEXT_PROFILE':'128k','MAX_NUM_SEQS':'1','LOAD_FORMAT':'auto','HEAD_IP':'192.168.1.20','SERVED_MODEL_NAME':'custom-api'})
assert arg(records[0],'--pipeline-parallel-size')=='4' and arg(records[0],'--served-model-name')=='custom-api'
assert arg(records[0],'--master-addr')=='192.168.1.20'
assert arg(records[0],'--max-model-len')=='131072'
assert '--hf-overrides' not in records[0]['args']
records,_,_=run('ngram',{'TP_SIZE':'4','PP_SIZE':'1','SPEC_METHOD':'ngram','SPEC_TOKENS':'6','MODEL_LIST':str(project/'models/one.txt'),'VLLM_USE_V2_MODEL_RUNNER':'0'})
spec=json.loads(arg(records[0],'--speculative-config'))
assert spec==dict(method='ngram',num_speculative_tokens=6,prompt_lookup_min=2,prompt_lookup_max=5)
records,_,_=run('dflash',{'TP_SIZE':'4','PP_SIZE':'1','SPEC_METHOD':'dflash'})
spec=json.loads(arg(records[0],'--speculative-config'))
assert spec['model']==str(project/'models/z-lab--Qwen3-Coder-Next-DFlash') and spec['draft_tensor_parallel_size']==1 and spec['num_speculative_tokens']==15
records,_,_=run('eagle-override',{'TP_SIZE':'4','PP_SIZE':'1','SPEC_METHOD':'eagle3','SPEC_MODEL':'org/explicit-draft','MODEL_LIST':str(project/'models/one.txt'),'SPEC_TOKENS':'5'})
assert json.loads(arg(records[0],'--speculative-config'))['model']=='org/explicit-draft'
records,_,_=run('two-node',{'SLURM_JOB_NUM_NODES':'2','TP_SIZE':'1','PP_SIZE':'2','VLLM_UI_ENABLE':'0'})
assert len(records)==2 and '--middleware' not in records[0]['args']
run('bad-topology',{'TP_SIZE':'3','PP_SIZE':'2'},fail='must equal NUM_NODES=4')
run('bad-ngram',{'SPEC_METHOD':'ngram'},fail='N-gram preset requires PP_SIZE=1')
run('bad-ngram-v2',{'SPEC_METHOD':'ngram','TP_SIZE':'4','PP_SIZE':'1','VLLM_USE_V2_MODEL_RUNNER':'1'},fail='does not support VLLM_USE_V2_MODEL_RUNNER=1')
run('stale-draft',{'SPEC_MODEL':'org/draft'},fail='requires SPEC_MODEL and SPEC_TOKENS to be unset')
run('missing-second',{'SPEC_METHOD':'dflash','TP_SIZE':'4','PP_SIZE':'1','MODEL_LIST':str(project/'models/one.txt')},fail='requires a second model')
run('unsupported-method',{'SPEC_METHOD':'dspark'},fail='SPEC_METHOD must be')

# Simulate installed vLLM metadata/config classes for the experimental handoff.
# The actual patched forward/relay helpers are tested in test_qwen3next_pp_patch.py.
support=project/'support'
for folder in ['vllm', 'vllm/model_executor', 'vllm/model_executor/models',
               'vllm/v1', 'vllm/v1/worker', 'vllm/v1/worker/gpu']:
    directory=support/folder
    directory.mkdir(exist_ok=True,parents=True)
    write(directory/'__init__.py','')
metadata=support/'vllm-0.30.0.dist-info'
metadata.mkdir()
write(metadata/'METADATA','Metadata-Version: 2.1\nName: vllm\nVersion: 0.30.0\n')
source=support/'vllm/model_executor/models/qwen3_next.py'
source.write_bytes((repo/'tests/fixtures/qwen3next_pp/qwen3_next.py.txt').read_bytes())
write(support/'vllm/__init__.py','''
    import ast, os, sys, types
    from pathlib import Path
    # Import only capability assignments from the real source; no CUDA imports.
    path=Path(__file__).parent/'model_executor/models/qwen3_next.py'
    tree=ast.parse(path.read_text())
    model=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen3NextModel')
    namespace={'os':os}
    for node in model.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('_tacc_pp_patch','supports_aux_hidden_states_over_pp') for t in node.targets):
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    module=types.ModuleType('vllm.model_executor.models.qwen3_next')
    module.Qwen3NextModel=type('Qwen3NextModel',(),{k:v for k,v in namespace.items() if k in ('_tacc_pp_patch','supports_aux_hidden_states_over_pp')})
    sys.modules[module.__name__]=module
''')
write(support/'vllm/config.py','''
    from types import SimpleNamespace
    class ParallelConfig:
        def __init__(self,**kwargs): self.__dict__.update(kwargs)
    class SpeculativeConfig:
        @staticmethod
        def create_draft_parallel_config(target,tp):
            return SimpleNamespace(pipeline_parallel_size=1,tensor_parallel_size=tp)
''')
write(support/'vllm/v1/worker/gpu/model_runner.py','def verify_supports_aux_hidden_states_over_pp(*args): pass\n')
experimental=dict(TACC_QWEN3NEXT_PP_DFLASH='1',TP_SIZE='1',PP_SIZE='4',
                  SPEC_METHOD='dflash',CONTEXT_PROFILE='128k')
run('prototype-needs-patch',experimental,fail='Patch is not installed')
patch_env=base_env.copy()
subprocess.run([sys.executable,str(repo/'utility/qwen3next_pp_patch.py'),'--apply'],env=patch_env,check=True,stdout=subprocess.PIPE)
records,_,_=run('prototype-four-nodes',experimental)
for r in records:
    assert r['env']['TACC_QWEN3NEXT_PP_DFLASH']=='1'
    assert r['env']['VLLM_USE_V2_MODEL_RUNNER']=='1'
    assert '--enforce-eager' in r['args']
    assert json.loads(arg(r,'--speculative-config'))['draft_tensor_parallel_size']==1
    assert arg(r,'--load-format')=='fastsafetensors'
records,_,_=run('prototype-two-nodes',{**experimental,'SLURM_JOB_NUM_NODES':'2','PP_SIZE':'2'})
assert len(records)==2
run('prototype-without-optin',{**experimental,'TACC_QWEN3NEXT_PP_DFLASH':'0'},fail='lacks auxiliary hidden-state relay')
run('prototype-wrong-method',{**experimental,'SPEC_METHOD':'eagle3'},fail='requires SPEC_METHOD=dflash')
for tp,pp in [(2,2),(4,1),(2,3),(1,1),(1,8)]:
    records,_,_=run(f'prototype-tp{tp}-pp{pp}',{
        **experimental,'TP_SIZE':str(tp),'PP_SIZE':str(pp),
        'SLURM_JOB_NUM_NODES':str(tp*pp)})
    assert len(records)==tp*pp
    for r in records:
        assert arg(r,'--tensor-parallel-size')==str(tp)
        assert arg(r,'--pipeline-parallel-size')==str(pp)
        assert json.loads(arg(r,'--speculative-config'))['draft_tensor_parallel_size']==tp
        assert r['env']['VLLM_USE_V2_MODEL_RUNNER']=='1'
run('prototype-mismatched-draft-tp',{**experimental,'TP_SIZE':'2','PP_SIZE':'2','SPEC_TP_SIZE':'1'},
    fail='requires SPEC_TP_SIZE=TP_SIZE')
run('prototype-instanttensor-pp',{**experimental,'LOAD_FORMAT':'instanttensor'},
    fail='cannot use instanttensor')
records,_,_=run('prototype-auto-loader',{**experimental,'LOAD_FORMAT':'auto'})
assert all(arg(r,'--load-format')=='auto' for r in records)
run('prototype-zero-tp',{**experimental,'TP_SIZE':'0'},fail='TP_SIZE must be positive')
run('prototype-zero-pp',{**experimental,'PP_SIZE':'0'},fail='PP_SIZE must be positive')
run('prototype-no-eager',{**experimental,'ENFORCE_EAGER':'0'},fail='requires ENFORCE_EAGER=1')
run('prototype-extended-context',{**experimental,'CONTEXT_PROFILE':'1m'},fail='requires native context')
run('prototype-v1',{**experimental,'VLLM_USE_V2_MODEL_RUNNER':'0'},fail='requires VLLM_USE_V2_MODEL_RUNNER=1')
run('prototype-pp1-v1',{**experimental,'TP_SIZE':'4','PP_SIZE':'1','VLLM_USE_V2_MODEL_RUNNER':'0'},
    fail='requires VLLM_USE_V2_MODEL_RUNNER=1')
write(project/'models/Qwen--Qwen3-Coder-Next-FP8/config.json',json.dumps(dict(model_type='qwen3',max_position_embeddings=262144)))
run('prototype-wrong-model',experimental,fail='only supports model_type=qwen3_next')
run('prototype-pp1-wrong-model',{**experimental,'TP_SIZE':'4','PP_SIZE':'1'},fail='only supports model_type=qwen3_next')
write(project/'models/Qwen--Qwen3-Coder-Next-FP8/config.json',config)
source.write_bytes(source.read_bytes()+b'\n# unexpected local edit\n')
run('prototype-modified-source',experimental,fail='differs from the pinned')
print(f'{count} scenarios passed. Slurm, CUDA, port binding and final vLLM execution were simulated. Experimental cases also stub vLLM config/imports; patch verification and launcher code ran unchanged.')
