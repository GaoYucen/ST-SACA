"""Bounded newly-generated engineering assets; never a scientific baseline.

Stages: labels -> router -> sac (seed 0/1). All output directories must be new.
Numerical imports are deferred until CPU isolation and CLI validation finish.
"""
import argparse
import copy
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'st-saca-engineering-training-pilot-v1'
STATION_SHA = '8b6135f9ab112953b4f8731463c3d5dc1a22bfea1085b48849ff3c96538bd965'
DEPOT = [104.06, 30.67]
CAPACITY = 30
DATA_SEED = 20261001
MAX_ARTIFACT = 32 * 1024**2


def require(condition, code):
    if not condition:
        raise AssertionError(code)


def safe(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'absolute_path_required')
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink_refused')
    return path


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def fingerprint(path, limit=MAX_ARTIFACT):
    path = safe(path)
    require(path.is_file() and path.stat().st_size <= limit, 'artifact_size_or_type')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    path = safe(path)
    require(path.is_file() and path.stat().st_size <= 1024**2, 'json_size')
    return json.loads(path.read_bytes())


def write_json(path, value):
    data = canonical(value)
    require(len(data) <= 1024**2, 'json_size')
    with safe(path).open('xb') as stream:
        stream.write(data)


def git_head():
    result = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=5, check=True)
    head = result.stdout.strip()
    require(len(head) == 40 and all(c in '0123456789abcdef' for c in head), 'source_head')
    return head


def distance(first, second):
    lon1, lat1, lon2, lat2 = map(math.radians, (*first, *second))
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def passenger_cost(coordinates, passengers, order):
    require(sorted(order) == list(range(len(coordinates))), 'route_permutation')
    travelled = total = 0.0
    previous = DEPOT
    for index in order:
        travelled += distance(previous, coordinates[index])
        total += travelled * passengers[index]
        previous = coordinates[index]
    return total  # passenger-km, no return-to-depot segment


def instance_id(station_ids, passengers):
    return hashlib.sha256(canonical(sorted(zip(station_ids, passengers)))).hexdigest()


def validate_dataset(dataset, stations=None):
    require(dataset.get('protocol') == PROTOCOL and dataset.get('station_sha256') == STATION_SHA, 'dataset_protocol')
    require(dataset.get('seed') == DATA_SEED and dataset.get('capacity') == CAPACITY and dataset.get('objective') == 'sum_passenger_weighted_cumulative_haversine_km_no_return', 'dataset_scope')
    rows = dataset.get('instances', [])
    require(len(rows) == 80, 'dataset_count')
    counts = {(split, n): 0 for split in ('train', 'validation') for n in (5, 6)}
    identities = set()
    for row in rows:
        n = len(row['station_ids'])
        require((row['split'], n) in counts, 'split_cardinality')
        require(len(set(row['station_ids'])) == n and all(type(k) is int and 0 <= k < 30 for k in row['station_ids']), 'station_indices')
        weights = row['passengers']
        require(len(weights) == n and all(type(w) is int and w >= 0 for w in weights) and sum(weights) == CAPACITY, 'passenger_capacity')
        identity = instance_id(row['station_ids'], weights)
        require(row['identity'] == identity and identity not in identities, 'split_leakage_or_duplicate')
        identities.add(identity)
        require(len(row['station_coords']) == n and all(len(c) == 2 and all(math.isfinite(v) for v in c) for c in row['station_coords']), 'coordinates')
        if stations is not None:
            require(row['station_coords'] == [stations[k] for k in row['station_ids']], 'station_coordinate_binding')
        cost = passenger_cost(row['station_coords'], weights, row['local_optimal_order'])
        require(math.isclose(cost, row['optimal_passenger_km'], rel_tol=1e-10, abs_tol=1e-8), 'label_cost')
        counts[row['split'], n] += 1
    require(counts == {('train',5):32,('train',6):32,('validation',5):8,('validation',6):8}, 'split_counts')
    return rows


def manifest(out, stage, artifacts, **extra):
    document = {'protocol':PROTOCOL, 'stage':stage, 'research_commit':git_head(),
                'asset_role':'engineering_pilot_only', 'scientific_result_verified':False,
                'artifacts':{name:fingerprint(out/name) for name in artifacts}, **extra}
    write_json(out/'pilot_manifest.json', document)
    return document


def load_stage(directory, expected):
    directory = safe(directory)
    document = read_json(directory/'pilot_manifest.json')
    require(document.get('protocol') == PROTOCOL and document.get('stage') == expected and document.get('research_commit') == git_head() and document.get('asset_role') == 'engineering_pilot_only' and document.get('scientific_result_verified') is False, 'stage_identity')
    expected_names = {'labels':{'pilot_labels.json'},'router':{'pilot_router_state.pt','pilot_router_stats.pt','pilot_router_training_state.pt'}}
    if expected == 'sac':
        seed=document.get('seed')
        require(type(seed) is int and seed in (0,1), 'stage_seed')
        names={f'pilot_sac_{method}_seed{seed}.pt' for method in ('st_saca','saca_baseline')}
    else:
        require(expected in expected_names, 'stage_identity')
        names=expected_names[expected]
    require(isinstance(document.get('artifacts'),dict) and set(document['artifacts'])==names, 'artifact_manifest')
    for name, expected_hash in document['artifacts'].items():
        require(Path(name).name == name and isinstance(expected_hash,str) and re.fullmatch('[0-9a-f]{64}',expected_hash), 'artifact_name_or_hash')
        require(fingerprint(directory/name) == expected_hash, 'artifact_fingerprint')
    return document


def read_station_artifact():
    from st_saca.routing import gener_data as generator
    path=ROOT/'data/stations/chengdu_30_bus_stations.txt'
    require(fingerprint(path)==STATION_SHA,'station_fingerprint')
    stations=generator.readbusstations(path)
    require(len(stations)==30 and all(len(p)==2 and math.isfinite(p[0]) and math.isfinite(p[1]) and -180<=p[0]<=180 and -90<=p[1]<=90 for p in stations),'station_geometry')
    return stations


def make_labels(out):
    from st_saca.routing import gener_data as generator
    stations = read_station_artifact()
    random.seed(DATA_SEED)
    rows, seen = [], set()
    for n in (5, 6):
        for split, count in (('train',32),('validation',8)):
            for _ in range(count):
                for attempt in range(100):
                    ids = random.sample(range(30), n)
                    weights = generator.generate_passengers_distributed(n, CAPACITY)
                    identity = instance_id(ids, weights)
                    if identity not in seen:
                        break
                else:
                    raise AssertionError('duplicate_draw_budget')
                seen.add(identity)
                coordinates = [stations[k] for k in ids]
                route, solver_cost = generator.TspPassengerTimeSolver(coordinates, weights, DEPOT).solve()
                independent_cost = passenger_cost(coordinates, weights, route)
                optimum = min(passenger_cost(coordinates, weights, order) for order in itertools.permutations(range(n)))
                require(math.isclose(solver_cost, independent_cost, rel_tol=1e-10, abs_tol=1e-8), 'solver_cost_mismatch')
                require(math.isclose(optimum, independent_cost, rel_tol=1e-10, abs_tol=1e-8), 'nonoptimal_label')
                rows.append({'identity':identity,'split':split,'station_ids':ids,'station_coords':coordinates,'passengers':weights,'local_optimal_order':route,'optimal_passenger_km':independent_cost})
    dataset = {'protocol':PROTOCOL,'station_sha256':STATION_SHA,'seed':DATA_SEED,'capacity':CAPACITY,'objective':'sum_passenger_weighted_cumulative_haversine_km_no_return','instances':rows}
    validate_dataset(dataset,stations)
    write_json(out/'pilot_labels.json', dataset)
    return manifest(out,'labels',['pilot_labels.json'],instances=80,training_instances=64,validation_instances=16,zero_weight_nodes=sum(w==0 for row in rows for w in row['passengers']),exhaustive_crosscheck=True,station_sha256=STATION_SHA)


def numerical():
    import numpy as np
    import torch
    require(not torch.cuda.is_available(), 'gpu_visible')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    return np, torch


def tensor_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_state(torch, path, state):
    with safe(path).open('xb') as stream:
        torch.save(state, stream)
    fingerprint(path)


def optimizer_steps(optimizer, expected):
    steps = [int(state['step'].item()) for state in optimizer.state.values() if 'step' in state]
    parameters = sum(len(group['params']) for group in optimizer.param_groups)
    require(len(steps) == parameters and steps and all(step == expected for step in steps), 'optimizer_step_count')
    return {'parameter_states':len(steps),'step_min':min(steps),'step_max':max(steps)}


def finite_gradients(torch, model):
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    require(gradients and all(bool(torch.isfinite(g).all()) for g in gradients), 'gradient_finite')
    norm = sum(float(g.detach().square().sum()) for g in gradients)
    require(math.isfinite(norm) and norm > 0, 'zero_gradient')
    return norm


def finite_parameters(torch, model):
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'parameter_finite')


def batch_tensors(rows, stats, np, torch):
    loc = torch.tensor(np.array([r['station_coords'] for r in rows],dtype=np.float32))
    weights = torch.tensor(np.array([r['passengers'] for r in rows],dtype=np.float32))
    starts = torch.tensor([DEPOT]*len(rows),dtype=torch.float32)
    labels = torch.tensor([r['local_optimal_order'] for r in rows],dtype=torch.long)
    return ((loc-stats['mean'][None,None,:2])/stats['std'][None,None,:2],
            (starts-stats['mean'][None,:2])/stats['std'][None,:2],
            (weights-stats['mean'][2])/stats['std'][2], labels)


def route_validation(model, rows, stats, np, torch):
    model.eval()
    losses = {}
    with torch.no_grad():
        for n in (5,6):
            selected = [r for r in rows if r['split']=='validation' and len(r['passengers'])==n]
            loc,start,weight,label = batch_tensors(selected,stats,np,torch)
            loss = model.supervised_loss(label_route_local=label,station_weights=weight,station_coords=loc,start_coord=start)
            require(bool(torch.isfinite(loss)), 'validation_loss')
            routes, logs = model(loc,start,weight)
            require(all(sorted(route.tolist()) == list(range(1,n+1)) for route in routes) and bool(torch.isfinite(logs).all()), 'validation_routes')
            losses[str(n)] = float(loss)
    return losses


def train_router(out, labels_dir):
    data_manifest = load_stage(labels_dir,'labels')
    dataset = read_json(labels_dir/'pilot_labels.json')
    rows = validate_dataset(dataset,read_station_artifact())
    np,torch = numerical()
    from st_saca.routing import am
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    train_rows = [r for r in rows if r['split']=='train']
    groups = [[{'loc':np.array(r['station_coords'],dtype=np.float32),'weight':np.array(r['passengers'],dtype=np.float32),'start':np.array(DEPOT,dtype=np.float32)} for r in train_rows]]
    stats = am.compute_normalization_stats(groups)
    require(all(tuple(stats[k].shape)==(3,) and bool(torch.isfinite(stats[k]).all()) for k in ('mean','std')) and bool((stats['std']>0).all()), 'normalization_stats')
    model = am.AttentionRouteModel(64,8,3).cpu()
    initial = tensor_digest(model)
    before = route_validation(model,rows,stats,np,torch)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-4)
    selector = random.Random(5000)
    losses = []
    for update in range(20):
        n = (5,6)[update % 2]
        selected = selector.sample([r for r in train_rows if len(r['passengers'])==n],8)
        loc,start,weight,label = batch_tensors(selected,stats,np,torch)
        model.train();optimizer.zero_grad()
        loss = model.supervised_loss(label_route_local=label,station_weights=weight,station_coords=loc,start_coord=start)
        require(bool(torch.isfinite(loss)), 'training_loss')
        loss.backward();finite_gradients(torch,model);optimizer.step();finite_parameters(torch,model)
        losses.append(float(loss.detach()))
    witness = optimizer_steps(optimizer,20)
    final = tensor_digest(model)
    require(final != initial, 'router_weights_unchanged')
    after = route_validation(model,rows,stats,np,torch)
    save_state(torch,out/'pilot_router_state.pt',model.state_dict())
    save_state(torch,out/'pilot_router_stats.pt',stats)
    save_state(torch,out/'pilot_router_training_state.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'updates':20})
    clone = am.AttentionRouteModel(64,8,3).cpu().eval()
    clone.load_state_dict(torch.load(out/'pilot_router_state.pt',map_location='cpu',weights_only=True))
    require(tensor_digest(clone)==final, 'router_roundtrip')
    require(route_validation(clone,rows,stats,np,torch)==after, 'router_validation_roundtrip')
    require(load_stage(labels_dir,'labels') == data_manifest, 'input_mutated')
    return manifest(out,'router',['pilot_router_state.pt','pilot_router_stats.pt','pilot_router_training_state.pt'],training_updates=20,optimizer_witness=witness,initial_model_sha256=initial,final_model_sha256=final,validation_metric='mean_token_nll_by_cardinality',validation_before=before,validation_after=after,normalization_train_ids_sha256=hashlib.sha256(canonical(sorted(r['identity'] for r in train_rows))).hexdigest(),labels_manifest_sha256=fingerprint(labels_dir/'pilot_manifest.json'),label_artifact_sha256=data_manifest['artifacts']['pilot_labels.json'],last_step_checkpoint=True,quality_claim=False)


def equal_state(torch, first, second):
    if isinstance(first,torch.Tensor):
        return isinstance(second,torch.Tensor) and first.dtype==second.dtype and torch.equal(first.cpu(),second.cpu())
    if isinstance(first,dict):
        return isinstance(second,dict) and first.keys()==second.keys() and all(equal_state(torch,first[k],second[k]) for k in first)
    if isinstance(first,(list,tuple)):
        return isinstance(second,type(first)) and len(first)==len(second) and all(equal_state(torch,a,b) for a,b in zip(first,second))
    return first==second


def update_sac(out, router_dir, seed):
    assets = load_stage(router_dir,'router')
    require(assets.get('training_updates')==20 and assets.get('quality_claim') is False, 'router_role')
    np,torch = numerical()
    real_load = torch.load
    allowed = {str(router_dir/'pilot_router_state.pt'),str(router_dir/'pilot_router_stats.pt')}
    def bound_load(path,*args,**kwargs):
        require(str(path) in allowed, 'unbound_checkpoint_load')
        name = Path(path).name
        require(fingerprint(Path(path))==assets['artifacts'][name], 'checkpoint_fingerprint')
        return real_load(path,map_location='cpu',weights_only=True)
    results = []
    checkpoint_names = []
    for method in ('st_saca','saca_baseline'):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        module = importlib.import_module('st_saca.agents.'+method)
        require(str(module.device)=='cpu', 'device')
        config = module.Config();config.demand_fluctuation=0.0;config.time_slots_per_episode=8;config.batch_size=4;config.memory_size=8
        require((config.num_destinations,config.num_buses,config.bus_capacity)==(30,10,30), 'config')
        require(fingerprint(ROOT/'data/stations/chengdu_30_bus_stations.txt')==STATION_SHA, 'station_fingerprint')
        paths = {str(module.ROUTING_CKPT_DIR/'best_model.pth'):router_dir/'pilot_router_state.pt',str(module.ROUTING_CKPT_DIR/'normalization_stats.pt'):router_dir/'pilot_router_stats.pt'}
        def bound_require(path,description):
            require(str(path) in paths, 'unexpected_asset')
            return paths[str(path)]
        with patch.object(module,'require_file',side_effect=bound_require),patch.object(torch,'load',side_effect=bound_load):
            env=module.BusBookingEnv(config)
        for parameter in env.dispatcher.parameters(): parameter.requires_grad_(False)
        env.dispatcher.eval()
        args=[config,40,60,env.dispatcher]+([env.dest_coords] if method=='st_saca' else [])
        agent=module.SAC(*args)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        state=env.reset()
        route_hash=tensor_digest(env.dispatcher)
        calls=[]
        hook=env.dispatcher.route_model.register_forward_hook(lambda *args:calls.append(1))
        for step in range(8):
            p,a=agent.select_action(state,deterministic=False,greedy_samples=0)
            next_state,reward,done,info=env.step((p,a))
            require(math.isfinite(float(reward)) and bool(torch.isfinite(next_state).all()), 'transition_finite')
            require(done is (step==7), 'done')
            require(info['orders_accepted']<=info['orders_proposed'], 'served_orders')
            agent.store_transition(state,np.concatenate([p,a]),reward,next_state,done)
            state=next_state
        hook.remove()
        require(len(agent.memory)==8 and calls, 'valid_replay_and_routing')
        old_actor=tensor_digest(agent.actor);old_critic=tensor_digest(agent.critic)
        old_target=copy.deepcopy(agent.target_critic.state_dict());old_alpha=agent.log_alpha.detach().clone()
        losses=agent.update()
        require(losses is not None and all(math.isfinite(float(v)) for v in losses), 'sac_loss')
        finite_gradients(torch,agent.actor);finite_gradients(torch,agent.critic)
        for model in (agent.actor,agent.critic,agent.target_critic): finite_parameters(torch,model)
        require(bool(torch.isfinite(agent.alpha)) and float(agent.alpha.detach())>0, 'alpha_finite_positive')
        require(agent.log_alpha.grad is not None and bool(torch.isfinite(agent.log_alpha.grad)) and not torch.equal(old_alpha,agent.log_alpha.detach()), 'alpha_update')
        witnesses={name:optimizer_steps(getattr(agent,'optimizer_'+name),1) for name in ('actor','critic','alpha')}
        require(old_actor!=tensor_digest(agent.actor) and old_critic!=tensor_digest(agent.critic), 'actor_critic_update')
        for name,value in agent.target_critic.state_dict().items():
            expected=config.tau*agent.critic.state_dict()[name]+(1-config.tau)*old_target[name]
            require(torch.allclose(value,expected,rtol=1e-6,atol=1e-7), 'target_soft_update')
        require(route_hash==tensor_digest(env.dispatcher), 'router_changed')
        require(all(p.grad is None for p in env.dispatcher.parameters()), 'router_gradient')
        name=f'pilot_sac_{method}_seed{seed}.pt';checkpoint_names.append(name)
        agent.actor.eval();agent.critic.eval();agent.target_critic.eval()
        state_dict=copy.deepcopy(agent.state_dict());save_state(torch,out/name,state_dict)
        clone=module.SAC(*args)
        clone.load_state_dict(real_load(out/name,map_location='cpu',weights_only=True))
        clone.actor.eval();clone.critic.eval();clone.target_critic.eval()
        require(equal_state(torch,state_dict,clone.state_dict()), 'sac_optimizer_roundtrip')
        with torch.no_grad():
            first=agent.actor(state.unsqueeze(0));second=clone.actor(state.unsqueeze(0))
            require(all(torch.equal(a,b) for a,b in zip(first,second)), 'same_mode_forward_roundtrip')
        results.append({'method':method,'seed':seed,'transitions':8,'replay_batch_size':4,'sac_update_calls':1,'optimizer_witness':witnesses,'actor_critic_changed':True,'router_unchanged':True,'state_optimizer_roundtrip':True,'am_forward_calls':len(calls),'critic_loss':float(losses[0]),'actor_loss':float(losses[1]),'runtime_config':{k:getattr(config,k) for k in ('gamma','tau','lr','alpha','hidden_dim','lambda_or','bus_capacity','num_buses','time_slots_per_episode','demand_fluctuation')}})
    require(load_stage(router_dir,'router')==assets, 'router_asset_mutation')
    return manifest(out,'sac',checkpoint_names,seed=seed,methods=results,sac_update_calls_total=2,optimizer_steps_total=6,router_manifest_sha256=fingerprint(router_dir/'pilot_manifest.json'),replay_or_rng_resume_claim=False,quality_claim=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('labels','router','sac'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--seed',type=int,choices=(0,1))
    args=parser.parse_args()
    require((args.stage=='labels')==(args.input is None), 'stage_input')
    require((args.stage=='sac')==(args.seed is not None), 'stage_seed')
    require('torch' not in sys.modules,'late_cpu_isolation')
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',WANDB_MODE='disabled',MPLBACKEND='Agg')
    os.environ['ST_SACA_DATA_DIR']=str(ROOT/'data')
    os.environ['ST_SACA_ROUTING_CKPT_DIR']=str(ROOT/'checkpoints/routing')
    sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'src'))
    out=safe(args.out);out.mkdir(parents=False,exist_ok=False)
    if args.input is not None: args.input=safe(args.input)
    with tempfile.TemporaryDirectory(prefix='pilot-mpl-',dir=out) as cache:
        os.environ['MPLCONFIGDIR']=cache
        result=make_labels(out) if args.stage=='labels' else train_router(out,args.input) if args.stage=='router' else update_sac(out,args.input,args.seed)
    require(sum(p.stat().st_size for p in out.iterdir() if p.is_file())<=128*1024**2,'stage_artifact_budget')
    summary={k:v for k,v in result.items() if k!='artifacts'}
    summary.update(status='success',manifest_sha256=fingerprint(out/'pilot_manifest.json'),artifact_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()))
    raw=canonical(summary);require(len(raw)<=8192,'summary_budget');print(raw.decode(),end='')


if __name__=='__main__':
    main()
