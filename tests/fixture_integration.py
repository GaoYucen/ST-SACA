"""Opt-in CPU-only, zero-training integration; never a scientific baseline.

Run once per reviewed seed: python -B tests/fixture_integration.py --seed 0
Only checkpoint input and station input are test doubles. All numerical modules,
policy forwards, dispatch, AM routing, fleet transitions and rewards are real.
"""
import argparse
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from unittest.mock import patch


def require(condition, code):
    if not condition:
        raise AssertionError(code)


def closed_km(start, coordinates, route):
    points = [start] + [coordinates[k] for k in route] + [start]
    total = 0.0
    for first, second in zip(points, points[1:]):
        lon1, lat1, lon2, lat2 = map(math.radians, (*first, *second))
        a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
        total += 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return total


def same(actual, expected, code):
    require(math.isfinite(float(actual)) and math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6), code)


def digest(module):
    result = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def run_method(module_name, seed, np, torch):
    module = importlib.import_module(module_name)
    require(str(module.device) == 'cpu', 'non_cpu_module')
    config = module.Config()
    config.demand_fluctuation = 0.0
    config.time_slots_per_episode = 4
    require((config.num_destinations, config.num_buses, config.bus_capacity) == (30, 10, 30), 'config_drift')
    # Baseline reset/step hard-code 20; its Config has no demand_amplitude.
    # Freeze this oracle independently without adding a production field.
    base_demand = 20
    if module_name.endswith('.st_saca'):
        require(config.demand_amplitude == base_demand, 'base_demand_drift')
    # Synthetic geometry is deliberately not the Chengdu station artifact.
    coordinates = np.array([[104.06 + .01 + .002*(k % 6), 30.67 + .01 + .002*(k // 6)] for k in range(30)])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    fixture_model = module.am.AttentionRouteModel(64, 8, 3).cpu().eval()
    fixture_state = copy.deepcopy(fixture_model.state_dict())
    stats = {'mean': torch.tensor([104.06, 30.67, 0.0]), 'std': torch.tensor([.02, .02, 30.0])}
    expected_paths = {str(module.ROUTING_CKPT_DIR / name): name for name in ('best_model.pth', 'normalization_stats.pt')}
    lookups = []

    def fixture_require(path, description):
        require(str(path) in expected_paths, 'unexpected_asset_lookup')
        return Path('/__st_saca_in_memory_fixture__') / expected_paths[str(path)]

    def fixture_load(path, *args, **kwargs):
        name = str(path)
        require(name in ('/__st_saca_in_memory_fixture__/best_model.pth', '/__st_saca_in_memory_fixture__/normalization_stats.pt'), 'unexpected_deserialization')
        lookups.append(name)
        return copy.deepcopy(fixture_state if name.endswith('best_model.pth') else stats)

    with patch.object(module, 'require_file', side_effect=fixture_require), patch.object(torch, 'load', side_effect=fixture_load), patch.object(module.gd, 'readbusstations', return_value=coordinates.tolist()):
        env = module.BusBookingEnv(config)
    require(len(lookups) == 2 and len(set(lookups)) == 2, 'asset_lookup_count')
    # The real constructor resets NumPy to 42. Explicitly reseed after it.
    np.random.seed(seed)
    torch.manual_seed(seed)
    actor_cls = module.SpatialActor if module_name.endswith('.st_saca') else module.Actor
    args = [config, 40, 60] + ([env.dest_coords] if actor_cls is getattr(module, 'SpatialActor', None) else [])
    actor = actor_cls(*args).cpu().eval()
    # Controlled final heads prevent random saturation from hiding dispatch.
    # Earlier policy layers and AM remain randomly initialized real modules.
    with torch.no_grad():
        for head in (actor.fc_p, actor.fc_a):
            head.weight.zero_()
            head.bias.zero_()
    env.dispatcher.eval()
    for model in (actor, env.dispatcher):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    before_hash = (digest(actor), digest(env.dispatcher))
    ids = [101 + 7*k for k in range(10)]
    records = []
    route_calls = []

    def check_route_forward(model, args, output):
        loc, start, weight = args
        route, log_probability = output
        require(loc.device.type == start.device.type == weight.device.type == 'cpu', 'route_device')
        require(bool(torch.isfinite(loc).all() and torch.isfinite(start).all() and torch.isfinite(weight).all()), 'route_input_finite')
        require(route.shape == (1, loc.shape[1]), 'route_shape')
        require(sorted(route[0].tolist()) == list(range(1, loc.shape[1]+1)), 'route_permutation')
        require(bool(torch.isfinite(log_probability).all()), 'route_log_probability')
        route_calls.append(loc.shape[1])

    handle = env.dispatcher.route_model.register_forward_hook(check_route_forward)
    original_dispatch = env.dispatcher.dispatch
    captured = []

    def recording_dispatch(orders, fleet):
        result = original_dispatch(orders, fleet)
        captured.append(copy.deepcopy(result))
        return result

    try:
        with patch.object(env.dispatcher, 'dispatch', side_effect=recording_dispatch):
            for index, case in enumerate(('mixed_idle_service', 'all_busy', 'zero_demand', 'terminal_service')):
                env.buses = {bid: [0.0, 30] for bid in ids}
                env.buses[ids[0]][0] = 2.5
                if case == 'all_busy':
                    env.buses = {bid: [2.5, 30] for bid in ids}
                env.N_p = np.full(30, 0 if case == 'zero_demand' else 20, dtype=int)
                old_fleet = copy.deepcopy(env.buses)
                demand = env.N_p.copy()
                captured.clear()
                state = env.get_state().unsqueeze(0)
                with torch.inference_mode():
                    action, log_probability, mean = actor.sample(state)
                require(action.shape == (1, 60) and mean.shape == (1, 60), 'actor_shape')
                require(bool(torch.isfinite(action).all() and torch.isfinite(log_probability).all()), 'actor_finite')
                require(bool(((action > 0) & (action < 1)).all()), 'actor_range')
                prices, allocation = np.split(action[0].numpy(), 2)
                allocation = np.clip(allocation, 0, 1)
                allocation /= allocation.sum() + 1e-6
                multiplier = (env.dist_max + env.dist_min - env.dist_k) / env.dist_max
                effective = demand * np.clip(1 - multiplier * prices**2, 0, 1)
                seats = sum(cap for remaining, cap in old_fleet.values() if remaining == 0)
                proposed = np.minimum(effective, seats * allocation * np.clip(multiplier * prices**2, 0, 1)).astype(int)
                reference_rng = np.random.RandomState()
                reference_rng.set_state(np.random.get_state())
                expected_next_demand = reference_rng.poisson(base_demand, 30)
                next_state, reward, done, info = env.step((prices.copy(), action[0, 30:].numpy().copy()))
                require(np.array_equal(env.N_p, expected_next_demand), 'demand_rng_advance')
                require(all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), reference_rng.get_state())), 'rng_state_advance')
                require(len(captured) == (1 if proposed.sum() else 0), 'dispatch_call_count')
                routes, assignments = captured[0] if captured else ({}, {})
                require(set(routes) == set(assignments), 'route_assignment_identity')
                served = np.zeros(30, dtype=int)
                distance = 0.0
                for bid, passengers in assignments.items():
                    require(bid in old_fleet and old_fleet[bid][0] == 0, 'busy_bus_dispatched')
                    require(0 < len(passengers) <= old_fleet[bid][1], 'capacity')
                    require(set(routes[bid]) == set(passengers) and len(routes[bid]) == len(set(routes[bid])), 'passenger_coverage')
                    for destination in passengers:
                        served[destination] += 1
                    distance += closed_km(config.departure_station, coordinates, routes[bid])
                require(bool((served <= proposed).all()), 'served_over_proposed')
                require(info['orders_accepted'] == int(served.sum()), 'served_total')
                require(info['orders_proposed'] == int(proposed.sum()), 'proposed_total')
                cost = config.beta_d * distance
                profit = float(np.dot(prices, served)) - cost
                orr = float(served.sum()) / (float(effective.sum()) + 1e-6)
                for field, expected in (('total_distance', distance), ('cost', cost), ('revenue', profit), ('orr', orr)):
                    same(info[field], expected, 'accounting_' + field)
                same(reward, profit + config.lambda_or * orr, 'reward')
                require(list(env.buses) == ids, 'bus_identity')
                for bid, (remaining, capacity) in old_fleet.items():
                    trip = closed_km(config.departure_station, coordinates, routes[bid]) / config.bus_speed if bid in routes else remaining
                    same(env.buses[bid][0], max(0.0, trip-config.time_slot_duration), 'fleet_clock')
                    require(env.buses[bid][1] == capacity, 'fleet_capacity')
                require(env.time_slots == list(range(1, index+2)), 'slot_clock')
                require(done is (index == 3), 'done_boundary')
                require(next_state.shape == (40,) and bool(torch.isfinite(next_state).all()), 'next_state')
                if case in ('all_busy', 'zero_demand'):
                    require(info['orders_accepted'] == info['orders_proposed'] == 0 and not routes, 'zero_service')
                else:
                    require(info['orders_accepted'] > 0 and routes, 'positive_service_coverage')
                records.append({'case': case, 'accepted': info['orders_accepted'], 'route_count': len(routes), 'done': done})
            frozen_state = copy.deepcopy(env.buses)
            frozen_rng = np.random.get_state()
            try:
                env.step((prices, allocation))
            except ValueError:
                pass
            else:
                raise AssertionError('post_terminal_step_accepted')
            require(env.buses == frozen_state and env.time_slots == [1, 2, 3, 4], 'post_terminal_mutation')
            require(all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), frozen_rng)), 'post_terminal_rng_mutation')
    finally:
        handle.remove()
    require(len(route_calls) > 0, 'am_never_called')
    require(before_hash == (digest(actor), digest(env.dispatcher)), 'weights_changed')
    require(all(parameter.grad is None for model in (actor, env.dispatcher) for parameter in model.parameters()), 'gradient_created')
    return {'method': module_name.rsplit('.', 1)[-1], 'seed': seed, 'steps': records, 'am_forward_calls': len(route_calls), 'weights_unchanged': True}


def execute(seed):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_available(), 'cuda_visible')
    # This opt-in CLI runs in its own process. Every temporary patch is restored.
    # Outside the two tightly-scoped constructor stubs every load/save fails.
    def forbidden(*args, **kwargs):
        raise AssertionError('checkpoint_or_training_operation_forbidden')
    with patch.object(torch, 'load', side_effect=forbidden), patch.object(torch, 'save', side_effect=forbidden), patch.object(torch.Tensor, 'backward', side_effect=forbidden), patch.object(torch.optim.Adam, 'step', side_effect=forbidden):
        results = [run_method(name, seed, np, torch) for name in ('st_saca.agents.st_saca', 'st_saca.agents.saca_baseline')]
    return {'protocol': 'synthetic-fixture-integration-v1', 'seed': seed, 'status': 'success', 'scientific_result_verified': False, 'training_updates': 0, 'production_checkpoint_deserializations': 0, 'torch': torch.__version__, 'numpy': np.__version__, 'results': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=True, type=int, choices=(0, 1))
    args = parser.parse_args()
    require('torch' not in sys.modules, 'torch_imported_before_cpu_setup')
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', WANDB_MODE='disabled', MPLBACKEND='Agg')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    # Only disposable cache files may be generated by imported plotting modules.
    with tempfile.TemporaryDirectory(prefix='st-saca-fixture-cache-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        try:
            result = execute(args.seed)
        except Exception as error:
            print(json.dumps({'protocol': 'synthetic-fixture-integration-v1', 'seed': args.seed, 'status': 'failure', 'error_type': type(error).__name__, 'scientific_result_verified': False}, sort_keys=True))
            raise
    encoded = json.dumps(result, sort_keys=True, allow_nan=False)
    require(len(encoded.encode()) <= 32768, 'summary_budget')
    print(encoded)


if __name__ == '__main__':
    main()
