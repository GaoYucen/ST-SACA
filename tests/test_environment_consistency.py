"""Synthetic control-flow tests: no torch, numpy, checkpoint or raw-data access.

Execute the actual environment methods extracted from each source AST. Only
numeric vector/RNG operations and dispatch/routes are deterministic test stubs;
constructors that load learned models are intentionally not executed.
"""
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from st_saca.environment_consistency import (
    available_seats, advance_fleet, dispatch_and_account,
)

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "st_saca": "src/st_saca/agents/st_saca.py",
    "saca": "src/st_saca/agents/saca_baseline.py",
    "grc_elg": "src/st_saca/baselines/grc_elg.py",
    "jdrl_pomo": "src/st_saca/baselines/jdrl_pomo.py",
}


class Vector:
    def __init__(self, values):
        self.values = list(values)

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def _op(self, other, operation):
        rhs = other.values if isinstance(other, Vector) else [other] * len(self)
        if len(rhs) != len(self):
            raise ValueError("Vector shape mismatch")
        return Vector(operation(a, b) for a, b in zip(self.values, rhs))

    def __mul__(self, other):
        return self._op(other, lambda a, b: a * b)

    __rmul__ = __mul__

    def __add__(self, other):
        return self._op(other, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, other):
        return self._op(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._op(other, lambda a, b: b - a)

    def __truediv__(self, other):
        return self._op(other, lambda a, b: a / b)

    def __pow__(self, other):
        return self._op(other, lambda a, b: a ** b)

    def astype(self, dtype):
        return Vector(dtype(value) for value in self)


class DeterministicRandom:
    def __init__(self):
        self.calls = []

    def poisson(self, rate, size):
        self.calls.append((rate, size))
        return Vector([rate] * size)


class NumericStub:
    """Just the one-dimensional operations used by the audited methods."""
    def __init__(self):
        self.random = DeterministicRandom()

    @staticmethod
    def array(values, dtype=None):
        return Vector((dtype(value) if dtype else value) for value in values)

    @staticmethod
    def zeros(size, dtype=None):
        return Vector([0] * size)

    @staticmethod
    def zeros_like(values):
        return Vector([0] * len(values))

    @staticmethod
    def sum(values):
        return sum(values)

    @staticmethod
    def clip(values, low, high):
        return Vector(max(low, min(high, value)) for value in values)

    @staticmethod
    def minimum(left, right):
        return Vector(min(a, b) for a, b in zip(left, right))

    sin = staticmethod(math.sin)


def load_environment(path, numeric):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    original = next(node for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == "BusBookingEnv")
    selected = {"reset", "demand_function", "supply_function", "calculate_reward",
                "update_bus_state", "step"}
    node = copy.deepcopy(original)
    node.body = [method for method in node.body
                 if isinstance(method, ast.FunctionDef) and method.name in selected]
    node.bases = []
    node.decorator_list = []
    namespace = dict(np=numeric, available_seats=available_seats,
                     advance_fleet=advance_fleet, dispatch_and_account=dispatch_and_account)
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace["BusBookingEnv"]


class FixedDispatcher:
    def __init__(self, routes=None, passengers=None):
        self.routes = routes if routes is not None else {0: [0]}
        self.passengers = passengers if passengers is not None else {0: [0]}
        self.calls = []

    def dispatch(self, orders, buses):
        self.calls.append((list(orders), copy.deepcopy(buses)))
        return copy.deepcopy(self.routes), copy.deepcopy(self.passengers)


class EnvironmentChecks:
    def setUp(self):
        self.numeric = NumericStub()
        cls = load_environment(METHODS[self.method], self.numeric)
        self.env = cls.__new__(cls)
        self.env.config = SimpleNamespace(
            num_destinations=2, num_buses=2, bus_capacity=10,
            bus_speed=1.0, time_slot_duration=1.0,
            time_slots_per_episode=2, demand_amplitude=20,
            demand_fluctuation=5.0, demand_frequency=1.0,
            beta_d=0.1, lambda_or=4.0,
        )
        self.env.num_buses = 2
        self.env.buses = {0: [0.0, 10], 1: [2.0, 10]}
        self.env.time_slots = []
        self.env.current_p = Vector([0.5, 0.5])
        self.env.N_p = Vector([20, 20])
        self.env.dist_k = Vector([1.0, 1.0])
        self.env.dist_min = self.env.dist_max = 1.0
        self.env.dispatcher = FixedDispatcher()
        self.env.calculate_route_distance = lambda route: 3.0 if route else 0.0
        self.env.get_state = lambda: (
            tuple(self.env.N_p), tuple((bid, tuple(v)) for bid, v in self.env.buses.items())
        )
        self.action = (Vector([0.5, 0.5]), Vector([0.5, 0.5]))

    def step(self):
        result = self.env.step(self.action)
        if self.method == "jdrl_pomo":
            state, reward, bus_rewards, done, info = result
            self.assertAlmostEqual(sum(bus_rewards), info["revenue"])
        else:
            state, reward, done, info = result
        self.assertAlmostEqual(reward, info["revenue"] + 4.0 * info["orr"])
        return state, reward, done, info

    def test_supply_excludes_busy_capacity(self):
        self.env.supply_function(self.action[0], self.action[1])
        self.assertEqual(self.env.N_s, 10)

    def test_partial_dispatch_counts_actual_service(self):
        _, _, _, info = self.step()
        self.assertEqual(info["orders_proposed"], 2)
        self.assertEqual(info["orders_accepted"], 1)
        self.assertAlmostEqual(info["revenue"], 0.5 - 0.3)
        self.assertAlmostEqual(info["orr"], 1.0 / (30.0 + 1e-6))
        self.assertAlmostEqual(info["cost"], 0.3)
        self.assertAlmostEqual(info["total_distance"], 3.0)

    def test_all_busy_has_no_phantom_revenue(self):
        self.env.buses = {0: [2.0, 10], 1: [3.0, 10]}
        _, reward, done, info = self.step()
        self.assertEqual((reward, info["revenue"], info["orders_accepted"]), (0.0, 0.0, 0))
        self.assertEqual(len(self.env.dispatcher.calls), 0)
        self.assertEqual(self.env.buses, {0: [1.0, 10], 1: [2.0, 10]})
        self.assertFalse(done)

    def test_zero_orders_advance_clock_demand_and_done(self):
        self.env.buses = {0: [4.0, 10], 1: [5.0, 10]}
        self.assertFalse(self.step()[2])
        self.assertTrue(self.step()[2])
        self.assertEqual(self.env.time_slots, [1, 2])
        self.assertEqual(len(self.numeric.random.calls), 2)
        self.assertAlmostEqual(self.numeric.random.calls[0][0], 20)
        self.assertAlmostEqual(self.numeric.random.calls[1][0], 20 + 5 * math.sin(1))
        self.assertEqual(self.env.buses, {0: [2.0, 10], 1: [3.0, 10]})

    def test_route_stays_on_original_bus_id(self):
        self.env.buses = {0: [0.5, 10], 1: [0.0, 10]}
        self.env.dispatcher = FixedDispatcher({1: [0]}, {1: [0]})
        self.step()
        self.assertEqual(self.env.buses, {0: [0.0, 10], 1: [2.0, 10]})

    def test_new_short_trip_finishes_by_next_slot(self):
        self.env.calculate_route_distance = lambda route: 0.5
        self.step()
        self.assertEqual(self.env.buses[0][0], 0.0)
        self.assertEqual(available_seats(self.env.buses), 10)

    def test_done_rejects_an_extra_step(self):
        self.env.time_slots = [1, 2]
        with self.assertRaisesRegex(ValueError, "Episode is complete"):
            self.env.step(self.action)
        self.assertEqual(self.numeric.random.calls, [])

    def test_empty_dispatch_has_zero_actual_reward(self):
        self.env.dispatcher = FixedDispatcher({}, {})
        _, reward, _, info = self.step()
        self.assertEqual(info["orders_proposed"], 2)
        self.assertEqual((reward, info["orders_accepted"], info["cost"]), (0.0, 0, 0.0))

    def assert_bad_dispatch(self, routes, passengers):
        before = copy.deepcopy(self.env.buses)
        self.env.dispatcher = FixedDispatcher(routes, passengers)
        with self.assertRaises(ValueError):
            self.env.step(self.action)
        self.assertEqual(self.env.buses, before)
        self.assertEqual(self.env.time_slots, [])
        self.assertEqual(self.numeric.random.calls, [])

    def test_busy_bus_dispatch_is_rejected(self):
        self.assert_bad_dispatch({1: [0]}, {1: [0]})

    def test_unvisited_served_destination_is_rejected(self):
        self.assert_bad_dispatch({0: [1]}, {0: [0]})

    def test_overserving_proposed_destination_is_rejected(self):
        self.assert_bad_dispatch({0: [0]}, {0: [0, 0]})

    def test_overcapacity_dispatch_is_rejected(self):
        self.assert_bad_dispatch({0: [0]}, {0: [0] * 11})

    def test_mismatched_bus_ids_are_rejected(self):
        self.assert_bad_dispatch({0: [0]}, {})

    def test_valid_destination_is_safe_for_coordinate_indexing(self):
        coordinates = [3.0, 4.0]
        self.env.calculate_route_distance = lambda route: sum(coordinates[k] for k in route)
        self.assertEqual(self.step()[3]["total_distance"], 3.0)

    def test_float_destination_is_rejected(self):
        self.assert_bad_dispatch({0: [0.0]}, {0: [0.0]})

    def test_bool_destination_is_rejected(self):
        self.assert_bad_dispatch({0: [False]}, {0: [False]})

    def test_bool_bus_id_is_rejected(self):
        self.assert_bad_dispatch({False: [0]}, {False: [0]})

    def test_reset_respects_configured_fleet_size(self):
        self.env.config.num_buses = 3
        self.env.reset()
        self.assertEqual(list(self.env.buses), [0, 1, 2])
        self.assertEqual(available_seats(self.env.buses), 30)
        self.assertEqual(self.env.time_slots, [])


for _name in METHODS:
    globals()["Test_" + _name] = type(
        "Test_" + _name, (EnvironmentChecks, unittest.TestCase), {"method": _name}
    )


class TestFleetAtomicity(unittest.TestCase):
    def test_invalid_later_route_does_not_partially_mutate_fleet(self):
        fleet = {0: [0.0, 10], 1: [2.0, 10]}
        before = copy.deepcopy(fleet)
        with self.assertRaises(ValueError):
            advance_fleet(fleet, {0: [0], 1: [1]}, lambda _: 3.0, 1.0, 1.0)
        self.assertEqual(fleet, before)

    def test_overflowed_travel_time_cannot_mutate_fleet(self):
        fleet = {0: [0.0, 10]}
        before = copy.deepcopy(fleet)
        with self.assertRaisesRegex(ValueError, "Travel time must be finite"):
            advance_fleet(fleet, {0: [0]}, lambda _: 1e308, 1e-308, 1.0)
        self.assertEqual(fleet, before)

    def test_float_bus_id_is_rejected(self):
        with self.assertRaises(ValueError):
            available_seats({0.0: [0.0, 10]})

    def test_one_shot_fractional_count_iterators_are_rejected(self):
        env = SimpleNamespace(
            config=SimpleNamespace(num_destinations=2, beta_d=0.1),
            buses={0: [0.0, 10]}, dispatcher=FixedDispatcher(),
            current_p=Vector([0.5, 0.5]), calculate_route_distance=lambda _: 3.0)
        for counts in ([1.9, 0], [-0.2, 0]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                dispatch_and_account(env, iter(counts))
        self.assertEqual(env.dispatcher.calls, [])

    def test_negative_timer_is_rejected(self):
        with self.assertRaises(ValueError):
            available_seats({0: [-1.0, 10]})


if __name__ == "__main__":
    unittest.main()
