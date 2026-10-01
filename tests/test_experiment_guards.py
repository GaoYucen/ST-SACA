"""Fail-fast guards and call-site coverage, independent of model dependencies."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from st_saca.experiment_guards import (
    ExperimentConfigurationError, config_snapshot,
    validate_ablation_config, validate_method_selection,
)

ROOT = Path(__file__).resolve().parents[1]


def source_tree(path):
    return ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)


def actual_config(path):
    tree = source_tree(path)
    node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "Config")
    namespace = {"np": SimpleNamespace(array=lambda values: list(values))}
    exec(compile(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]), path, "exec"), namespace)
    return namespace["Config"]()


def function(path, name, owner=None):
    tree = source_tree(path)
    nodes = tree.body
    if owner:
        nodes = next(x for x in nodes if isinstance(x, ast.ClassDef) and x.name == owner).body
    return next(x for x in nodes if isinstance(x, ast.FunctionDef) and x.name == name)


def first_call(node):
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    statement = body[0]
    return statement.value.func.id if isinstance(statement, ast.Expr) and isinstance(
        statement.value, ast.Call) and isinstance(statement.value.func, ast.Name) else None


class TestConfigurationGuards(unittest.TestCase):
    def setUp(self):
        self.reference = dict(lr=0.0003, lambda_or=4.0, num_buses=10,
                              departure_station=[104.06, 30.67], seed=42)

    def test_verified_method_labels_are_not_blocked(self):
        for method in ("st-saca", "saca"):
            validate_method_selection(method)

    def test_unverified_baselines_are_blocked(self):
        for method in ("grc-elg", "jdrl-pomo"):
            with self.subTest(method=method), self.assertRaises(ExperimentConfigurationError) as ctx:
                validate_method_selection(method)
            self.assertEqual(ctx.exception.code, "unverified-baseline")

    def test_all_reports_both_blockers_before_partial_suite(self):
        with self.assertRaises(ExperimentConfigurationError) as ctx:
            validate_method_selection("all")
        self.assertEqual(len(ctx.exception.issues), 2)

    def test_unknown_method_is_blocked(self):
        with self.assertRaises(ExperimentConfigurationError):
            validate_method_selection("unknown")

    def test_wo_route_requires_identical_controls(self):
        validate_ablation_config("wo-route", self.reference, self.reference)
        for key, value in (("lr", 0.003), ("lambda_or", 0.1), ("num_buses", 20),
                           ("departure_station", [0, 0]), ("seed", 1)):
            actual = dict(self.reference, **{key: value})
            with self.subTest(key=key), self.assertRaises(ExperimentConfigurationError):
                validate_ablation_config("wo-route", actual, self.reference)

    def test_wo_orr_only_allows_zero_lambda_change(self):
        actual = dict(self.reference, lambda_or=0)
        validate_ablation_config("wo-orr", actual, self.reference)
        with self.assertRaises(ExperimentConfigurationError):
            validate_ablation_config("wo-orr", dict(actual, lambda_or=0.1), self.reference)

    def test_reports_both_lr_and_lambda_drift(self):
        with self.assertRaises(ExperimentConfigurationError) as ctx:
            validate_ablation_config("wo-route", dict(self.reference, lr=0.003, lambda_or=0.1),
                                     self.reference)
        self.assertIn("lr:", str(ctx.exception))
        self.assertIn("lambda_or:", str(ctx.exception))

    def test_missing_or_added_fields_are_rejected(self):
        for actual in ({k: v for k, v in self.reference.items() if k != "seed"},
                       dict(self.reference, allow_unverified=True)):
            with self.assertRaises(ExperimentConfigurationError):
                validate_ablation_config("wo-route", actual, self.reference)

    def test_invalid_numeric_values_are_rejected(self):
        for value in (True, "0.003", float("nan"), float("inf"), 0, -1):
            with self.subTest(value=value), self.assertRaises(ExperimentConfigurationError):
                validate_ablation_config("wo-route", dict(self.reference, lr=value), self.reference)

    def test_zero_full_reference_lambda_is_rejected(self):
        with self.assertRaises(ExperimentConfigurationError):
            validate_ablation_config("wo-orr", dict(self.reference, lambda_or=0),
                                     dict(self.reference, lambda_or=0))

    def test_snapshot_handles_array_like_config_without_numpy(self):
        class ArrayLike:
            def tolist(self):
                return [104.06, 30.67]
        config = SimpleNamespace(**dict(self.reference, departure_station=ArrayLike()))
        snapshot = config_snapshot(config)
        self.assertEqual(snapshot["departure_station"], (104.06, 30.67))
        validate_ablation_config("wo-route", snapshot, self.reference)

    def test_actual_wo_orr_defaults_fail_without_being_rewritten(self):
        actual = config_snapshot(actual_config("src/st_saca/experiments/ablation_wo_orr.py"))
        reference = config_snapshot(actual_config("src/st_saca/agents/st_saca.py"))
        with self.assertRaises(ExperimentConfigurationError) as ctx:
            validate_ablation_config("wo-orr", actual, reference)
        self.assertIn("lambda_or:", str(ctx.exception))
        self.assertIn("lr:", str(ctx.exception))
        self.assertIn("num_buses: missing", str(ctx.exception))

    def test_actual_wo_route_configuration_fails_before_seed_or_environment(self):
        path = "src/st_saca/experiments/ablation_wo_route.py"
        node = copy.deepcopy(function(path, "train_ablation_wo_route"))
        guard_index = next(i for i, stmt in enumerate(node.body)
                           if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                           and isinstance(stmt.value.func, ast.Name)
                           and stmt.value.func.id == "_validate_config")
        node.body = node.body[:guard_index + 1]
        reference = actual_config("src/st_saca/agents/st_saca.py")
        namespace = {
            "SACA": SimpleNamespace(Config=lambda: copy.deepcopy(reference)),
            "_validate_config": lambda config: validate_ablation_config(
                "wo-route", config_snapshot(config), config_snapshot(reference)),
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
        with self.assertRaises(ExperimentConfigurationError):
            namespace["train_ablation_wo_route"]()


class TestGuardPlacement(unittest.TestCase):
    def test_baseline_guards_are_first_before_side_effects(self):
        targets = [
            ("src/st_saca/baselines/grc_elg.py", "train_grc_elg", None),
            ("src/st_saca/baselines/grc_elg.py", "__init__", "BusBookingEnv"),
            ("src/st_saca/baselines/grc_elg.py", "__init__", "ELG_TSP_Solver"),
            ("src/st_saca/baselines/jdrl_pomo.py", "train_jdrl", None),
            ("src/st_saca/baselines/jdrl_pomo.py", "__init__", "BusBookingEnv"),
            ("src/st_saca/baselines/jdrl_pomo.py", "__init__", "AttentionDispatcherRouter"),
            ("src/st_saca/experiments/train.py", "_load_method", None),
            ("src/st_saca/experiments/train.py", "main", None),
        ]
        for path, name, owner in targets:
            with self.subTest(path=path, function=name, owner=owner):
                self.assertEqual(first_call(function(path, name, owner)), "validate_method_selection")

    def test_ablation_guards_are_before_environment_or_seed(self):
        for path, name, owner in [
            ("src/st_saca/experiments/ablation_wo_orr.py", "train_saca", None),
            ("src/st_saca/experiments/ablation_wo_orr.py", "__init__", "BusBookingEnv"),
            ("src/st_saca/experiments/ablation_wo_route.py", "__init__", "AblationEnv"),
        ]:
            with self.subTest(path=path, function=name):
                self.assertEqual(first_call(function(path, name, owner)), "_validate_config")

    def test_guard_module_imports_only_standard_library(self):
        tree = source_tree("src/st_saca/experiment_guards.py")
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(x.name for x in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module)
        self.assertEqual(set(imported), {"__future__", "math", "collections.abc", "numbers"})


if __name__ == "__main__":
    unittest.main()
