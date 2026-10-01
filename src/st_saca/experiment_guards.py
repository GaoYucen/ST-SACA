"""Fail-closed configuration/provenance guards; no framework or file imports."""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real


class ExperimentConfigurationError(ValueError):
    def __init__(self, code, issues):
        self.code = code
        self.issues = tuple(issues)
        super().__init__(f"{code}: " + "; ".join(self.issues))


def _plain(value, path):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result):
            raise ExperimentConfigurationError("invalid-config-value", [f"{path} must be finite"])
        return result
    if isinstance(value, (list, tuple)):
        return tuple(_plain(item, f"{path}[{i}]") for i, item in enumerate(value))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ExperimentConfigurationError("invalid-config-value", [f"{path} must have string keys"])
        return {key: _plain(value[key], f"{path}.{key}") for key in sorted(value)}
    raise ExperimentConfigurationError(
        "invalid-config-value", [f"{path} has unsupported type {type(value).__name__}"]
    )


def config_snapshot(config):
    values = config if isinstance(config, Mapping) else vars(config)
    result = {}
    for name, value in values.items():
        converter = getattr(value, "tolist", None)
        result[name] = converter() if callable(converter) else value
    return _plain(result, "config")


def _same(left, right):
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def validate_ablation_config(variant, actual, reference):
    """Validate config isolation only; never rewrite scientific defaults."""
    if variant not in {"wo-route", "wo-orr"}:
        raise ExperimentConfigurationError("unknown-ablation", [f"unsupported ablation {variant!r}"])
    actual, reference = _plain(actual, "actual"), _plain(reference, "reference")
    issues = []
    for label, values in (("actual", actual), ("reference", reference)):
        for name in ("lr", "lambda_or"):
            value = values.get(name)
            if isinstance(value, bool) or not isinstance(value, Real):
                issues.append(f"{label}.{name} must be a finite number")
    if issues:
        raise ExperimentConfigurationError("invalid-ablation-config", issues)
    if actual["lr"] <= 0 or reference["lr"] <= 0:
        issues.append("actual.lr and reference.lr must be positive")
    if reference["lambda_or"] <= 0:
        issues.append("the full-model reference must retain positive lambda_or")
    if actual["lambda_or"] < 0:
        issues.append("actual.lambda_or must be nonnegative")
    expected = dict(reference)
    if variant == "wo-orr":
        expected["lambda_or"] = 0.0
    for name in sorted(set(actual) | set(expected)):
        if name not in actual:
            issues.append(f"{name}: missing; reference expects {expected[name]!r}")
        elif name not in expected:
            issues.append(f"{name}: unexpected field {actual[name]!r}")
        elif not _same(actual[name], expected[name]):
            issues.append(f"{name}: got {actual[name]!r}; expected {expected[name]!r}")
    if issues:
        issues.append("Keep all control settings unchanged except the declared ablation: "
                      "wo-orr requires lambda_or=0; wo-route changes only its router. "
                      "Review the configuration explicitly; defaults have not been rewritten.")
        raise ExperimentConfigurationError("ablation-config-drift", issues)


_BASELINE_BLOCKERS = {
    "grc-elg": (
        "ELG routing policies are randomly initialized without training/checkpoint loading, "
        "and evaluation constructs a new router. A reviewed trained-router binding and "
        "preservation of that exact router in evaluation are required."
    ),
    "jdrl-pomo": (
        "This implementation loads AM best_model.pth/normalization_stats.pt rather than "
        "POMO outputs, and lacks a consistent coordinate, passenger-weight, normalization "
        "and augmentation contract. Review the complete binding; renaming files is not a fix."
    ),
}


def validate_method_selection(method):
    """No runtime bypass: replace a blocker only with a reviewed implementation."""
    if method not in {"all", "st-saca", "saca", *_BASELINE_BLOCKERS}:
        raise ExperimentConfigurationError("unknown-method", [f"unsupported method {method!r}"])
    selected = tuple(_BASELINE_BLOCKERS) if method == "all" else (method,)
    issues = [f"{name}: {_BASELINE_BLOCKERS[name]}" for name in selected
              if name in _BASELINE_BLOCKERS]
    if issues:
        raise ExperimentConfigurationError("unverified-baseline", issues)


def block_legacy_wo_orr_environment():
    """Configuration parity cannot certify the separate, unrepaired environment."""
    raise ExperimentConfigurationError(
        "unrepaired-ablation-environment",
        ["wo-orr still owns a legacy environment without the shared fleet/service "
         "repairs. Review and integrate that environment before enabling this "
         "ablation; matching scalar settings alone is insufficient."]
    )
