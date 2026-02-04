from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, get_type_hints, get_origin, get_args



# =========================
# Dataclasses
# =========================

@dataclass
class LRStep:
    step: int
    value: float


@dataclass
class ReplayStep:
    step: int
    capacity: int
    batch_size: int


@dataclass
class BootstrapSchedule:
    enabled: bool = True
    start_step: Optional[int] = None  # None => from start


@dataclass
class RunConfig:
    experiment_name: str = "default_name"
    seed: int = 42
    device: Literal["auto", "cpu", "cuda"] = "auto"
    deterministic_torch: bool = True


@dataclass
class EnvConfig:
    max_episode_steps: int = 300


@dataclass
class MCTSConfig:
    num_simulations: int = 200
    cpuct: float = 1.0
    max_depth: int = 64
    temperature: float = 0.5
    pw_k: float = 2.0
    pw_alpha: float = 0.5
    gamma_mcts: float = 0.85


@dataclass
class ActionSamplingConfig:
    K_uniform_per_node: int = 8
    warmstart_steps: int = 20000
    novelty_eps: float = 1e-3
    novelty_metric: Literal["l2", "linf"] = "l2"
    num_candidates: int = 1
    diversity_lambda: float = 0.0
    diversity_sigma: float = 0.25
    policy_beta: float = 1.0
    max_resample_attempts: int = 16


@dataclass
class NetworkConfig:
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])


@dataclass
class OptimConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4


@dataclass
class LossConfig:
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    value_target: Literal["mcts", "mc"] = "mcts"


@dataclass
class ReplayConfig:
    capacity: int = 500
    K_max: int = 16


@dataclass
class TrainLoopConfig:
    total_env_steps: int = 200_000
    train_updates_per_step: int = 1
    min_replay_factor: int = 10
    grad_clip_norm: float = 1.0


@dataclass
class EvalConfig:
    eval_every_steps: int = 10_000
    eval_episodes: int = 10


@dataclass
class SchedulesConfig:
    lr: List[LRStep] = field(default_factory=list)
    replay: List[ReplayStep] = field(default_factory=list)
    bootstrap: BootstrapSchedule = field(default_factory=BootstrapSchedule)


@dataclass
class OutputConfig:
    root_dir: str = "runs"
    save_checkpoints: bool = True
    checkpoint_every_steps: int = 10_000
    save_config_json: bool = True


@dataclass
class ExperimentConfig:
    run: RunConfig = field(default_factory=RunConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    action_sampling: ActionSamplingConfig = field(default_factory=ActionSamplingConfig)
    net: NetworkConfig = field(default_factory=NetworkConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    train: TrainLoopConfig = field(default_factory=TrainLoopConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    schedules: SchedulesConfig = field(default_factory=SchedulesConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# =========================
# Loading / merging
# =========================

def load_config(config_path: str | Path) -> ExperimentConfig:
    """
    Loads a JSON config and returns an ExperimentConfig.

    Supports:
    - Full configs like configs/stepwise_base.json
    - Partial override configs, either:
        (a) by using "__base__": "configs/stepwise_base.json"
        (b) by passing a fully expanded config file
    """
    config_path = Path(config_path)
    raw = _read_json(config_path)

    # Optional base chain
    if "__base__" in raw:
        base_path = (config_path.parent / raw["__base__"]).resolve()
        base_raw = _read_json(base_path)
        raw = _deep_merge(base_raw, {k: v for k, v in raw.items() if k != "__base__"})

    cfg = _from_dict(ExperimentConfig, raw)
    _validate_config(cfg)
    return cfg


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override into base.
    - dict merges recursively
    - non-dict values replace
    """
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# =========================
# Dict -> dataclass
# =========================

from typing import get_type_hints, get_origin, get_args

def _from_dict(cls, data: Dict[str, Any]):
    """
    Convert a nested dict to nested dataclasses.
    Unknown keys raise a ValueError (helps catch typos).
    Works with `from __future__ import annotations` by resolving type hints.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    kwargs = {}
    field_map = {f.name: f for f in fields(cls)}
    type_hints = get_type_hints(cls)  # <-- critical

    # unknown key check
    for k in data.keys():
        if k not in field_map:
            raise ValueError(f"Unknown config key for {cls.__name__}: '{k}'")

    for name, f in field_map.items():
        if name not in data:
            continue

        val = data[name]
        ftype = type_hints.get(name, f.type)

        # Nested dataclass
        if _is_dataclass_type(ftype) and isinstance(val, dict):
            kwargs[name] = _from_dict(ftype, val)
            continue

        # List of dataclasses
        origin = get_origin(ftype)
        args = get_args(ftype)

        if origin is list and args and _is_dataclass_type(args[0]) and isinstance(val, list):
            inner = args[0]
            kwargs[name] = [_from_dict(inner, x) if isinstance(x, dict) else x for x in val]
            continue

        # Plain assignment
        kwargs[name] = val

    return cls(**kwargs)



def _is_dataclass_type(t) -> bool:
    try:
        return is_dataclass(t)
    except Exception:
        return False


# =========================
# Validation
# =========================

def _validate_config(cfg: ExperimentConfig) -> None:
    if cfg.train.total_env_steps <= 0:
        raise ValueError("train.total_env_steps must be > 0")

    if cfg.eval.eval_every_steps <= 0:
        raise ValueError("eval.eval_every_steps must be > 0")

    if cfg.output.checkpoint_every_steps <= 0:
        raise ValueError("output.checkpoint_every_steps must be > 0")

    # Bootstrap sanity
    b = cfg.schedules.bootstrap
    if b.start_step is not None and b.start_step < 0:
        raise ValueError("schedules.bootstrap.start_step must be >= 0 or null")

    # Schedule monotonicity checks (optional but helpful)
    _assert_non_decreasing_steps([x.step for x in cfg.schedules.lr], "schedules.lr")
    _assert_non_decreasing_steps([x.step for x in cfg.schedules.replay], "schedules.replay")


def _assert_non_decreasing_steps(steps: List[int], name: str) -> None:
    if not steps:
        return
    for i in range(1, len(steps)):
        if steps[i] < steps[i - 1]:
            raise ValueError(f"{name} steps must be non-decreasing (got {steps})")
        
        
# =========================
# Saving
# =========================

def config_to_dict(cfg) -> dict[str, Any]:
    # asdict handles nested dataclasses + lists of dataclasses
    return asdict(cfg)

def save_resolved_config(cfg, run_dir: str | Path, filename: str = "config.json") -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(config_to_dict(cfg), f, indent=2, sort_keys=True)
    return out_path
