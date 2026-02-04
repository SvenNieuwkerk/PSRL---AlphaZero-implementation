# train.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

import gymnasium as gym

# --- imports from modules ---
from experiment_config import load_config, save_resolved_config, ExperimentConfig

from acorl.envs.seeker.seeker_exploration import SeekerExplorationEnvConfig
from rl_competition.competition.environment import create_exploration_seeker
from acorl.envs.constraints.seeker import SeekerInputSetPolytopeCalculator
from acorl.env_wrapper.adaption_fn import ConditionalAdaptionEnvWrapper
from acorl.acrl_algos.alpha_projection.mapping import alpha_projection_interface_fn

from MCTS_AC import MCTSPlanner_AC
from network import SeekerAlphaZeroNet
from utils import (
    ReplayBufferHybrid,
    topk_from_policy,
    train_step_mle,
    grow_replay,
    env_set_state,
)


# -------------------------
# Reproducibility
# -------------------------
def set_global_seeds(seed: int, deterministic_torch: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -------------------------
# Run directory naming and print helpers
# -------------------------
def short_config_hash(cfg: ExperimentConfig) -> str:
    blob = json.dumps(asdict(cfg), sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:8]


def make_run_dir(cfg: ExperimentConfig) -> Path:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    h = short_config_hash(cfg)
    name = f"{cfg.run.experiment_name}__{ts}__seed{cfg.run.seed}__{h}"
    return Path(cfg.output.root_dir) / name

def format_hms(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def make_log_prefix(cfg: ExperimentConfig, env_variant: str, run_dir) -> str:
    # run_dir is .../<parent>/<2d|3d>
    parent = Path(run_dir).parent.name  # e.g. base__seed42__9567da53
    short = parent.split("__")[-1] if "__" in parent else parent
    pid = os.getpid()
    return f"[{cfg.run.experiment_name}|{env_variant}|{short}|pid{pid}]"


# -------------------------
# Env + step_fn setup
# -------------------------
def build_envs_and_step_fn_3d():
    env_real, env_config = create_exploration_seeker()
    obs0, info0 = env_real.reset()

    # simulation env for MCTS step_fn
    env_sim, env_config_sim = create_exploration_seeker()
    env_sim = env_sim.unwrapped

    constraint_calculator = SeekerInputSetPolytopeCalculator(env_config=env_config_sim)
    env_sim_AC = ConditionalAdaptionEnvWrapper(
        env_sim,
        constraint_calculator.compute_relevant_input_set,
        constraint_calculator.compute_fail_safe_input,
        constraint_calculator.get_set_representation(),
        alpha_projection_interface_fn,
    )

    def sync_conditional_adaption_wrapper(env_wrapped, obs, *, constraint_calculator):
        info = {"boundary_size": float(getattr(env_wrapped.unwrapped, "_size", 10.0))}
        info["relevant_input_set"] = constraint_calculator.compute_relevant_input_set(obs, info)
        info["fail_safe_input"] = constraint_calculator.compute_fail_safe_input(obs, info)
        if hasattr(env_wrapped.unwrapped, "_boundary_size"):
            info["boundary_size"] = env_wrapped.unwrapped._boundary_size
        env_wrapped._previous_obs = obs
        env_wrapped._previous_info = info

    def step_fn(node, action):
        env_set_state(env_sim_AC, node, num_obstacles=env_config.num_obstacles)

        obs = np.asarray(node.state, dtype=env_sim._dtype)
        sync_conditional_adaption_wrapper(
            env_sim_AC,
            obs,
            constraint_calculator=constraint_calculator,
        )

        action = np.asarray(action, dtype=env_sim_AC.unwrapped._dtype)
        next_obs, reward, terminated, truncated, info = env_sim_AC.step(action)
        next_obs = np.array(next_obs, copy=True)

        done = bool(terminated or truncated)
        next_coin_collected = bool(getattr(env_sim_AC.unwrapped, "_coin_collected", False))
        return next_obs, next_coin_collected, reward, done, info

    obs_dim = env_real.observation_space.shape[0]
    action_dim = env_real.action_space.shape[0]
    return env_real, env_sim, env_sim_AC, env_config, obs_dim, action_dim, step_fn

def build_envs_and_step_fn_2d():
    env_config = SeekerExplorationEnvConfig(
        randomize=True,
        num_obstacles=10,
        dim=2,
        log=False,
    )
    env_real = gym.make(env_config.id, **env_config.model_dump(exclude={'id'}))
    obs0, info0 = env_real.reset()

    # simulation env for MCTS step_fn
    env_sim = gym.make(env_config.id, **env_config.model_dump(exclude={'id'}))
    env_sim = env_sim.unwrapped

    constraint_calculator = SeekerInputSetPolytopeCalculator(env_config=env_config)
    env_sim_AC = ConditionalAdaptionEnvWrapper(
        env_sim,
        constraint_calculator.compute_relevant_input_set,
        constraint_calculator.compute_fail_safe_input,
        constraint_calculator.get_set_representation(),
        alpha_projection_interface_fn,
    )

    def sync_conditional_adaption_wrapper(env_wrapped, obs, *, constraint_calculator):
        info = {"boundary_size": float(getattr(env_wrapped.unwrapped, "_size", 10.0))}
        info["relevant_input_set"] = constraint_calculator.compute_relevant_input_set(obs, info)
        info["fail_safe_input"] = constraint_calculator.compute_fail_safe_input(obs, info)
        if hasattr(env_wrapped.unwrapped, "_boundary_size"):
            info["boundary_size"] = env_wrapped.unwrapped._boundary_size
        env_wrapped._previous_obs = obs
        env_wrapped._previous_info = info

    def step_fn(node, action):
        env_set_state(env_sim_AC, node, num_obstacles=env_config.num_obstacles)

        obs = np.asarray(node.state, dtype=env_sim._dtype)
        sync_conditional_adaption_wrapper(
            env_sim_AC,
            obs,
            constraint_calculator=constraint_calculator,
        )

        action = np.asarray(action, dtype=env_sim_AC.unwrapped._dtype)
        next_obs, reward, terminated, truncated, info = env_sim_AC.step(action)
        next_obs = np.array(next_obs, copy=True)

        done = bool(terminated or truncated)
        next_coin_collected = bool(getattr(env_sim_AC.unwrapped, "_coin_collected", False))
        return next_obs, next_coin_collected, reward, done, info

    obs_dim = env_real.observation_space.shape[0]
    action_dim = env_real.action_space.shape[0]
    return env_real, env_sim, env_sim_AC, env_config, obs_dim, action_dim, step_fn



# -------------------------
# Scheduling helpers
# -------------------------
def set_bootstrap_state(planner, cfg: ExperimentConfig, global_step: int):
    b = cfg.schedules.bootstrap
    if not b.enabled:
        planner.deactivate_bootstrapping()
        return

    if b.start_step is None:
        planner.activate_bootstrapping()
        return

    if global_step < b.start_step:
        planner.deactivate_bootstrapping()
    else:
        planner.activate_bootstrapping()


def apply_schedules(
    cfg: ExperimentConfig,
    global_step: int,
    optimizer: optim.Optimizer,
    replay_buffer: ReplayBufferHybrid,
) -> Tuple[ReplayBufferHybrid, int]:
    """
    Applies LR schedule and replay/batch schedule if there's an entry exactly at global_step.
    Returns: (maybe_new_replay_buffer, current_batch_size)
    """
    # Default batch size is taken from the first replay schedule entry if present.
    batch_size = cfg.schedules.replay[0].batch_size if cfg.schedules.replay else 32

    # LR schedule
    for s in cfg.schedules.lr:
        if s.step == global_step:
            for pg in optimizer.param_groups:
                pg["lr"] = float(s.value)

    # Replay + batch schedule
    for s in cfg.schedules.replay:
        if s.step == global_step:
            batch_size = int(s.batch_size)
            # Grow replay (your helper already exists)
            replay_buffer = grow_replay(replay_buffer, new_capacity=int(s.capacity))

    return replay_buffer, batch_size


# -------------------------
# Main training
# -------------------------
def train_one(cfg: ExperimentConfig, run_dir: Path, env_variant: str):
    # device
    if cfg.run.device == "cpu":
        device = torch.device("cpu")
    elif cfg.run.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_global_seeds(cfg.run.seed, deterministic_torch=cfg.run.deterministic_torch)

    # create dirs
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # save config immediately = write resolved config to the run folder right now
    # so the run is self-contained even if the process crashes later.
    if cfg.output.save_config_json:
        save_resolved_config(cfg, run_dir, filename="config.json")
        (run_dir / "meta.json").write_text(json.dumps({"env_variant": env_variant}, indent=2))

    # envs + step_fn
    if env_variant == "3d":
        env_real, env_sim, env_sim_AC, env_config, obs_dim, action_dim, step_fn = build_envs_and_step_fn_3d()
    else:
        env_real, env_sim, env_sim_AC, env_config, obs_dim, action_dim, step_fn = build_envs_and_step_fn_2d()

    # net + optimizer + planner
    net = SeekerAlphaZeroNet(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=tuple(cfg.net.hidden_sizes)).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=cfg.optim.learning_rate, weight_decay=cfg.optim.weight_decay)

    planner = MCTSPlanner_AC(
        net=net,
        device=str(device),
        step_fn=step_fn,
        num_simulations=cfg.mcts.num_simulations,
        cpuct=cfg.mcts.cpuct,
        gamma=cfg.mcts.gamma_mcts,
        pw_k=cfg.mcts.pw_k,
        pw_alpha=cfg.mcts.pw_alpha,
        max_depth=cfg.mcts.max_depth,
        temperature=cfg.mcts.temperature,
        rng=np.random.default_rng(cfg.run.seed),
        K_uniform_per_node=cfg.action_sampling.K_uniform_per_node,
        warmstart_iters=cfg.action_sampling.warmstart_steps,
        novelty_eps=cfg.action_sampling.novelty_eps,
        novelty_metric=cfg.action_sampling.novelty_metric,
        num_candidates=cfg.action_sampling.num_candidates,
        diversity_lambda=cfg.action_sampling.diversity_lambda,
        diversity_sigma=cfg.action_sampling.diversity_sigma,
        policy_beta=cfg.action_sampling.policy_beta,
        max_resample_attempts=cfg.action_sampling.max_resample_attempts,
    )

    # replay
    replay_buffer = ReplayBufferHybrid(
        capacity=cfg.replay.capacity,
        obs_dim=obs_dim,
        action_dim=action_dim,
        K_max=cfg.replay.K_max,
        seed=cfg.run.seed,
    )

    global_step = 0
    it = 0
    episode_return = 0.0
    episode_len = 0
    logs = {"loss_total": []}  # expand later

    obs, info = env_real.reset()
    coin_collected = bool(getattr(env_real.unwrapped, "_coin_collected", False))

    # set initial bootstrap state (matches your explicit deactivate in the nobootstrap file)
    set_bootstrap_state(planner, cfg, global_step)

    start_time = time.perf_counter()
    prefix = make_log_prefix(cfg, env_variant, run_dir)
    print(f"{prefix} START TRAINING | device={device} | run_dir={run_dir}")

    batch_size = cfg.schedules.replay[0].batch_size if cfg.schedules.replay else 32

    while global_step < cfg.train.total_env_steps:
        # schedules (lr/replay/batch), driven by config
        replay_buffer, batch_size = apply_schedules(cfg, global_step, optimizer, replay_buffer)

        # bootstrap schedule (driven by config)
        set_bootstrap_state(planner, cfg, global_step)

        # ---- 1 step collect ----
        root = planner.search(obs, coin_collected=coin_collected)
        probs, actions = planner.policy_from_root(root)
        probs, actions = topk_from_policy(probs, actions, replay_buffer.K_max)

        mu_star, log_std_star, z_mcts = planner.targets_from_root(root)

        action = planner.act(root, training=True)
        next_obs, reward, terminated, truncated, info = env_real.step(action)
        done = bool(terminated or truncated)
        next_coin = bool(getattr(env_real.unwrapped, "_coin_collected", False))

        replay_buffer.add(
            obs,
            mu_star,
            log_std_star,
            float(z_mcts),
            0.0,  # z_mc unused in your current loop
            actions,
            probs,
        )

        episode_return += float(reward)
        episode_len += 1
        global_step += 1
        obs = next_obs
        coin_collected = next_coin

        # ---- online train ----
        if len(replay_buffer) >= cfg.train.min_replay_factor * batch_size:
            for _ in range(cfg.train.train_updates_per_step):
                batch = replay_buffer.sample(batch_size, device=device)
                loss_dict = train_step_mle(
                    net=net,
                    optimizer=optimizer,
                    batch=batch,
                    value_target=cfg.loss.value_target,
                    w_value=cfg.loss.value_loss_weight,
                    w_policy=cfg.loss.policy_loss_weight,
                    grad_clip_norm=cfg.train.grad_clip_norm,
                )
                # store what you want
                logs["loss_total"].append(loss_dict.get("loss_total", None))

        # ---- episode end ----
        if done or episode_len >= cfg.env.max_episode_steps:
            obs, info = env_real.reset()
            coin_collected = bool(getattr(env_real.unwrapped, "_coin_collected", False))
            episode_return = 0.0
            episode_len = 0

        # ---- step bookkeeping in planner for warmstart ----
        planner.set_training_iter(global_step) #TODO: probably a better solution here with scheduler of bootsrap combination

        # ---- checkpoints ----
        if cfg.output.save_checkpoints and (global_step % cfg.output.checkpoint_every_steps == 0):
            ckpt_path = ckpt_dir / f"ckpt_step_{global_step:09d}.pt"
            torch.save(
                {
                    "step": global_step,
                    "net": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng_seed": cfg.run.seed,
                    # keep for backwards-compat with your evaluation script approach:
                    # (eventually you can store the full nested config too)
                    "cfg": asdict(cfg),
                    "python_random_state": random.getstate(),
                    "numpy_random_state": np.random.get_state(),
                    "torch_random_state": torch.get_rng_state(),
                    "torch_cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
                ckpt_path,
            )
            elapsed = time.perf_counter() - start_time
            print(f"{prefix} step={global_step} saved={ckpt_path.name} elapsed={format_hms(elapsed)}")

    env_real.close()
    env_sim.close()

    (done_path := run_dir / "DONE.json").write_text(
    json.dumps({"status": "ok", "final_step": global_step}, indent=2),
    encoding="utf-8",
    ) # done marker



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to config JSON")
    ap.add_argument("--run-dir", type=str, default=None, help="Optional: override run directory (parent)")
    ap.add_argument("--env", type=str, default="3d", choices=["2d", "3d", "both"], help="Which env variant to train")
    args = ap.parse_args()

    cfg = load_config(args.config)

    parent_run_dir = Path(args.run_dir) if args.run_dir else make_run_dir(cfg)
    parent_run_dir.mkdir(parents=True, exist_ok=True)

    if args.env in ("2d", "both"):
        train_one(cfg, parent_run_dir / "2d", env_variant="2d")

    if args.env in ("3d", "both"):
        train_one(cfg, parent_run_dir / "3d", env_variant="3d")



if __name__ == "__main__":
    main()
