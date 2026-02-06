# eval.py
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# Use same env-building logic as train.py
import gymnasium as gym

from acorl.envs.seeker.seeker_exploration import SeekerExplorationEnvConfig
from rl_competition.competition.environment import create_exploration_seeker
from acorl.envs.constraints.seeker import SeekerInputSetPolytopeCalculator
from acorl.env_wrapper.adaption_fn import ConditionalAdaptionEnvWrapper
from acorl.acrl_algos.alpha_projection.mapping import alpha_projection_interface_fn

from MCTS_AC import MCTSPlanner_AC
from network import SeekerAlphaZeroNet
from utils import (
    env_set_state,
    serialize_mcts_tree,
    pack_serialized_trees,
    make_slim_trace,
)

from experiment_config import load_config, ExperimentConfig


# -------------------------
# Env builders (copied from train.py pattern)
# -------------------------

def build_envs_and_step_fn_3d():
    env_real, env_config = create_exploration_seeker()
    _obs0, _info0 = env_real.reset()

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
            env_sim_AC, obs, constraint_calculator=constraint_calculator
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
    # matches your train.py approach
    env_config = SeekerExplorationEnvConfig(
        randomize=True,
        num_obstacles=10,
        dim=2,
        log=False,
    )
    env_real = gym.make(env_config.id, **env_config.model_dump(exclude={"id"}))
    _obs0, _info0 = env_real.reset()

    env_sim = gym.make(env_config.id, **env_config.model_dump(exclude={"id"}))
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
            env_sim_AC, obs, constraint_calculator=constraint_calculator
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
# Evaluation loop (full-depth trees, slim storage)
# -------------------------

@torch.no_grad()
def eval_checkpoint(
    ckpt_path: Path,
    *,
    env_variant: str,
    seeds: Sequence[int],
    max_steps: int,
    tree_state_mode: str,
    save_dir: Path,
):
    # Locate run_dir and config.json
    # .../<run_dir>/checkpoints/<ckpt>.pt
    run_dir = ckpt_path.parent.parent
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find resolved config at {config_path}")

    cfg: ExperimentConfig = load_config(config_path)

    # Build envs
    if env_variant == "3d":
        env_eval, env_sim, env_sim_AC, env_config, obs_dim, action_dim, step_fn = build_envs_and_step_fn_3d()
    elif env_variant == "2d":
        env_eval, env_sim, env_sim_AC, env_config, obs_dim, action_dim, step_fn = build_envs_and_step_fn_2d()
    else:
        raise ValueError("env_variant must be '2d' or '3d'")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Rebuild net
    # Prefer inferred hidden sizes from state_dict (robust), else config
    sd = ckpt["net"]
    inferred_hidden: List[int] = []
    for k, v in sd.items():
        if k.endswith("weight") and v.ndim == 2:
            inferred_hidden.append(int(v.shape[0]))
    # inferred_hidden contains many layers; we only need internal ones.
    # If this heuristic is too rough for your net class, just use cfg.net.hidden_sizes.
    hidden_sizes = cfg.net.hidden_sizes

    net = SeekerAlphaZeroNet(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=tuple(hidden_sizes))
    net.load_state_dict(sd)
    net.eval()

    # Planner: use cfg (nested)
    planner = MCTSPlanner_AC(
        net=net,
        device="cpu",
        step_fn=step_fn,
        num_simulations=cfg.mcts.num_simulations,
        cpuct=cfg.mcts.cpuct,
        gamma=cfg.mcts.gamma_mcts,
        pw_k=cfg.mcts.pw_k,
        pw_alpha=cfg.mcts.pw_alpha,
        max_depth=cfg.mcts.max_depth,
        temperature=cfg.mcts.temperature,
        rng=np.random.default_rng(int(cfg.run.seed)),
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

    # Run evaluation episodes
    per_seed_slim: List[dict] = []
    summary_rows: List[dict] = []

    # Ensure output dirs
    save_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        obs, info = env_eval.reset(seed=int(seed))
        coin = bool(getattr(env_eval.unwrapped, "_coin_collected", False))

        trace = {
            "seed": int(seed),
            "info": info,
            "states": [obs],
            "chosen_idx": [],
            "actions": [],
            "rewards": [],
            "mc_return": [],
            # store serialized trees per step (packed later)
            "trees": [],
        }

        ep_return = 0.0
        terminal_reward = None

        for t in range(int(max_steps)):
            root = planner.search(obs, coin_collected=coin)
            action = planner.act(root, training=False)

            # chosen idx among root children (same logic as your current run_eval_episodes) :contentReference[oaicite:5]{index=5}
            if len(root.children) > 0:
                idx = int(np.argmin([np.max(np.abs(ch.action - action)) for ch in root.children]))
            else:
                idx = -1

            trace["chosen_idx"].append(idx)
            trace["actions"].append(np.asarray(action, dtype=np.float32))

            # Serialize full tree (full depth)
            tree = serialize_mcts_tree(
                root,
                num_obstacles=int(getattr(env_config, "num_obstacles", 10)),
                state_mode=tree_state_mode,
            )
            trace["trees"].append(tree)

            obs, reward, terminated, truncated, info = env_eval.step(action)
            coin = bool(getattr(env_eval.unwrapped, "_coin_collected", False))

            trace["states"].append(obs)
            trace["rewards"].append(float(reward))
            ep_return += float(reward)

            if terminated or truncated:
                terminal_reward = float(reward)
                break

        # MC return-to-go (same idea as your current implementation) :contentReference[oaicite:6]{index=6}
        G = 0.0
        mc = [0.0] * len(trace["rewards"])
        for i in range(len(trace["rewards"]) - 1, -1, -1):
            r = trace["rewards"][i]
            G = r + 0.99 * G
            mc[i] = float(G)
        trace["mc_return"] = mc

        # Save per-seed packed trees
        packed = pack_serialized_trees(trace["trees"])
        np.savez_compressed(save_dir / f"trees_seed_{int(seed)}.npz", **packed)

        # Slim trace for plotting the path
        slim = make_slim_trace(
            trace,
            num_obstacles=int(getattr(env_config, "num_obstacles", 10)),
            goal_reward=100.0,
            collision_reward=-100.0,
            tol=1e-6,
            include_chosen_idx=True,
        )
        per_seed_slim.append(slim)

        # Row stats
        summary_rows.append(
            {
                "seed": int(seed),
                "ep_len": int(slim["ep_len"]),
                "total_return": float(slim["total_return"]),
                "terminated_goal": bool(slim["terminated_goal"]),
                "terminated_crash": bool(slim["terminated_crash"]),
                "terminated_timeout": bool(slim["terminated_timeout"]),
            }
        )

    # Aggregate summary
    returns = np.asarray([r["total_return"] for r in summary_rows], dtype=np.float32)
    success = np.asarray([r["terminated_goal"] for r in summary_rows], dtype=np.bool_)
    crash = np.asarray([r["terminated_crash"] for r in summary_rows], dtype=np.bool_)

    summary = {
        "checkpoint": str(ckpt_path),
        "step": int(ckpt.get("step", -1)),
        "env_variant": env_variant,
        "num_seeds": int(len(seeds)),
        "max_steps": int(max_steps),
        "tree_state_mode": tree_state_mode,
        "return_mean": float(np.mean(returns)) if len(returns) else 0.0,
        "return_std": float(np.std(returns)) if len(returns) else 0.0,
        "success_rate": float(np.mean(success)) if len(success) else 0.0,
        "crash_rate": float(np.mean(crash)) if len(crash) else 0.0,
        "per_seed": summary_rows,
    }

    # Write summary + slim traces
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (save_dir / "slim_traces.json").write_text(json.dumps(per_seed_slim, indent=2), encoding="utf-8")

    env_eval.close()
    env_sim.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt")
    ap.add_argument("--env", type=str, required=True, choices=["2d", "3d"], help="Env variant (2d/3d)")
    ap.add_argument("--seeds", type=str, default="1000:1049", help="Seed range 'a:b' inclusive, or comma list")
    ap.add_argument("--max-steps", type=int, default=None, help="Max steps per episode (default: cfg.env.max_episode_steps)")
    ap.add_argument("--tree-state", type=str, default="full", choices=["full", "agentpos", "none"],
                    help="What to store per MCTS node")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    run_dir = ckpt_path.parent.parent
    cfg = load_config(run_dir / "config.json")

    # parse seeds
    if ":" in args.seeds:
        a, b = args.seeds.split(":")
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    max_steps = int(args.max_steps) if args.max_steps is not None else int(cfg.env.max_episode_steps)

    out_dir = run_dir / "eval" / ckpt_path.stem
    eval_checkpoint(
        ckpt_path,
        env_variant=args.env,
        seeds=seeds,
        max_steps=max_steps,
        tree_state_mode=args.tree_state,
        save_dir=out_dir,
    )
    print(f"Wrote eval to: {out_dir}")


if __name__ == "__main__":
    main()
