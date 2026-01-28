# --- Standard libs ---
import os, pickle
import math
import random
from dataclasses import dataclass
from typing import Dict, Any
import time

# --- Scientific stack ---
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --- ACORL and RL ---
from acorl.envs.seeker.seeker import SeekerEnv, SeekerEnvConfig

# --- Own Code ---
from MCTS_AC import MCTSPlanner_AC
from network import SeekerAlphaZeroNet
from utils import (
    ReplayBufferHybrid,
    topk_from_policy,
    collect_one_episode_hybrid,
    train_step_mle,
    train_step_mcts_distill,
    train_step_hybrid,
    run_eval_episodes,
    run_debug_eval_episode,
    grow_replay,
    env_set_state,
)

print("lower exploration")

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --- Reproducibility ---
def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


RNG_SEED = 42
set_global_seeds(RNG_SEED)

BASE_DIR = "eval_checkpoints"
os.makedirs(BASE_DIR, exist_ok=True)

run_name = f"ckpt_3D_exploration_AC_stepwise_cpuct05{time.strftime('%d%m%Y_%H%M%S', time.localtime())}"
EVAL_DIR = os.path.join(BASE_DIR, run_name)
os.makedirs(EVAL_DIR, exist_ok=True)

# === Config ===

@dataclass
class Config:
    # ========================
    # Environment
    # ========================
    max_episode_steps: int = 300  # max steps in environment before cut off (goal not reached, obstacle not crashed into --> prevent forever stepping)

    # ========================
    # MCTS core
    # ========================
    num_simulations: int = 200    # Number of MCTS simulations per real environment step
    cpuct: float = 0.5            # Exploration vs exploitation tradeoff in PUCT; Higher -> more exploration guided by policy prior
    max_depth: int = 64           # Safety cap on tree depth during a simulation

    # For root action selection / Action sampling temperature at root
    # >1.0 = more stochastic, 1.0 = proportional to visits, ~0 = greedy
    temperature: float = 0.5

    # ========================
    # Progressive Widening
    # ========================
    pw_k: float = 2.0
    # Controls how many actions are allowed per node:
    #   K_max = pw_k * N(s)^pw_alpha
    pw_alpha: float = 0.5
    # Growth rate of branching factor
    # 0.5 is common; smaller = more conservative expansion

    # ========================
    # Action sampling (baseline, non-fancy, but no duplicates)
    # ========================
    # --- Uniform warmstart ---
    # No uniform warmstart, no diversity scoring
    K_uniform_per_node: int = 8
    # First K children per node are sampled uniformly in [-1,1]^2
    # Set to 0 to disable
    warmstart_iters: int = 20
    # Number of *training iterations* during which ALL nodes use uniform sampling
    # 0 disables global warmstart; use this if you want uniform sampling only early in training

    # --- Novelty reject (hard deduplication) ---
    # Deduplicate actions (keep this ON to satisfy “no duplicate actions”)
    novelty_eps: float = 1e-3      # small but > 0
    # Minimum distance between actions to be considered "new"
    # In [-1,1]^2, values around 0.05–0.15 are reasonable
    # Set <=0 to disable
    novelty_metric: str = "l2"
    # Distance metric for novelty check:
    # "linf" = max(|dx|, |dy|)  (good for box action spaces)
    # "l2"   = Euclidean distance

    # --- Diversity scoring (soft repulsion) ---
    # Disable candidate scoring / diversity
    num_candidates: int = 1
    # Number of candidate actions sampled before choosing the best
    # <=1 disables diversity scoring
    diversity_lambda: float = 0.0
    # Strength of diversity penalty
    # Higher -> stronger push away from already-sampled actions
    # Set <=0 to disable
    diversity_sigma: float = 0.25  # unused
    # Length scale for diversity penalty
    # Roughly: how far actions must be before they stop "repelling" each other
    policy_beta: float = 1.0       # unused
    # Weight of policy log-probability in candidate scoring
    # Higher -> follow policy more closely
    # Lower -> prioritize diversity more

    # --- Resampling control ---
    max_resample_attempts: int = 16
    # How many times expansion may retry to find a novel action
    # If all fail, expansion is declined and MCTS falls back to selection
    
    # ========================
    # Training
    # ========================
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    train_steps_per_iter: int = 50    # Gradient updates per outer iteration

    # (Only used by our baseline loss function)
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0  # applies to mu/log_std regression

    # ========================
    # Data collection
    # ========================
    collect_episodes_per_iter: int = 10     # Number of real env episodes collected per training iteration
    replay_buffer_capacity: int = 500
    gamma_mcts: float = 0.85     # Discount factor for return backup in MCTS
    gamma_mc = 0.9

    # ========================
    # Logging / evaluation
    # ========================
    eval_every: int = 25
    eval_episodes: int = 10   # use 10 fixed seeds for smoother eval curves


cfg = Config()


# === ENVIRONMENTS ===

from rl_competition.competition.environment import create_exploration_seeker
from acorl.envs.constraints.seeker import SeekerInputSetPolytopeCalculator
from acorl.env_wrapper.adaption_fn import ConditionalAdaptionEnvWrapper
from acorl.acrl_algos.alpha_projection.mapping import alpha_projection_interface_fn

# --- Real environment for rollouts / data collection ---

env_real, env_config = create_exploration_seeker()
obs0, info0 = env_real.reset()

obs_dim = env_real.observation_space.shape[0]
action_dim = env_real.action_space.shape[0]

print("obs_dim:", obs_dim, "action_dim:", action_dim)
print("action_space:", env_real.action_space)

# --- Simulation environment for MCTS step_fn ---
env_sim, env_config_sim = create_exploration_seeker()
env_sim = env_sim.unwrapped

constraint_calculator = SeekerInputSetPolytopeCalculator(env_config=env_config_sim)
env_sim_AC = ConditionalAdaptionEnvWrapper(env_sim, 
                                        constraint_calculator.compute_relevant_input_set,
                                        constraint_calculator.compute_fail_safe_input,
                                        constraint_calculator.get_set_representation(),
                                        alpha_projection_interface_fn)

def sync_conditional_adaption_wrapper(
    env_wrapped,
    obs,
    *,
    constraint_calculator,
):
    """
    Sync ConditionalAdaptionEnvWrapper cache after env_set_state().
    """
    info = {
        "boundary_size": float(getattr(env_wrapped.unwrapped, "_size", 10.0)),  # SeekerEnv uses _size for boundary :contentReference[oaicite:2]{index=2}
    }

    # Compute constraint info for THIS obs
    info["relevant_input_set"] = constraint_calculator.compute_relevant_input_set(obs, info)
    info["fail_safe_input"] = constraint_calculator.compute_fail_safe_input(obs, info)

    # Optional but harmless
    if hasattr(env_wrapped.unwrapped, "_boundary_size"):
        info["boundary_size"] = env_wrapped.unwrapped._boundary_size

    env_wrapped._previous_obs = obs
    env_wrapped._previous_info = info

def step_fn(node, action):
    """
    MCTS transition function: set env_sim to `state`, take `action`, return next_state/reward/done/info.
    USES ACTION CONSTRAINED ENVIRONMENT
    Returns: next_state, reward, done, info  (matching MCTSPlanner expectations)
    """
    # 1) teleport base env
    env_set_state(env_sim_AC, node, num_obstacles=env_config.num_obstacles)

    # 2) sync wrapper cache
    obs = np.asarray(node.state, dtype=env_sim._dtype)
    sync_conditional_adaption_wrapper(
        env_sim_AC,
        obs,
        constraint_calculator=constraint_calculator,
    )


    # 3) step the WRAPPED env
    action = np.asarray(action, dtype=env_sim_AC.unwrapped._dtype)
    next_obs, reward, terminated, truncated, info = env_sim_AC.step(action)

    next_obs = np.array(next_obs, copy=True) #break reference to internal env buffer (??)
    
    done = bool(terminated or truncated)
    next_coin_collected = bool(getattr(env_sim_AC.unwrapped, "_coin_collected", False))
    
    return next_obs, next_coin_collected, reward, done, info 

# === Network ===
net = SeekerAlphaZeroNet(obs_dim=obs_dim, action_dim=action_dim).to(device)

# Optional: print one forward pass sanity
obs_t = torch.from_numpy(obs0).float().unsqueeze(0).to(device)
with torch.no_grad():
    mu_t, log_std_t, v_t = net(obs_t)

print("mu:", mu_t.cpu().numpy())
print("log_std:", log_std_t.cpu().numpy())
print("v:", v_t.item())

# --- Optimizer (we'll use later) ---
optimizer = optim.AdamW(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

# === PLANNER === 
planner = MCTSPlanner_AC(
    net=net,
    device=str(device),
    step_fn=step_fn,
    num_simulations=cfg.num_simulations,
    cpuct=cfg.cpuct,
    gamma=cfg.gamma_mcts,
    pw_k=cfg.pw_k,
    pw_alpha=cfg.pw_alpha,
    max_depth=cfg.max_depth,
    temperature=cfg.temperature,
    rng=np.random.default_rng(RNG_SEED),
    
    K_uniform_per_node=cfg.K_uniform_per_node,
    warmstart_iters=cfg.warmstart_iters,
    novelty_eps=cfg.novelty_eps,
    novelty_metric=cfg.novelty_metric,
    num_candidates=cfg.num_candidates,
    diversity_lambda=cfg.diversity_lambda,
    diversity_sigma=cfg.diversity_sigma,
    policy_beta=cfg.policy_beta,
    max_resample_attempts=cfg.max_resample_attempts,
)

# === LOGS ===

# Logging containers (easy to plot later)
logs = {
    "loss_total": [],
    "loss_value": [],
    "loss_policy": [],
    "loss_policy_distill": [],
    "loss_mu": [],
    "loss_log_std": [],
    "ep_return": [],
    "ep_length": [],
    "eval_return_mean": [],
    "eval_return_std": [],
    "eval_length_mean": [],
    "eval_length_succes": [],
    "eval_length_collision": [],
    "success_rate": [],
    "collision_rate": [],
    "max_step_rate": [],
    "iter_idx_eval": [],
}

# === TRAINING ===
total_env_steps = 200_000            # like paper x-axis
eval_every_steps = 10_000            # evaluate periodically
train_updates_per_step = 1           # 1 SGD update per env step

# Buffer - Start small and grow
replay_buffer = ReplayBufferHybrid(
    capacity=cfg.replay_buffer_capacity,     # Option A: small replay
    obs_dim=obs_dim,
    action_dim=action_dim,
    K_max=16,
    seed=RNG_SEED,
)

global_step = 0
it = 0
episode_return = 0.0
episode_len = 0

obs, info = env_real.reset()
coin_collected = bool(getattr(env_real.unwrapped, "_coin_collected", False))

start_time = time.perf_counter()
print("START ONLINE TRAINING")

while global_step < total_env_steps:
    # optional: grow batch size + replay
    # (simple schedule – adjust freely)
    if global_step == 20_000:
        for pg in optimizer.param_groups:
            pg["lr"] = 2.5e-4
        new_cap = 2000
        new_batch = 32
        cfg.batch_size = new_batch
        replay_buffer = grow_replay(replay_buffer, new_capacity=new_cap)
        print(f"GROW batch_size -> {new_batch}, replay -> {new_cap}")
    elif global_step == 60_000:
        for pg in optimizer.param_groups:
            pg["lr"] = 2e-4
        new_cap = 5000
        new_batch = 64
        cfg.batch_size = new_batch
        replay_buffer = grow_replay(replay_buffer, new_capacity=new_cap)
        print(f"GROW batch_size -> {new_batch}, replay -> {new_cap}")
    elif global_step == 120_000:
        for pg in optimizer.param_groups:
            pg["lr"] = 1.5e-4
        new_cap = 10000
        new_batch = 64
        cfg.batch_size = new_batch
        replay_buffer = grow_replay(replay_buffer, new_capacity=new_cap)
        print(f"GROW batch_size -> {new_batch}, replay -> {new_cap}")

    # ---- 1 step collect (MCTS + env step + add to replay) ----
    root = planner.search(obs, coin_collected=coin_collected)
    probs, actions = planner.policy_from_root(root)  # probs: (K,), actions: (K, action_dim)
    probs, actions = topk_from_policy(probs, actions, replay_buffer.K_max)

    mu_star, log_std_star, z_mcts = planner.targets_from_root(root)

    # execute action
    action = planner.act(root, training=True)
    next_obs, reward, terminated, truncated, info = env_real.step(action)
    done = bool(terminated or truncated)
    next_coin = bool(getattr(env_real.unwrapped, "_coin_collected", False))

    replay_buffer.add(
        obs,
        mu_star,
        log_std_star,
        float(z_mcts),
        0.0, # z_mc unused
        actions,
        probs,
    )

    step_record = {
        "z_mcts": float(z_mcts),
        "reward": float(reward),
        "done": done,
    }

    episode_return += reward
    episode_len += 1
    global_step += 1

    obs = next_obs
    coin_collected = next_coin

    # ---- Online train (1 minibatch SGD step) ----
    if len(replay_buffer) >= 10*cfg.batch_size: # start training when replay_buffer is full and can be sampled
        for _ in range(train_updates_per_step):
            batch = replay_buffer.sample(cfg.batch_size, device=device)
            loss_dict = train_step_mle(
                net=net,
                optimizer=optimizer,
                batch=batch,
                value_target="mcts",
                w_value=cfg.value_loss_weight,
                w_policy=cfg.policy_loss_weight,
                grad_clip_norm=1.0,
            )
            for k, v in loss_dict.items():
                logs[k].append(v)

    # ---- Episode end / max steps ----
    if done or episode_len >= cfg.max_episode_steps:
        logs["ep_return"].append(float(episode_return))
        logs["ep_length"].append(int(episode_len))
        if episode_return <= -100:
            print("CRASH:", info)
            print("agent pos:", env_real.unwrapped._agent_position)


        obs, info = env_real.reset()
        coin_collected = bool(getattr(env_real.unwrapped, "_coin_collected", False))
        print(f"Train episode ended: Length: {episode_len}, Reward: {episode_return}")
        if episode_return > 0:
            print("                POSITIVE REWARD")
        episode_return = 0.0
        episode_len = 0
        
    # ---- disable warmstart ----
    if global_step % 1000:
        it += 1
        planner.set_training_iter(it)

    # ---- Periodic checkpoint/eval ----
    if global_step % eval_every_steps == 0:
        elapsed = time.perf_counter() - start_time
        print(f"[step {global_step}] elapsed={elapsed:.1f}s | last_loss={logs['loss_total'][-1] if logs['loss_total'] else None}")

        ckpt_path = os.path.join(EVAL_DIR, f"ckpt_step_{global_step:09d}.pt")
        torch.save({
            "step": global_step,
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng_seed": RNG_SEED,
            "cfg": cfg.__dict__,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, ckpt_path)
    
    if global_step % 100 == 0:
        print("global step: ", global_step)
        now = time.perf_counter()
        print(f"time: {now-start_time}")
        last_time = now