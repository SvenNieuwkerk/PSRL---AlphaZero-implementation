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
from MCTS import MCTSPlanner
from network import SeekerAlphaZeroNet
from utils import (
    ReplayBufferHybrid,
    collect_one_episode_hybrid,
    train_step_mle,
    train_step_mcts_distill,
    train_step_hybrid,
    run_eval_episodes,
    run_debug_eval_episode,
    grow_replay,
    env_set_state,
)
from plot_utils import (
#    decode_obs,
    plot_seeker_obs,
    plot_seeker_trajectory,
#    collect_tree_edges,
    plot_mcts_tree_xy_limited,
    inspect_debug_trace_xy,
    plot_dbg_step,
)

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

EVAL_DIR = "eval_traces_3D_exploration"
#s = time.strftime("%d%m%Y_%H%M%S", time.localtime())
s = "temp2"
EVAL_DIR = EVAL_DIR+"_"+s
os.makedirs(EVAL_DIR, exist_ok=True)

# === Config ===

@dataclass
class Config:
    # ========================
    # Environment
    # ========================
    max_episode_steps: int = 200  # max steps in environment before cut off (goal not reached, obstacle not crashed into --> prevent forever stepping)

    # ========================
    # MCTS core
    # ========================
    num_simulations: int = 400    # Number of MCTS simulations per real environment step
    cpuct: float = 1.5            # Exploration vs exploitation tradeoff in PUCT; Higher -> more exploration guided by policy prior
    max_depth: int = 64           # Safety cap on tree depth during a simulation

    # For root action selection / Action sampling temperature at root
    # >1.0 = more stochastic, 1.0 = proportional to visits, ~0 = greedy
    temperature: float = 1.0

    # ========================
    # Progressive Widening
    # ========================
    pw_k: float = 1.5
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
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    train_steps_per_iter: int = 200    # Gradient updates per outer iteration

    # (Only used by our baseline loss function)
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0  # applies to mu/log_std regression

    # ========================
    # Data collection
    # ========================
    collect_episodes_per_iter: int = 10     # Number of real env episodes collected per training iteration
    replay_buffer_capacity: int = 8*batch_size
    gamma_mcts: float = 0.975      # Discount factor for return backup in MCTS
    gamma_mc = 0.9

    # ========================
    # Logging / evaluation
    # ========================
    eval_every: int = 25
    eval_episodes: int = 50   # use 10 fixed seeds for smoother eval curves


cfg = Config()


# === ENVIRONMENTS ===

from rl_competition.competition.environment import create_exploration_seeker

# --- Real environment for rollouts / data collection ---

env_real, env_config = create_exploration_seeker()
obs0, info0 = env_real.reset()

obs_dim = env_real.observation_space.shape[0]
action_dim = env_real.action_space.shape[0]

print("obs_dim:", obs_dim, "action_dim:", action_dim)
print("action_space:", env_real.action_space)

# --- Simulation environment for MCTS step_fn ---
env_sim, _ = create_exploration_seeker()
_ = env_sim.reset() # otherwise order enforcing wrapper is crying
# TODO: Think about using unwrapped env from here on

# --- evaluation environment for evaluation during training ---
env_eval, _ = create_exploration_seeker()

def step_fn(node, action):
    """
    MCTS transition function: set env_sim to `state`, take `action`, return next_state/reward/done/info.
    Returns: next_state, reward, done, info  (matching MCTSPlanner expectations)
    """
    env_set_state(env_sim, node, num_obstacles=env_config.num_obstacles)

    action = np.asarray(action, dtype=env_sim.unwrapped._dtype)
    next_obs, reward, terminated, truncated, info = env_sim.unwrapped.step(action)

    next_obs = np.array(next_obs, copy=True) #break reference to internal env buffer
    
    done = bool(terminated or truncated)
    next_coin_collected = bool(getattr(env_sim, "_coin_collected", False))
    
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
planner = MCTSPlanner(
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

# === BUFFER AND LOGS ===
# Buffer
replay = ReplayBufferHybrid(
    capacity=cfg.replay_buffer_capacity,
    obs_dim=obs_dim,
    action_dim=action_dim,
    K_max=32,
)

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
EVAL_SEEDS = list(range(1000, 1000 + cfg.eval_episodes))  # fixed "validation set"

eval_stats, traces = run_eval_episodes(
            env_eval=env_eval,
            planner=planner,
            seeds=EVAL_SEEDS,
            max_steps=cfg.max_episode_steps,
            goal_reward=env_eval.unwrapped._goal_reward,
            collision_reward=env_eval.unwrapped._collision_reward,
        )

for seed, tr in traces.items():
    tr["training_iteration"] = 0

path = os.path.join(EVAL_DIR, f"traces_ep000000.pkl")
    
with open(path, "wb") as f:
    pickle.dump(traces, f)


num_iters = 400

start_time = time.perf_counter()
print("START")

episodes_collected = 0
episodes_to_collect = 100

episode_success = np.zeros(episodes_to_collect, dtype = bool)
episode_collision = np.zeros(episodes_to_collect, dtype = bool)
episode_max_step = np.zeros(episodes_to_collect)
episode_reward = np.full((episodes_to_collect, cfg.max_episode_steps),np.nan,dtype=float)
episode_z_mc = np.full((episodes_to_collect, cfg.max_episode_steps),np.nan,dtype=float)
episode_z_mcts = np.full((episodes_to_collect, cfg.max_episode_steps),np.nan,dtype=float)

for it in range(num_iters+1):
    now = time.perf_counter()
    print(f"main loop iter: {it}")
    print(f"time: {now-start_time}")
    last_time = now
    
    planner.set_training_iter(it)

    # ---- Replay buffer size increase ----
    if it == 40:
        replay = grow_replay(replay, new_capacity=20 * cfg.batch_size)
    elif it == 100:
        replay = grow_replay(replay, new_capacity=50 * cfg.batch_size)

    # ---- Collect ----
    ep_returns = []
    for _ in range(cfg.collect_episodes_per_iter):
        stats, episode = collect_one_episode_hybrid(
            env_real=env_real,
            planner=planner,
            replay_buffer=replay,
            max_steps=cfg.max_episode_steps,
            gamma=cfg.gamma_mc,
            training=True,
        )
        if episodes_collected < episodes_to_collect:
            episode_success[episodes_collected] = stats["success"]
            episode_collision[episodes_collected] = stats["collision"]
            episode_max_step[episodes_collected] = stats["max_steps_reached"]
            for i, step in enumerate(episode):
                episode_reward[episodes_collected, i] = step["reward"]
                episode_z_mc[episodes_collected, i] = step["z_mc"]
                episode_z_mcts[episodes_collected, i] = step["z_mcts"]
        episodes_collected += 1

        ep_returns.append(stats["return"])
        logs["ep_return"].append(stats["return"])
        logs["ep_length"].append(stats["length"])
        
    now = time.perf_counter()
    print("COLLECT loop time:", now - last_time)
    last_time = now

    # ---- Train (baseline MLE/value regression) ----
    if len(replay) >= cfg.batch_size:
        for _ in range(cfg.train_steps_per_iter):
            batch = replay.sample(cfg.batch_size, device=device, rng=np.random.default_rng(RNG_SEED))

            if True:
                loss_dict = train_step_mle(
                    net=net,
                    optimizer=optimizer,
                    batch=batch,
                    value_target="mcts",
                    w_value=cfg.value_loss_weight,
                    w_policy=cfg.policy_loss_weight,
                    grad_clip_norm=1.0,
                )
            else:
                loss_dict = train_step_mcts_distill(
                    net=net,
                    optimizer=optimizer,
                    batch=batch,
                    value_rms=None,
                    value_target="mc",
                    w_value=cfg.value_loss_weight,
                    w_policy=cfg.policy_loss_weight,
                    grad_clip_norm=1.0,
                )
            
            
            for k, v in loss_dict.items():
                logs[k].append(v)
                
        now = time.perf_counter()
        print("TRAIN loop time:", now - last_time)
        last_time = now

    # ---- Eval (fixed seeds) ----
    if it != 0 and (it % cfg.eval_every) == 0:
        eval_stats, traces = run_eval_episodes(
            env_eval=env_eval,
            planner=planner,
            seeds=EVAL_SEEDS,
            max_steps=cfg.max_episode_steps,
            goal_reward=env_eval.unwrapped._goal_reward,
            collision_reward=env_eval.unwrapped._collision_reward,
        )

        for seed, tr in traces.items():
            tr["training_iteration"] = it

        path = os.path.join(EVAL_DIR, f"traces_ep{it:06d}.pkl")
    
        with open(path, "wb") as f:
            pickle.dump(traces, f)

        
        logs["iter_idx_eval"].append(it)
        logs["eval_return_mean"].append(eval_stats["eval_return_mean"])
        logs["eval_return_std"].append(eval_stats["eval_return_std"])
        logs["eval_length_mean"].append(eval_stats["eval_length_mean"])
        logs["success_rate"].append(eval_stats["success_rate"])
        logs["collision_rate"].append(eval_stats["collision_rate"])
        logs["eval_length_succes"].append(eval_stats["eval_length_mean_successes"])
        logs["eval_length_collision"].append(eval_stats["eval_length_mean_collisions"])
        logs["max_step_rate"].append(eval_stats["max_step_rate"])

        now = time.perf_counter()
        print("EVAL time:", now - last_time)
        last_time = now
        
        print(
            f"[Eval it={it}] "
            f"R={eval_stats['eval_return_mean']:.2f}±{eval_stats['eval_return_std']:.2f} "
            f"succ={eval_stats['success_rate']:.2f} "
            f"coll={eval_stats['collision_rate']:.2f} "
            f"max={eval_stats['max_step_rate']:.2f} "
            f"len_suc={eval_stats['eval_length_mean_successes']:.1f} "
            f"len_col={eval_stats['eval_length_mean_collisions']:.1f} "
        )

    last_loss = logs["loss_total"][-1] if logs["loss_total"] else None
    print(
        f"Iter {it} | replay={len(replay)} | "
        f"train_return_mean={np.mean(ep_returns):.2f} | last_loss={last_loss}"
    )
