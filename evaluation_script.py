from dataclasses import dataclass
import numpy as np
import random
import torch
from pathlib import Path
import pickle

import gymnasium as gym

from rl_competition.competition.environment import create_exploration_seeker
from acorl.envs.constraints.seeker import SeekerInputSetPolytopeCalculator
from acorl.env_wrapper.adaption_fn import ConditionalAdaptionEnvWrapper
from acorl.acrl_algos.alpha_projection.mapping import alpha_projection_interface_fn

from utils import (
    env_set_state,
    run_eval_episodes
)
from network import SeekerAlphaZeroNet
from MCTS_AC import MCTSPlanner_AC



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
    num_simulations: int = 400    # Number of MCTS simulations per real environment step
    cpuct: float = 4            # Exploration vs exploitation tradeoff in PUCT; Higher -> more exploration guided by policy prior
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
    batch_size: int = 8
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
    replay_buffer_capacity: int = batch_size
    gamma_mcts: float = 0.85     # Discount factor for return backup in MCTS
    gamma_mc = 0.9

    # ========================
    # Logging / evaluation
    # ========================
    eval_every: int = 25
    eval_episodes: int = 10   # use 10 fixed seeds for smoother eval curves

# === eval output dir ===
out_dir = Path("eval_results")
out_dir.mkdir(exist_ok=True)

# === Paths of checkpoints ===
# go one level up, then into eval_folder
eval_root = Path("..") / "eval_checkpoints"

run_folders = sorted(
    [p for p in eval_root.iterdir() if p.is_dir()]
)

files_per_run = {
    run.name: sorted([f for f in run.iterdir() if f.is_file()])
    for run in run_folders
}

selected_files = {}

for run, files in files_per_run.items():
    selected_files[run] = files[::-2][::-1] if files else []

# === Environments setup ===
env_eval, env_config = create_exploration_seeker()

env_sim = gym.make(env_config.id, **env_config.model_dump(exclude={'id'}))
env_sim = env_sim.unwrapped

constraint_calculator = SeekerInputSetPolytopeCalculator(env_config=env_config)
env_sim_AC = ConditionalAdaptionEnvWrapper(env_sim, 
                                        constraint_calculator.compute_relevant_input_set,
                                        constraint_calculator.compute_fail_safe_input,
                                        constraint_calculator.get_set_representation(),
                                        alpha_projection_interface_fn)


obs_dim = env_eval.observation_space.shape[0]
action_dim = env_eval.action_space.shape[0]

# Step_fn for MCTS
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
    obs = np.asarray(node.state, dtype=env_sim.unwrapped._dtype)
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

# --- set torch device CPU ---
device = torch.device("cpu")

# --- loop over experiments and checkpoints ---
for run_name, ckpt_paths in selected_files.items():
    for ckpt_path in ckpt_paths:
        # ---- load checkpoint ----
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # ---- get dims ----
        hidden_dims = tuple(
            ckpt["net"][k].shape[0]
            for k in sorted(ckpt["net"])
            if k.startswith("body") and k.endswith("weight")
        )

        # ---- get config ----
        cfg_dict = ckpt["cfg"]  # saved via cfg.__dict__
        cfg = Config(**cfg_dict)

        # ---- recreate network ----
        net = SeekerAlphaZeroNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_dims,
            ).to(device)

        # ---- load weights ----
        net.load_state_dict(ckpt["net"])

        # ---- inference mode ----
        net.eval()

        # --- create Planner ---
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

        # EVAL RUNS
        EVAL_SEEDS = list(range(1000, 1000 + 50))  # 100 eval seeds -> 100 validation runs

        # reset env_eval
        np.random.seed(42)
        obs, info = env_eval.reset(seed=42)

        stats, traces = run_eval_episodes(
            env_eval=env_eval,
            planner=planner,
            seeds=EVAL_SEEDS,
            max_steps=999, # Environment cuts off at 150 automatically
        )

        out_path = out_dir / f"{run_name}_{ckpt_path.stem}.pkl"

        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "stats": stats,
                    "traces": traces,
                    "run": run_name,
                    "checkpoint": ckpt_path.name,
                },
                f,
            )