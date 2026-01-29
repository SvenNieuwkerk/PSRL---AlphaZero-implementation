import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from dataclasses import dataclass
from pathlib import Path
import pickle
import random
import numpy as np
import gymnasium as gym
import time

from concurrent.futures import ProcessPoolExecutor, as_completed

from rl_competition.competition.environment import create_exploration_seeker
from acorl.envs.constraints.seeker import SeekerInputSetPolytopeCalculator
from acorl.env_wrapper.adaption_fn import ConditionalAdaptionEnvWrapper
from acorl.acrl_algos.alpha_projection.mapping import alpha_projection_interface_fn

from utils import env_set_state, run_eval_episodes
from network import SeekerAlphaZeroNet
from MCTS_AC import MCTSPlanner_AC

print("RUNNING FILE:", __file__)

# -------------------------
# Reproducibility
# -------------------------
def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


RNG_SEED = 42

# EVAL RUNS
EVAL_SEEDS = list(range(1000, 1000 + 50))  # 50 eval seeds

# Output dir
OUT_DIR = Path("eval_results")
OUT_DIR.mkdir(exist_ok=True)

# Checkpoint root
SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = (SCRIPT_DIR.parent / "eval_checkpoints").resolve()


# -------------------------
# Worker (one ckpt)
# -------------------------
def eval_one_ckpt(run_name: str, ckpt_path_str: str) -> str:
    """
    Evaluate a single checkpoint and save a pickle file.
    Returns output pickle path (string).
    """

    set_global_seeds(RNG_SEED)

    device = torch.device("cpu")
    ckpt_path = Path(ckpt_path_str)

    # --- Environments setup (build inside worker; don't share across processes) ---
    env_eval, env_config = create_exploration_seeker()

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

    obs_dim = env_eval.observation_space.shape[0]
    action_dim = env_eval.action_space.shape[0]

    def sync_conditional_adaption_wrapper(env_wrapped, obs, *, constraint_calculator):
        info = {
            "boundary_size": float(getattr(env_wrapped.unwrapped, "_size", 10.0)),
        }
        info["relevant_input_set"] = constraint_calculator.compute_relevant_input_set(obs, info)
        info["fail_safe_input"] = constraint_calculator.compute_fail_safe_input(obs, info)

        if hasattr(env_wrapped.unwrapped, "_boundary_size"):
            info["boundary_size"] = env_wrapped.unwrapped._boundary_size

        env_wrapped._previous_obs = obs
        env_wrapped._previous_info = info

    def step_fn(node, action):
        # teleport base env
        env_set_state(env_sim_AC, node, num_obstacles=env_config.num_obstacles)

        # sync wrapper cache
        obs = np.asarray(node.state, dtype=env_sim.unwrapped._dtype)
        sync_conditional_adaption_wrapper(
            env_sim_AC,
            obs,
            constraint_calculator=constraint_calculator,
        )

        # step wrapped env
        action = np.asarray(action, dtype=env_sim_AC.unwrapped._dtype)
        next_obs, reward, terminated, truncated, info = env_sim_AC.step(action)
        next_obs = np.array(next_obs, copy=True)

        done = bool(terminated or truncated)
        next_coin_collected = bool(getattr(env_sim_AC.unwrapped, "_coin_collected", False))

        return next_obs, next_coin_collected, reward, done, info

    # --- Load checkpoint ---
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # --- Hidden dims inferred from state_dict (ordered safely) ---
    net_sd = ckpt["net"]
    hidden_dims = tuple(
        net_sd[k].shape[0]
        for k in sorted(net_sd)
        if k.startswith("body") and k.endswith("weight")
    )

    # --- Config: HARD fail if anything missing ---
    cfg = ckpt["cfg"]  # dict saved via cfg.__dict__

    # These lines will raise KeyError if missing
    num_simulations = cfg["num_simulations"]
    cpuct = cfg["cpuct"]
    gamma_mcts = cfg["gamma_mcts"]
    pw_k = cfg["pw_k"]
    pw_alpha = cfg["pw_alpha"]
    max_depth = cfg["max_depth"]
    temperature = cfg["temperature"]

    K_uniform_per_node = cfg["K_uniform_per_node"]
    warmstart_iters = cfg["warmstart_iters"]
    novelty_eps = cfg["novelty_eps"]
    novelty_metric = cfg["novelty_metric"]
    num_candidates = cfg["num_candidates"]
    diversity_lambda = cfg["diversity_lambda"]
    diversity_sigma = cfg["diversity_sigma"]
    policy_beta = cfg["policy_beta"]
    max_resample_attempts = cfg["max_resample_attempts"]

    # --- Recreate and load network ---
    net = SeekerAlphaZeroNet(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_dims,
    ).to(device)

    net.load_state_dict(net_sd)
    net.eval()

    # --- Planner ---
    planner = MCTSPlanner_AC(
        net=net,
        device=str(device),
        step_fn=step_fn,
        num_simulations=num_simulations,
        cpuct=cpuct,
        gamma=gamma_mcts,
        pw_k=pw_k,
        pw_alpha=pw_alpha,
        max_depth=max_depth,
        temperature=temperature,
        rng=np.random.default_rng(RNG_SEED),

        K_uniform_per_node=K_uniform_per_node,
        warmstart_iters=warmstart_iters,
        novelty_eps=novelty_eps,
        novelty_metric=novelty_metric,
        num_candidates=num_candidates,
        diversity_lambda=diversity_lambda,
        diversity_sigma=diversity_sigma,
        policy_beta=policy_beta,
        max_resample_attempts=max_resample_attempts,
    )

    # reset env_eval (consistent starts)
    np.random.seed(RNG_SEED)
    env_eval.reset(seed=RNG_SEED)

    stats, traces = run_eval_episodes(
        env_eval=env_eval,
        planner=planner,
        seeds=EVAL_SEEDS,
        max_steps=999, # Environment cuts off at 150 automatically
        show_progress=False,
    )

    out_path = OUT_DIR / f"{run_name}_{ckpt_path.stem}.pkl"
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

    # tidy up
    env_eval.close()
    env_sim.close()

    return str(out_path)


# -------------------------
# Main
# -------------------------
def build_selected_files(eval_root: Path) -> dict[str, list[Path]]:
    run_folders = sorted([p for p in eval_root.iterdir() if p.is_dir()])

    files_per_run = {
        run.name: sorted([f for f in run.iterdir() if f.is_file()])
        for run in run_folders
    }

    selected_files = {}
    for run, files in files_per_run.items():
        # every second file, but always include last (because of reverse slicing)
        selected_files[run] = files[::-2][::-1] if files else []

    return selected_files


if __name__ == "__main__":
    set_global_seeds(RNG_SEED)

    selected_files = build_selected_files(EVAL_ROOT)

    # Flatten jobs (one job per checkpoint)
    jobs: list[tuple[str, str]] = []
    for run_name, ckpt_paths in selected_files.items():
        for ckpt_path in ckpt_paths:
            jobs.append((run_name, str(ckpt_path)))

    # Choose how many CPU cores to use
    max_workers = 5  # set 4 or 5 as you like

    print(f"Found {len(jobs)} checkpoints total. Using {max_workers} workers.")

    total = len(jobs)
    done = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(eval_one_ckpt, run_name, ckpt_path): (run_name, ckpt_path)
            for run_name, ckpt_path in jobs
        }

        for fut in as_completed(futures):
            run_name, ckpt_path = futures[fut]
            try:
                out_path = fut.result()
                done += 1
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (total - done) / rate if rate > 0 else float("inf")

                print(f"[{done:>3}/{total}] OK  {run_name}  {Path(ckpt_path).name}  ->  {Path(out_path).name}  | ETA ~ {eta/60:.1f} min")
            except Exception as e:
                done += 1
                print(f"[{done:>3}/{total}] FAIL {run_name}  {Path(ckpt_path).name}  | {type(e).__name__}: {e}")
                # If you want to stop immediately on first error:
                # raise
