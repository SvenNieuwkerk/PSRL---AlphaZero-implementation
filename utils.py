from __future__ import annotations

from typing import Any, Dict, Tuple, List, Optional, Sequence
import numpy as np
import torch
import math
import torch.nn.functional as F
from pprint import pprint



LOG_2PI = math.log(2.0 * math.pi)


class ReplayBufferHybrid:
    """
    Ring buffer for the hybrid continuous-AlphaZero training sample.

    Stores (fixed shapes via padding):
      - obs:            (obs_dim,)
      - mu_star:        (action_dim,)
      - log_std_star:   (action_dim,)
      - z_mcts:         scalar (e.g., Σ π_i Q_i from root)
      - z_mc:           scalar (Monte Carlo return-to-go)
      - actions_pad:    (K_max, action_dim)   padded root actions
      - probs_pad:      (K_max,)              padded root visit-probs
      - mask:           (K_max,)              1 for valid entries, 0 for padding

    This supports losses:
      - Value regression (choose target): (v - z_mcts)^2 or (v - z_mc)^2 or mixture
      - Discrete imitation:              -sum_i probs_i log p(a_i|s)   (masked)
      - Gaussian regression:             ||mu-mu*||^2 + ||log_std-log_std*||^2
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        K_max: int = 32,
        dtype=np.float32,
    ):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.K_max = int(K_max)
        self.dtype = dtype

        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=self.dtype)
        self.mu_star = np.zeros((self.capacity, self.action_dim), dtype=self.dtype)
        self.log_std_star = np.zeros((self.capacity, self.action_dim), dtype=self.dtype)

        # two value targets
        self.z_mcts = np.zeros((self.capacity,), dtype=self.dtype)
        self.z_mc = np.zeros((self.capacity,), dtype=self.dtype)

        self.actions_pad = np.zeros((self.capacity, self.K_max, self.action_dim), dtype=self.dtype)
        self.probs_pad = np.zeros((self.capacity, self.K_max), dtype=self.dtype)
        self.mask = np.zeros((self.capacity, self.K_max), dtype=self.dtype)

        self.size = 0
        self.ptr = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: np.ndarray,
        mu_star: np.ndarray,
        log_std_star: np.ndarray,
        z_mcts: float,
        z_mc: float,
        actions: np.ndarray,
        probs: np.ndarray,
    ) -> None:
        """
        Add one sample.

        Args:
          obs: (obs_dim,)
          mu_star: (action_dim,)
          log_std_star: (action_dim,)
          z_mcts: scalar (e.g., Σ π Q from root)
          z_mc: scalar (MC return-to-go)

          actions: (K, action_dim) root child actions (preferably sorted by probs desc or N desc)
          probs: (K,) normalized probabilities for those actions

        Notes:
          - We store only up to K_max actions; if K > K_max, we keep the first K_max.
          - We zero-pad the rest and set mask accordingly.
          - If we truncate, we renormalize probs over the stored subset.
        """
        obs = np.asarray(obs, dtype=self.dtype).reshape(self.obs_dim,)
        mu_star = np.asarray(mu_star, dtype=self.dtype).reshape(self.action_dim,)
        log_std_star = np.asarray(log_std_star, dtype=self.dtype).reshape(self.action_dim,)
        z_mcts = float(z_mcts)
        z_mc = float(z_mc)

        actions = np.asarray(actions, dtype=self.dtype)
        probs = np.asarray(probs, dtype=self.dtype)

        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(f"actions must be (K, action_dim={self.action_dim}), got {actions.shape}")
        if probs.ndim != 1 or probs.shape[0] != actions.shape[0]:
            raise ValueError(f"probs must be (K,), got {probs.shape} for actions {actions.shape}")

        K = int(actions.shape[0])
        K_use = min(K, self.K_max)

        # Write fixed fields
        self.obs[self.ptr] = obs
        self.mu_star[self.ptr] = mu_star
        self.log_std_star[self.ptr] = log_std_star
        self.z_mcts[self.ptr] = z_mcts
        self.z_mc[self.ptr] = z_mc

        # Clear padded slots
        self.actions_pad[self.ptr].fill(0.0)
        self.probs_pad[self.ptr].fill(0.0)
        self.mask[self.ptr].fill(0.0)

        # Write top-K_use
        self.actions_pad[self.ptr, :K_use] = actions[:K_use]
        self.probs_pad[self.ptr, :K_use] = probs[:K_use]

        # Normalize probs for the stored subset (important if we truncated)
        probs_sum = float(self.probs_pad[self.ptr, :K_use].sum())
        if probs_sum > 0.0:
            self.probs_pad[self.ptr, :K_use] /= probs_sum
        else:
            if K_use > 0:
                self.probs_pad[self.ptr, :K_use] = 1.0 / float(K_use)

        self.mask[self.ptr, :K_use] = 1.0

        # Advance ring pointer
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch and return torch tensors:

          obs_t:          (B, obs_dim)
          mu_star_t:      (B, action_dim)
          log_std_star_t: (B, action_dim)
          z_mcts_t:       (B,)
          z_mc_t:         (B,)
          actions_t:      (B, K_max, action_dim)
          probs_t:           (B, K_max)
          mask_t:         (B, K_max)
        """
        if self.size == 0:
            raise RuntimeError("ReplayBufferHybrid is empty")

        B = int(batch_size)
        rng = rng if rng is not None else np.random.default_rng()
        idx = rng.integers(0, self.size, size=B)

        obs_t = torch.from_numpy(self.obs[idx]).to(device=device, dtype=torch.float32)
        mu_star_t = torch.from_numpy(self.mu_star[idx]).to(device=device, dtype=torch.float32)
        log_std_star_t = torch.from_numpy(self.log_std_star[idx]).to(device=device, dtype=torch.float32)

        z_mcts_t = torch.from_numpy(self.z_mcts[idx]).to(device=device, dtype=torch.float32)
        z_mc_t = torch.from_numpy(self.z_mc[idx]).to(device=device, dtype=torch.float32)

        actions_t = torch.from_numpy(self.actions_pad[idx]).to(device=device, dtype=torch.float32)
        probs_t = torch.from_numpy(self.probs_pad[idx]).to(device=device, dtype=torch.float32)
        mask_t = torch.from_numpy(self.mask[idx]).to(device=device, dtype=torch.float32)

        return obs_t, mu_star_t, log_std_star_t, z_mcts_t, z_mc_t, actions_t, probs_t, mask_t

def collect_one_episode_hybrid(
    *,
    env_real,
    planner,
    replay_buffer,
    max_steps: int,
    gamma: float,
    training: bool = True,
    store_debug_root_every: int = 0,
    obs = None,
    goal_reward: float = 100.0,
    collision_reward: float = -100.0
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Collect a single episode using:
      - MCTS at each step for action selection
      - hybrid targets for training

    Per step we compute immediately:
      - probs, actions from planner.policy_from_root(root)
      - mu_star, log_std_star, z_mcts from planner.targets_from_root(root)

    We also collect rewards, and after the episode finishes we compute MC returns:
      z_mc_t = r_t + gamma*r_{t+1} + ...

    Then we push (obs, mu_star, log_std_star, z_mcts, z_mc, actions, probs) into replay_buffer
    WITHOUT re-running MCTS.

    Returns:
      stats: episode stats
      debug_roots: optional snapshots of root children (heavy) if store_debug_root_every > 0
    """
    if obs is None: # new collection, else debug collection with fixed obs
        obs, info = env_real.reset()

    else:
        print("DEBUG: COLLECTION RUN WITH OBS:", obs)
        env_set_state(env_real, obs)

    episode = []  # list of dicts, one per step (stores everything except z_mc until end)
    rewards = []

    debug_roots = []

    ep_return = 0.0
    ep_len = 0
    done = False
    terminated_flag = False
    truncated_flag = False

    for t in range(max_steps):
        # 1) Search from current observation
        root = planner.search(obs)
        # roots for debugging
        debug_roots.append(root)

        # 2) Get discrete MCTS policy over sampled actions
        probs, actions = planner.policy_from_root(root)  # probs: (K,), actions: (K, action_dim)
        probs, actions = topk_from_policy(probs, actions, replay_buffer.K_max)

        # 3) Get Gaussian-fit + value target from root
        mu_star, log_std_star, z_mcts = planner.targets_from_root(root)

        # 4) Choose action to execute in the real environment
        action = planner.act(root, training=training)

        # 5) Step real env
        next_obs, reward, terminated, truncated, step_info = env_real.step(action)
        done = bool(terminated or truncated)

        # Store step record (z_mc will be computed later)
        episode.append({
            "obs": obs.copy(),
            "actions": np.asarray(actions, dtype=np.float32),
            "probs": np.asarray(probs, dtype=np.float32),
            "mu_star": np.asarray(mu_star, dtype=np.float32),
            "log_std_star": np.asarray(log_std_star, dtype=np.float32),
            "z_mcts": float(z_mcts),
            "reward": float(reward),
        })
        rewards.append(float(reward))

        ep_return += float(reward)
        ep_len += 1

        # advance
        obs = next_obs
        terminated_flag = bool(terminated)
        truncated_flag = bool(truncated)

        if done:
            break

    reward = episode[-1]["reward"]
    success = False
    collision = False
    max_steps_reached = False
    if np.isclose(reward, goal_reward):
        success = True
    elif np.isclose(reward, collision_reward):
        collision = True
    elif ep_len == max_steps:
        max_steps_reached = True

    # 6) Compute Monte-Carlo returns (return-to-go) backwards
    G = 0.0
    #z_mc_list = [0.0] * len(episode)
    for i in range(len(episode) - 1, -1, -1):
        r = episode[i]["reward"]
        G = r + gamma * G
        episode[i]["z_mc"] = float(G)
        #z_mc_list[i] = float(G)


    # 7) Add to replay buffer with both z targets
    for i, step in enumerate(episode):
        replay_buffer.add(
            step["obs"],
            step["mu_star"],
            step["log_std_star"],
            step["z_mcts"],
            step["z_mc"],
            #z_mc_list[i],
            step["actions"],
            step["probs"],
        )

    stats = {
        "return": float(ep_return),
        "length": int(ep_len),
        "done": bool(done),
        "terminated": bool(terminated_flag),
        "truncated": bool(truncated_flag),
        "success": success,
        "collision": collision,
        "max_steps_reached": max_steps_reached,
    }
    return stats, episode


def diag_gaussian_log_prob(mu: torch.Tensor, log_std: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    Compute log N(actions | mu, diag(std^2)) for diagonal Gaussian.

    Shapes:
      mu:      (B, A)
      log_std: (B, A)
      actions: (B, K, A)  or (B, A) # B=batch, K=top_K, A=action_dim (2 for teleport x,y)

    Returns:
      logp: (B, K) if actions is (B, K, A)
            (B,)   if actions is (B, A)
    """
    std = torch.exp(log_std)  # (B, A)

    if actions.dim() == 2:
        # (B, A)
        diff = (actions - mu) / (std + 1e-8)
        logp = -0.5 * (diff * diff + 2.0 * log_std + LOG_2PI).sum(dim=-1)
        return logp

    if actions.dim() == 3:
        # (B, K, A): broadcast mu/std to (B, 1, A)
        mu_e = mu.unsqueeze(1)
        log_std_e = log_std.unsqueeze(1)
        std_e = std.unsqueeze(1)

        diff = (actions - mu_e) / (std_e + 1e-8)
        logp = -0.5 * (diff * diff + 2.0 * log_std_e + LOG_2PI).sum(dim=-1)  # (B, K)
        return logp

    raise ValueError(f"actions must be (B,A) or (B,K,A), got shape {tuple(actions.shape)}")


def train_step_mle(
    *,
    net,
    optimizer,
    batch,
    w_value: float = 1.0,
    w_policy: float = 1.0,
    grad_clip_norm: float | None = 1.0,
):
    """
    Baseline supervised step:
      - policy head regression to (mu_star, log_std_star)
      - value head regression to z_mcts (Σ π Q from root)

    Ignores imitation targets (actions/probs) and ignores z_mc.
    """
    (obs, mu_star, log_std_star, z_mcts, z_mc, actions, probs, mask) = batch

    mu, log_std, v = net(obs)
    v = v.view(-1)

    # Policy regression (acts like MLE under diagonal Gaussian target fit)
    loss_mu = F.mse_loss(mu, mu_star)
    loss_log_std = F.mse_loss(log_std, log_std_star)
    loss_policy = loss_mu + loss_log_std

    # Value regression to MCTS target
    loss_value = F.mse_loss(v, z_mcts)

    total = loss_policy + (loss_policy.detach() / (loss_value.detach() + 1e-8)) * loss_value
    #total = loss_policy + loss_value

    optimizer.zero_grad(set_to_none=True)
    total.backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), float(grad_clip_norm))
    optimizer.step()

    return {
        "loss_total": float(total.item()),
        "loss_value": float(loss_value.item()),
        "loss_policy": float(loss_policy.item()),
        "loss_mu": float(loss_mu.item()),
        "loss_log_std": float(loss_log_std.item()),
    }


def train_step_mcts_distill(
    *,
    net,
    optimizer,
    batch: Tuple[torch.Tensor, ...],
    value_rms,
    value_target: str = "mc",   # "mc" (your teammate) or "mcts"
    w_value: float = 1.0,
    w_policy: float = 1.0,
    grad_clip_norm: Optional[float] = 1.0,
) -> Dict[str, float]:
    """
    MCTS policy distillation (AlphaZero-style policy improvement loss):
      L_policy = - E_s [ sum_i probs_i(s) * log p_theta(a_i | s) ]
    where probs_i are MCTS visit-probabilities over sampled continuous actions.

    Value loss is supervised either by:
      - MC return-to-go z_mc (teammate's approach), or
      - MCTS root value target z_mcts (Σ π Q).

    batch must be:
      obs, mu_star, log_std_star, z_mcts, z_mc, actions, probs, mask
    """
    (obs, mu_star, log_std_star, z_mcts, z_mc, actions, probs, mask) = batch

    mu, log_std, v = net(obs)
    v = v.view(-1)

    # --- Value target choice ---
    if value_target == "mc":
        z = z_mc
    elif value_target == "mcts":
        z = z_mcts
    else:
        raise ValueError(f"value_target must be 'mc' or 'mcts', got {value_target}")

    value_loss = F.smooth_l1_loss(v, z)
    #value_loss = F.mse_loss(v, z)

    # --- Policy distillation loss ---
    logp = diag_gaussian_log_prob(mu, log_std, actions)  # (B, K)

    probs_masked = probs * mask
    denom = probs_masked.sum(dim=1, keepdim=True).clamp_min(1e-8)
    probs_norm = probs_masked / denom

    imitation_loss = -(probs_norm * logp).sum(dim=1).mean()

    total = w_policy * imitation_loss + w_value * value_loss

    optimizer.zero_grad(set_to_none=True)
    total.backward()

    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), float(grad_clip_norm))

    optimizer.step()

    return {
        "loss_total": float(total.item()),
        "loss_value": float(value_loss.item()),
        "loss_policy_distill": float(imitation_loss.item()),
    }


def train_step_hybrid(
    *,
    net,
    optimizer,
    batch: Tuple[torch.Tensor, ...],
    # value target mixing
    w_value_mcts: float = 1.0,
    w_value_mc: float = 0.0,
    # policy losses
    w_policy_imitation: float = 1.0,
    w_gaussian_reg: float = 0.0,
    # optional extras
    grad_clip_norm: Optional[float] = 1.0,
) -> Dict[str, float]:
    """
    One gradient update step for the hybrid objective.

    batch must be:
      obs, mu_star, log_std_star, z_mcts, z_mc, actions, probs, mask
    Shapes:
      obs:        (B, obs_dim)
      mu_star:    (B, A)
      log_std*:   (B, A)
      z_mcts:     (B,)
      z_mc:       (B,)
      actions:    (B, K, A)
      probs:      (B, K)  (should sum to 1 across valid K)
      mask:       (B, K)  (1 valid, 0 padded)
    """
    (obs, mu_star, log_std_star, z_mcts, z_mc, actions, probs, mask) = batch

    mu, log_std, v = net(obs)          # mu/log_std: (B,A), v: (B,1) or (B,)
    v = v.view(-1)                     # (B,)

    # ----- Value target (choose one or mix) -----
    # MSE per-sample
    lv_mcts = (v - z_mcts).pow(2)
    lv_mc = (v - z_mc).pow(2)

    value_loss = w_value_mcts * lv_mcts.mean() + w_value_mc * lv_mc.mean()

    # ----- Discrete imitation loss over MCTS root actions -----
    # logp(a_i|s) for stored actions
    logp = diag_gaussian_log_prob(mu, log_std, actions)  # (B,K)

    # Mask out padding. probs should already be normalized over valid entries,
    # but masking makes this robust if K is truncated/padded.
    probs_masked = probs * mask  # (B,K)

    # Renormalize just in case (safe)
    denom = probs_masked.sum(dim=1, keepdim=True).clamp_min(1e-8)
    probs_norm = probs_masked / denom

    imitation_loss = -(probs_norm * logp).sum(dim=1).mean()

    # ----- Gaussian regression loss (your mu*, log_std* targets) -----
    gaussian_reg_loss = F.mse_loss(mu, mu_star) + F.mse_loss(log_std, log_std_star)

    # ----- Total loss -----
    total = value_loss + w_policy_imitation * imitation_loss + w_gaussian_reg * gaussian_reg_loss

    optimizer.zero_grad(set_to_none=True)
    total.backward()

    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(grad_clip_norm))

    optimizer.step()

    return {
        "loss_total": float(total.item()),
        "loss_value": float(value_loss.item()),
        "loss_value_mcts": float(lv_mcts.mean().item()),
        "loss_value_mc": float(lv_mc.mean().item()),
        "loss_imitation": float(imitation_loss.item()),
        "loss_gauss_reg": float(gaussian_reg_loss.item()),
    }

def topk_from_policy(probs: np.ndarray, actions: np.ndarray, K_max: int):
    probs = np.asarray(probs)
    actions = np.asarray(actions)

    order = np.argsort(-probs)[: min(K_max, len(probs))]
    probs_top = probs[order].astype(np.float32)
    actions_top = actions[order].astype(np.float32)

    s = float(probs_top.sum())
    if s > 0.0:
        probs_top /= s
    else:
        probs_top[:] = 1.0 / len(probs_top)

    return probs_top, actions_top

from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np

def run_eval_episodes(
    *,
    env_eval,
    planner,
    seeds: Sequence[int],
    max_steps: int,
    goal_reward: float = 100.0,
    collision_reward: float = -100.0,
) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    """
    Eval over multiple seeds.

    Returns:
      stats: aggregate metrics (like your old function, but WITHOUT plot_data)
      traces: dict mapping seed -> per-episode trace dict (dbg-like), containing:
        seed, info, states, roots, chosen_idx, actions, network_outputs, targets_from_root
    """
    returns = np.zeros(len(seeds))
    lengths = np.zeros(len(seeds))
    successes = np.zeros(len(seeds), dtype=bool)
    collisions = np.zeros(len(seeds), dtype=bool)

    traces: Dict[int, Dict[str, Any]] = {}

    for i, seed in enumerate(seeds):
        obs, info = env_eval.reset(seed=int(seed))

        trace = {
            "seed": int(seed),
            "info": info,
            "states": [obs],
            "roots": [],
            "chosen_idx": [],
            "actions": [],
            "network_outputs": [],
            "targets_from_root": [],
        }

        ep_return = 0.0
        ep_len = 0
        terminal_reward: Optional[float] = None

        for _t in range(int(max_steps)):
            root = planner.search(obs)
            action = planner.act(root, training=False)

            mu, log_std, v = planner._policy_value(obs)
            mu_star, log_std_star, v_star = planner.targets_from_root(root)

            if len(root.children) > 0:
                idx = int(np.argmin([np.max(np.abs(ch.action - action)) for ch in root.children]))
            else:
                idx = -1

            trace["roots"].append(root)
            trace["chosen_idx"].append(idx)
            trace["actions"].append(action)
            trace["network_outputs"].append((mu, log_std, v))
            trace["targets_from_root"].append((mu_star, log_std_star, v_star))

            obs, reward, terminated, truncated, info = env_eval.step(action)
            trace["states"].append(obs)

            ep_return += float(reward)
            ep_len += 1

            if terminated or truncated:
                terminal_reward = float(reward)
                break

        returns[i] = ep_return
        lengths[i] = ep_len
        traces[int(seed)] = trace

        if terminal_reward is not None:
            if np.isclose(terminal_reward, goal_reward):
                successes[i] = True
            elif np.isclose(terminal_reward, collision_reward):
                collisions[i] = True

    n = len(seeds)
    max_steps_mask = np.logical_not(np.logical_or(successes, collisions))

    stats = {
        "eval_return_mean": float(np.mean(returns)) if n else 0.0,
        "eval_return_std": float(np.std(returns)) if n else 0.0,
        "eval_length_mean": float(np.mean(lengths)) if n else 0.0,
        "success_rate": np.sum(successes) / float(n) if n else 0.0,
        "collision_rate": np.sum(collisions) / float(n) if n else 0.0,
        "max_step_rate": np.sum(max_steps_mask) / float(n) if n else 0.0,
        "eval_length_mean_successes": float(np.mean(lengths[successes])) if np.sum(successes) > 0 else 0.0,
        "eval_length_mean_collisions": float(np.mean(lengths[collisions])) if np.sum(collisions) > 0 else 0.0,
        "returns": returns,
        "lengths": lengths,
        "successes": successes,
        "collisions": collisions,
    }

    return stats, traces


def run_debug_eval_episode(
    *,
    env_eval,
    planner,
    seed: int,
    max_steps: int,
    obs = None,
)-> Dict[str, Any]:
    
    if obs is None: # new collection, else debug collection with fixed obs
        obs, info = env_eval.reset(seed=int(seed))

    else:
        print("DEBUG: EVAL DEBUG EPISODE RUN WITH OBS:", obs)
        info = None
        env_set_state(env_eval, obs)

    states = [obs]
    roots = []
    chosen_idx = []
    actions_taken = []
    network_outputs = []
    targets_from_root = []

    for t in range(int(max_steps)):
        root = planner.search(obs)
        action = planner.act(root, training=False)
        mu, log_std, v = planner._policy_value(obs)
        mu_star, log_std_star, v_star = planner.targets_from_root(root)
        obs, reward, terminated, truncated, info = env_eval.step(action)



        roots.append(root)
        actions_taken.append(action)
        states.append(obs)
        network_outputs.append((mu, log_std, v))
        targets_from_root.append((mu_star, log_std_star, v_star))
        
        # chosen child index
        idx = int(np.argmin([np.max(np.abs(ch.action - action)) for ch in root.children]))
            # calculates nearest child-action to actual taken action, returns index of that
        chosen_idx.append(idx)
        
        if terminated or truncated:
            break

    return {
        "seed": seed,
        "info": info,
        "states": states,
        "roots": roots,
        "chosen_idx": chosen_idx,
        "actions": actions_taken,
        "network_outputs": network_outputs,
        "targets_from_root": targets_from_root
    }

def env_set_state(env, obs) -> None:
    """
    Overwrite env internal state to match the flat observation vector.

    obs layout:
      [agent_x, agent_y, goal_x, goal_y, (obs_x, obs_y, obs_r)*N]
    """
    obs = np.asarray(obs, dtype=env._dtype)

    # agent and goal
    env._agent_position = obs[0:2].copy()
    env._goal_position = obs[2:4].copy()

    # obstacles
    obstacles = obs[4:].reshape(-1, 3)
    env._obstacle_position = obstacles[:, 0:2].copy()
    env._obstacle_radius = obstacles[:, 2].copy()