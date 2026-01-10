from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Any, Dict, List
import numpy as np
from numpy.typing import NDArray
import math
import torch
import torch.nn as nn


# --- Type aliases for clarity ---
State = NDArray[np.generic]
Action = NDArray[np.generic]

# policy_value_fn returns:
#   policy_params: any object you need to sample actions / compute log-prob (e.g., (mu, log_std) for Gaussian)
#   value: float scalar V(s)
PolicyParams = Any

# step_fn returns next_state, reward, done, and optionally info
StepFn = Callable[[State, Action], Tuple[State, float, bool, Dict[str, Any]]]


@dataclass
class Child:
    """corresponds to action taken (edge in graph) and (child) node reached by that action"""
    action: Action
    child_node: MCTSNode
    r_sa: float = 0.0 # reward observed for taking action a at state s
    N_sa: int = 0
    Q_sa: float = 0.0
    P_sa_raw: float = 0.0 # raw probability densities for action a in state s given by policy network
    P_sa: float = 0.0 # normalized across children (sum(P_sa) over all a = 1)

@dataclass
class MCTSNode:
    state: State
    parent: Optional[MCTSNode] = None
    parent_action: Optional[np.ndarray] = None

    N: int = 0
    children: List[Child] = field(default_factory=list)

    mu: Optional[np.ndarray] = None
    log_std: Optional[np.ndarray] = None
    v: Optional[float] = None

    is_terminal: bool = False
    terminal_value: Optional[float] = None

    def is_fully_expanded(self, k: float, alpha: float) -> bool:
        """progressive widening check for if the node is fully expanded"""
        if self.N == 0:
            return False
        
        K_max = k * (self.N ** alpha)
        return len(self.children) >= K_max

    def add_child(
        self,
        action: Action,
        child_state: State,
        prior_raw: float,
        reward: float,
    ) -> "MCTSNode":
        child_node = MCTSNode(state=child_state, parent=self, parent_action=action)

        prior_raw = float(prior_raw)
        reward = float(reward)

        # Temporarily set P_sa = P_raw; planner will normalize across siblings after insertion.
        self.children.append(
            Child(
                action=action,
                child_node=child_node,
                r_sa=reward,
                P_sa_raw=prior_raw,
                P_sa=prior_raw,
            )
        )
        return child_node

class MCTSPlanner:
    """
    AlphaZero-style MCTS planner with progressive widening for continuous actions.

    Assumptions:
      - Tree nodes represent states.
      - Child edges represent sampled continuous actions from that state, each with stats (N_sa, Q_sa, P_sa).
      - Leaf evaluation uses value bootstrap unless terminal.
    """

    def __init__(
        self,
        *,
        net: nn.Module,
        device: str = "cpu",
        step_fn: StepFn,
        num_simulations: int = 200,
        cpuct: float = 1.5,
        gamma: float = 0.99,
        pw_k: float = 2.0,
        pw_alpha: float = 0.5,
        max_depth: int = 64,
        # Optional AlphaZero-like root noise
        #add_root_dirichlet_noise: bool = False,
        #dirichlet_alpha: float = 0.3,
        #dirichlet_epsilon: float = 0.25,
        # Action selection from root after search
        temperature: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.net = net
        self.device = device
        self.step_fn = step_fn

        self.num_simulations = num_simulations
        self.cpuct = cpuct
        self.gamma = gamma

        # progressive widening (pw) params
        self.pw_k = pw_k
        self.pw_alpha = pw_alpha
        self.max_depth = max_depth

        #self.add_root_dirichlet_noise = add_root_dirichlet_noise
        #self.dirichlet_alpha = dirichlet_alpha
        #self.dirichlet_epsilon = dirichlet_epsilon

        self.temperature = temperature
        self.rng = rng if rng is not None else np.random.default_rng()

    # -------- Public API --------

    def search(self, root_state: State) -> "MCTSNode":
        """
        Run MCTS from root_state and return the root node (containing edge stats as children).

        High-level:
        1) Create a root node for this observation/state
        2) (Optional but recommended) evaluate root once to cache mu/log_std/v
        3) Run `num_simulations` times: one_simulation(root)
        4) Return root so caller can:
            - pick an action (act / policy_from_root)
            - extract training targets (N_sa, Q_sa, actions)
        """
        root = MCTSNode(state=root_state)

        # Cache policy/value at root immediately (saves re-checks in first sim)
        self.evaluate_node(root)

        # Run MCTS simulations
        for _ in range(self.num_simulations):
            self.run_one_simulation(root)

        return root

    def act(self, root: "MCTSNode", training: bool = True) -> Action:
        """
        Choose a continuous action to execute in the real environment
        based on the MCTS-improved policy at the root.

        If training: sample proportional to visit counts (with temperature).
        Else: choose argmax visit count (or temperature->0 behavior).

        Args:
        root: root node returned by `search`
        training: if True, sample according to visit-count policy;
                  if False, act greedily (argmax)

        Returns:
        action: np.ndarray (continuous action)
        """
        probs, actions = self.policy_from_root(root)

        if training:
            # Sample action index according to the improved policy
            idx = self.rng.choice(len(actions), p=probs)
        else:
            # Greedy action (most visited)
            idx = int(np.argmax(probs))

        return actions[idx]

    def policy_from_root(self, root: "MCTSNode") -> Tuple[np.ndarray, List[Action]]:
        """
        Return a discrete policy over the root's currently-sampled continuous actions.

        Returns:
        probs: shape (K,) probabilities over root.children actions
        actions: list of K action vectors (np.ndarray)
        """
        if not root.children:
            raise RuntimeError("policy_from_root called on root with no children")

        actions: List[Action] = [child.action for child in root.children]
        visits = np.array([child.N_sa for child in root.children], dtype=np.float64)

        # Handle the degenerate case where no edge was ever visited
        if visits.sum() <= 0.0:
            probs = np.ones_like(visits) / len(visits)
            return probs, actions

        # Apply temperature (AlphaZero-style)
        if self.temperature <= 0.0:
            # temperature -> 0  => argmax
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            # pi_i ∝ N_i^(1 / tau)
            visits = visits ** (1.0 / self.temperature)
            probs = visits / visits.sum()

        return probs, actions
    
    def targets_from_root(
        self,
        root: "MCTSNode",
        *,
        eps: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Build supervised training targets from an MCTS root node.

        Returns:
            mu_star: (action_dim,) weighted mean of root actions under π (from visit counts)
            log_std_star: (action_dim,) weighted log-std (diagonal Gaussian MLE) under π
            z: float, value target for V(s_root) computed as Σ π_i Q(s_root,a_i)

        Notes:
        - π is derived from visit counts (same as policy_from_root).
        - We fit a diagonal Gaussian to the discrete action set sampled by progressive widening.
        - z is the expected return under the MCTS-improved policy (visit-weighted Q).
        """
        probs, actions = self.policy_from_root(root)  # probs shape (K,)

        # Stack actions -> (K, action_dim)
        A = np.stack(actions, axis=0).astype(np.float64)  # stable numerics
        w = probs.astype(np.float64)
        w = w / (w.sum() + eps)

        # Weighted mean (mu*)
        mu_star = (w[:, None] * A).sum(axis=0)

        # Weighted diagonal variance (sigma^2*)
        diff = A - mu_star[None, :]
        var_star = (w[:, None] * (diff ** 2)).sum(axis=0)

        # Avoid collapse
        std_star = np.sqrt(np.maximum(var_star, eps))
        log_std_star = np.log(std_star)

        # Value target z = Σ π_i Q_i
        Q = np.array([child.Q_sa for child in root.children], dtype=np.float64)
        z = float((w * Q).sum())

        return mu_star.astype(np.float32), log_std_star.astype(np.float32), z



    # -------- Core MCTS steps --------
    def run_one_simulation(self, root: "MCTSNode") -> float:
        """
        Run a single MCTS simulation starting from `root`:
          - selection down the tree by PUCT
          - expansion by progressive widening
          - leaf evaluation by value network / terminal outcome
          - backup along the traversed path

        Returns:
            G_root (float): backed-up return from the root perspective
                            (mainly for debugging; usually ignored by caller)
        """
        node = root
        path: list[tuple["MCTSNode", "Child"]] = []
        depth = 0

        while True:
            # --- Case 1: terminal node ---
            if node.is_terminal:
                leaf_value = float(node.terminal_value)
                break

            # --- Case 2: first visit to this node (no policy/value cached yet) ---
            if node.v is None:
                leaf_value = float(self.evaluate_node(node))
                break

            # --- Case 3: depth cutoff (truncate search) ---
            if depth >= self.max_depth:
                leaf_value = float(node.v)
                break

            # --- Case 4: expansion allowed (progressive widening) ---
            if not self.is_fully_expanded(node):
                edge, reward, done = self.expand_child(node)
                path.append((node, edge))

                if done:
                    leaf_value = float(edge.child_node.terminal_value)
                else:
                    # Evaluate the newly created leaf
                    leaf_value = float(self.evaluate_node(edge.child_node))

                break

            # --- Case 5: selection among existing children (PUCT) ---
            edge = self.select_child(node)
            path.append((node, edge))

            node = edge.child_node
            depth += 1

        # --- Backup ---
        self.backup(path, leaf_value)

        return leaf_value


    def select_child(self, node: "MCTSNode") -> "Child":
        """
        Select an existing child edge from node by PUCT argmax.
        Assumes:
          - node has at least one child
          - node is considered fully expanded (i.e. no new actions allowed right now)

        Returns:
            The Child edge with the highest PUCT score.
        """
        if not node.children:
            raise RuntimeError("select_child called on node with no children")

        best_edge = node.children[0]
        best_score = self.puct_score(node, best_edge)

        for edge in node.children[1:]:
            score = self.puct_score(node, edge)
            if score > best_score:
                best_score = score
                best_edge = edge

        return best_edge

    def expand_child(self, node: "MCTSNode") -> Tuple["Child", float, bool]:
        """
        Expand exactly one new child from node (sample a new continuous action).

        Returns:
        edge: the newly-added Child edge
        reward: immediate reward from stepping (s,a)->s'
        done: whether next state is terminal
        """
        # Ensure policy/value are cached on this node
        self.evaluate_node(node)
        if node.mu is None or node.log_std is None:
            raise AssertionError("Node policy params not cached.")

        # 1) Sample a new action from the node's Gaussian policy
        action = self._sample_action(node.mu, node.log_std)  # np.ndarray, shape (action_dim,)

        # 2) Compute raw prior weight from policy density at that action
        prior_raw = self._prior_weight(node.mu, node.log_std, action)

        # 3) Transition
        next_state, reward, done, info = self.step_fn(node.state, action)
        reward = float(reward)
        done = bool(done)

        # 4) Add child edge/node (stores reward on edge + stores P_raw)
        child_node = node.add_child(
            action=action,
            child_state=next_state,
            prior_raw=prior_raw,
            reward=reward,
        )
        edge = node.children[-1]

        # 5) Mark terminal if done
        if done:
            child_node.is_terminal = True
            child_node.terminal_value = float(self.terminal_mapping(reward, info))

        # 6) Normalize priors at THIS node (recommended for every node, not only root)
        self.normalize_node_priors(node)

        return edge, reward, done


    def evaluate_node(self, node: "MCTSNode") -> float:
        """
        Evaluate a node using the policy/value network and cache the results.

        This function should be called exactly once per node (unless you
        deliberately want to re-evaluate, which is uncommon in AlphaZero-style MCTS).

        Returns:
            v (float): value estimate V(s) for this node
        """
        # If already evaluated, just return cached value
        if node.v is not None:
            return node.v

        # Query policy + value network
        mu, log_std, v = self._policy_value(node.state)

        # Cache results on the node
        node.mu = mu
        node.log_std = log_std
        node.v = float(v)

        return node.v

    def backup(self, path: List[Tuple["MCTSNode", "Child"]], leaf_value: float) -> None:
        """
        Backup along a traversed path using rewards stored on edges (edge.r_sa).

        Args:
            path: list of (parent_node, edge_taken) from root down to the leaf edge.
                The *last* tuple is the edge that led into the leaf node/state.
            leaf_value: value at the leaf:
                - if terminal leaf: terminal_value
                - else: value network estimate V(s_leaf)

        Update rule (from leaf back to root):
            G <- r_sa + gamma * G
            parent.N += 1
            edge.N_sa += 1
            edge.Q_sa <- running average toward G
        """
        g = float(leaf_value)

        # Walk backwards: leaf -> ... -> root
        for parent, edge in reversed(path):
            # Incorporate immediate reward on this edge
            g = float(edge.r_sa) + self.gamma * g

            # Update counts
            parent.N += 1
            edge.N_sa += 1

            # Running-average update for Q(s,a)
            edge.Q_sa += (g - edge.Q_sa) / edge.N_sa

    # -------- Progressive widening / priors --------

    def is_fully_expanded(self, node: "MCTSNode") -> bool:
        """Wrapper around node.is_fully_expanded using planner pw_k/pw_alpha."""
        return node.is_fully_expanded(self.pw_k, self.pw_alpha)

    def apply_root_noise(self, root: "MCTSNode") -> None:
        """
        Optionally apply Dirichlet noise to root child priors (AlphaZero trick),
        typically after root has some children (or as children are added).
        """
        pass

    # -------- Utilities --------

    def puct_score(self, node: "MCTSNode", edge: "Child") -> float:
        """
        Compute the PUCT score for edge (s,a) emanating from `node` (state s).
        PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Notes:
        - Q(s,a) is exploitation (learned from backups).
        - The second term is exploration guided by the policy prior P(s,a).
        - As N(s,a) grows, the exploration bonus shrinks for that edge.
        """
        # Exploitation term
        q = float(edge.Q_sa)

        # Exploration term
        n_s = float(node.N)
        n_sa = float(edge.N_sa)
        p = float(edge.P_sa)

        # If node.N is 0, sqrt(0)=0 and exploration is 0. That's fine.
        u = self.cpuct * p * (math.sqrt(n_s) / (1.0 + n_sa))

        return q + u

    def normalize_node_priors(self, node: "MCTSNode") -> None:
        """
        Normalize priors for a node's children using their stored P_raw values.

        After this, sum(child.P_sa) == 1 over node.children (unless fallback uniform).
        """
        if not node.children:
            return

        raw = np.array([float(ch.P_sa_raw) for ch in node.children], dtype=np.float64)

        # Guard against negatives / NaNs / inf
        raw = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
        s = float(raw.sum())

        if s <= 0.0:
            # Fallback: uniform priors
            uniform = 1.0 / len(node.children)
            for ch in node.children:
                ch.P_sa = uniform
            return

        for ch, r in zip(node.children, raw):
            ch.P_sa = float(r / s)

    def terminal_mapping(self, reward: float, info: Dict[str, Any]) -> float:
        """
        Map environment reward/info into terminal value used for backup.
        Often equals reward, but you may want to clamp or map to +/-1, etc.
        """
        return float(reward)

    def _policy_value(self, obs_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Returns:
        mu_np: (action_dim,)
        log_std_np: (action_dim,)
        v: float
        """
        obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).to(self.device)  # [1, obs_dim]
        with torch.no_grad():
            mu_t, log_std_t, v_t = self.net(obs_t)
        return (
            mu_t.squeeze(0).cpu().numpy(),
            log_std_t.squeeze(0).cpu().numpy(),
            float(v_t.item()),
        )
    
    def _sample_action(self, mu: np.ndarray, log_std: np.ndarray) -> np.ndarray:
        std = np.exp(log_std)
        a = mu + std * self.rng.standard_normal(size=mu.shape)
        return np.clip(a, -1.0, 1.0) # if your env expects actions clipped to [-1, 1]
    
    def _log_prob_diag_gaussian(self, mu: np.ndarray, log_std: np.ndarray, a: np.ndarray) -> float:
        # log N(a; mu, std) summed over dims
        std = np.exp(log_std)
        var = std * std
        # constant term: -0.5 * log(2π) per dim
        log2pi = np.log(2.0 * np.pi)
        return float(np.sum(-0.5 * (((a - mu) ** 2) / var + 2.0 * log_std + log2pi)))

    def _prior_weight(self, mu: np.ndarray, log_std: np.ndarray, a: np.ndarray) -> float:
        return float(np.exp(self._log_prob_diag_gaussian(mu, log_std, a)))
