from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Any, Dict, List
import numpy as np
import math
import torch
import torch.nn as nn


# --- Type aliases for clarity ---
State = np.ndarray
Action = np.ndarray

# policy_value_fn returns:
#   policy_params: any object you need to sample actions / compute log-prob (e.g., (mu, log_std) for Gaussian)
#   value: float scalar V(s)
PolicyParams = Any
PolicyValueFn = Callable[[State], Tuple[PolicyParams, float]]

# sample_action returns an action vector from policy params
SampleActionFn = Callable[[PolicyParams], Action]

# prior_weight_fn returns a scalar prior weight for an action under the policy (e.g., exp(log_prob(a|s)))
PriorWeightFn = Callable[[PolicyParams, Action], float]

# step_fn returns next_state, reward, done, and optionally info
StepFn = Callable[[State, Action], Tuple[State, float, bool, Dict[str, Any]]]


@dataclass
class Child:
    """corresponds to action taken (edge in graph) and (child) node reached by that action"""
    action: np.ndarray
    child_node: MCTSNode
    r_sa: float = 0.0 # reward observed for taking action a at state s
    N_sa: int = 0
    Q_sa: float = 0.0
    P_sa: float = 0.0

@dataclass
class MCTSNode:
    state: np.ndarray
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

    def add_child(self, action: np.ndarray, child_state: np.ndarray, prior: float, reward: float) -> "MCTSNode":
        child_node = MCTSNode(state=child_state, parent=self, parent_action=action)
        self.children.append(Child(action=action, child_node=child_node, P_sa=prior, r_sa=float(reward)))
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
        """Run MCTS from root_state and return the root node (tree)."""
        pass

    def act(self, root: "MCTSNode", training: bool = True) -> Action:
        """
        Choose an action from the root after search.
        If training: sample proportional to visit counts (with temperature).
        Else: choose argmax visit count (or temperature->0 behavior).
        """
        pass

    def policy_from_root(self, root: "MCTSNode") -> Tuple[np.ndarray, List[Action]]:
        """
        Return a discrete policy over the root's currently-sampled continuous actions.

        Returns:
          probs: shape (K,) probabilities over root.children actions
          actions: list of K action vectors (np.ndarray)
        """
        pass

    # -------- Core MCTS steps --------

    def run_one_simulation(self, root: "MCTSNode") -> float:
        """
        Run one simulation:
          - selection down the tree by PUCT
          - expansion by progressive widening
          - leaf evaluation by value network / terminal outcome
          - backup along the traversed path

        Returns:
          G: backed-up return from root perspective for this simulation
        """
        pass

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
        pass

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

    def backup(self, path: List[Tuple["MCTSNode", "Child", float]], leaf_value: float) -> None:
        """
        Backup values along the path.
        `path` is a list of (parent_node, edge_taken, reward_received).
        leaf_value is either terminal_value or value_net(leaf_state).

        Applies:
          G = r + gamma * G
          parent.N += 1
          edge.N_sa += 1
          edge.Q_sa <- running average toward G
        """
        pass

    # -------- Progressive widening / priors --------

    def is_fully_expanded(self, node: "MCTSNode") -> bool:
        """Wrapper around node.is_fully_expanded using planner pw_k/pw_alpha."""
        return node.is_fully_expanded(self.pw_k, self.pw_alpha)

    def compute_prior(self, policy_params: PolicyParams, action: Action) -> float:
        """Compute prior weight P(s,a) for a sampled action under the policy."""
        pass

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

    def normalize_root_priors(self, root: "MCTSNode") -> None:
        """Optional: normalize P_sa over root children (helps when using densities as priors)."""
        pass

    def terminal_mapping(self, reward: float, info: Dict[str, Any]) -> float:
        """
        Map environment reward/info into terminal value used for backup.
        Often equals reward, but you may want to clamp or map to +/-1, etc.
        """
        pass

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


