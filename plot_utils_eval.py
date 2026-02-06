
"""
Plot utilities for the *new* eval data layout written by eval.py:

<eval_dir>/
  summary.json
  slim_traces.json
  trees_seed_<seed>.npz

This module provides:
  - load_eval_dir: read summary + slim traces + index available tree files
  - plot_seeker_trajectory_slim: trajectory plot using slim trace (no need for full obs states)
  - plot_mcts_tree_xy_limited_np: MCTS tree plot from packed npz arrays (no Python tree objects)
  - plot_eval_step: equivalent of old plot_dbg_step (trajectory up to k + tree at k)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np
import matplotlib.pyplot as plt


# We reuse decode_obs from your existing plot_utils.py if present.
# (Fallback: raise a clear error.)
try:
    from plot_utils import decode_obs
except Exception as e:  # pragma: no cover
    decode_obs = None


@dataclass
class EvalRun:
    """In-memory representation of one eval directory."""
    eval_dir: Path
    summary: Dict[str, Any]
    slim_traces: List[Dict[str, Any]]
    trees_by_seed: Dict[int, Path]


def load_eval_dir(eval_dir: str | Path) -> EvalRun:
    eval_dir = Path(eval_dir)
    summary_path = eval_dir / "summary.json"
    slim_path = eval_dir / "slim_traces.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    if not slim_path.exists():
        raise FileNotFoundError(f"Missing {slim_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    slim_traces = json.loads(slim_path.read_text(encoding="utf-8"))

    trees_by_seed: Dict[int, Path] = {}
    for p in sorted(eval_dir.glob("trees_seed_*.npz")):
        m = __import__("re").match(r"trees_seed_(\d+)\.npz$", p.name)
        if m:
            trees_by_seed[int(m.group(1))] = p

    return EvalRun(eval_dir=eval_dir, summary=summary, slim_traces=slim_traces, trees_by_seed=trees_by_seed)


# -------------------------
# Trajectory plot (slim)
# -------------------------

def _add_circle(ax, x: float, y: float, r: float, **kw):
    ax.add_patch(plt.Circle((float(x), float(y)), float(r), **kw))


def plot_seeker_trajectory_slim(
    slim: Dict[str, Any],
    *,
    title: Optional[str] = None,
    L: Optional[float] = 10.0,
    show_line: bool = True,
    show_points: bool = True,
    annotate: bool = False,
    cmap: str = "viridis",
):
    """
    Slim trace plot compatible with eval.py output.

    slim keys (from utils.make_slim_trace):
      agent_pos: (T+1, dim)
      goal_pos:  (dim,)
      obstacles: (N, dim+1) but stored as list of [x,y,(z),r]
      coin_pos:  (dim,) or None
    """
    agent_pos = np.asarray(slim["agent_pos"], dtype=float)
    T = agent_pos.shape[0]
    t = np.arange(T)

    goal = np.asarray(slim["goal_pos"], dtype=float)
    obstacles = np.asarray(slim.get("obstacles", []), dtype=float)
    coin = slim.get("coin_pos", None)
    coin = np.asarray(coin, dtype=float) if coin is not None else None

    fig, ax = plt.subplots(figsize=(6, 6))

    # obstacles: last column radius
    if obstacles.size:
        ox = obstacles[:, 0]
        oy = obstacles[:, 1]
        r = obstacles[:, -1]
        for x, y, rr in zip(ox, oy, r):
            _add_circle(ax, x, y, rr, color="red", alpha=0.3)

    # goal
    ax.scatter(float(goal[0]), float(goal[1]), s=120, marker="*", label="Goal", zorder=6)

    # coin (if any)
    if coin is not None:
        ax.scatter(float(coin[0]), float(coin[1]), s=90, marker="o", label="Coin", zorder=6)

    # path
    if show_line:
        ax.plot(agent_pos[:, 0], agent_pos[:, 1], linewidth=2, alpha=0.7, label="Path", zorder=2)

    if show_points:
        sc = ax.scatter(agent_pos[:, 0], agent_pos[:, 1], c=t, cmap=cmap, s=35, zorder=5)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Step")

    ax.scatter(agent_pos[0, 0], agent_pos[0, 1], color="cyan", s=90, label="Start", zorder=7)
    ax.scatter(agent_pos[-1, 0], agent_pos[-1, 1], color="black", s=90, label="End", zorder=7)

    if annotate:
        for i, (x, y) in enumerate(agent_pos[:, :2]):
            ax.text(float(x), float(y), str(i), fontsize=8)

    if L is None:
        # auto bounds
        pts = [agent_pos[:, :2], goal[:2].reshape(1,2)]
        if coin is not None:
            pts.append(coin[:2].reshape(1,2))
        if obstacles.size:
            pts.append(np.c_[ox - r, oy - r])
            pts.append(np.c_[ox + r, oy + r])
        pts = np.concatenate(pts, axis=0)
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        pad = 0.5
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
    else:
        ax.set_xlim(-float(L), float(L))
        ax.set_ylim(-float(L), float(L))

    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    if title:
        ax.set_title(title)
    plt.show()


# -------------------------
# Packed tree helpers
# -------------------------

def load_packed_trees(npz_path: str | Path) -> Dict[str, np.ndarray]:
    """Load np.savez_compressed output from utils.pack_serialized_trees."""
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def slice_step_tree(packed: Dict[str, np.ndarray], step: int) -> Dict[str, np.ndarray]:
    """
    Return a dict for a single step (rooted MCTS tree) using step offsets.

    Indices in edge_parent/edge_child remain valid within this slice (0..nn-1).
    """
    step = int(step)
    node_off = packed["step_node_offsets"]
    edge_off = packed["step_edge_offsets"]
    if step < 0 or step >= len(node_off) - 1:
        raise IndexError(f"step={step} out of range [0, {len(node_off)-2}]")

    ns, ne = int(node_off[step]), int(node_off[step + 1])
    es, ee = int(edge_off[step]), int(edge_off[step + 1])

    out: Dict[str, np.ndarray] = {}
    for k, v in packed.items():
        if k.startswith("step_"):
            continue
        if k.startswith("node_"):
            out[k] = v[ns:ne]
        elif k.startswith("edge_"):
            out[k] = v[es:ee]
        else:
            # extras (e.g. dim for agentpos mode) are concatenated per-step;
            # we slice by step.
            if v.ndim == 1 and len(v) == (len(node_off) - 1):  # step-aligned
                out[k] = v[step:step+1]
            else:
                out[k] = v
    out["__nn__"] = np.asarray([ne - ns], dtype=np.int32)
    out["__ne__"] = np.asarray([ee - es], dtype=np.int32)
    return out


def _node_xy_from_tree(tree: Dict[str, np.ndarray], i: int, *, num_obstacles: int) -> Tuple[float, float]:
    if "node_agent_pos" in tree and tree["node_agent_pos"].size:
        xy = tree["node_agent_pos"][i][:2]
        return float(xy[0]), float(xy[1])

    if "node_state" in tree and tree["node_state"].size:
        if decode_obs is None:
            raise RuntimeError("plot_utils.decode_obs is not available; cannot decode node_state.")
        agent, *_ = decode_obs(np.asarray(tree["node_state"][i]), num_obstacles=num_obstacles)
        agent = np.asarray(agent, dtype=float)
        return float(agent[0]), float(agent[1])

    raise ValueError("Tree does not contain node_agent_pos or node_state.")


def plot_mcts_tree_xy_limited_np(
    tree: Dict[str, np.ndarray],
    *,
    num_obstacles: int,
    # environment context (one observation from the rollout is enough for obstacles/goal/coin)
    obs0: Optional[np.ndarray] = None,
    slim: Optional[Dict[str, Any]] = None,
    L: Optional[float] = 10.0,
    title: Optional[str] = None,
    ax=None,
    max_depth: int = 6,
    top_k_per_node: int = 5,
    chosen_child_idx: Optional[int] = None,
    edge_color: str = "0.25",
    chosen_edge_color: Optional[str] = None,
    nonterminal_node_color: Optional[str] = None,
    terminal_node_color: str = "tab:red",
    unsafe_node_color: str = "tab:orange",
):
    """
    Replacement for plot_mcts_tree_xy_limited that works on serialized numpy trees.

    Provide either:
      - obs0: a full observation vector from the rollout (lets us draw obstacles/goal/coin via decode_obs), OR
      - slim: a slim trace dict (lets us draw obstacles/goal/coin directly).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    # ----- draw environment background -----
    if slim is not None:
        goal_xy = np.asarray(slim["goal_pos"], dtype=float)[:2]
        obstacles = np.asarray(slim.get("obstacles", []), dtype=float)
        coin = slim.get("coin_pos", None)
        coin_xy = np.asarray(coin, dtype=float)[:2] if coin is not None else None

        if obstacles.size:
            for row in obstacles:
                _add_circle(ax, row[0], row[1], row[-1], alpha=0.25)
    else:
        if obs0 is None:
            raise ValueError("Provide either slim=... or obs0=...")

        if decode_obs is None:
            raise RuntimeError("plot_utils.decode_obs is not available; cannot decode obs0.")
        agent, goal, obstacles, coin, dim = decode_obs(np.asarray(obs0), num_obstacles=num_obstacles)
        goal_xy = np.asarray(goal, dtype=float)[:2]
        obstacles = np.asarray(obstacles, dtype=float)
        coin_xy = np.asarray(coin, dtype=float)[:2] if coin is not None else None

        if obstacles.size:
            for row in obstacles:
                _add_circle(ax, row[0], row[1], row[-1], alpha=0.25)

    ax.scatter(goal_xy[0], goal_xy[1], s=120, marker="*", label="Goal", zorder=6)
    if coin_xy is not None:
        ax.scatter(coin_xy[0], coin_xy[1], s=90, marker="o", label="Coin", zorder=6)

    # ----- build adjacency -----
    parent = tree.get("edge_parent", np.zeros((0,), dtype=np.int32)).astype(np.int32, copy=False)
    child = tree.get("edge_child", np.zeros((0,), dtype=np.int32)).astype(np.int32, copy=False)
    N_sa  = tree.get("edge_N_sa", np.zeros((len(parent),), dtype=np.int32)).astype(np.int32, copy=False)

    nn = int(tree.get("__nn__", [tree.get("node_N", np.zeros((0,), dtype=np.int32)).shape[0]])[0])

    # list of outgoing edge indices per parent node, preserving original order
    out_edges: List[List[int]] = [[] for _ in range(nn)]
    for ei, p in enumerate(parent.tolist()):
        if 0 <= p < nn:
            out_edges[p].append(ei)

    # chosen child: interpret chosen_child_idx as index into root's outgoing edges in original order
    chosen_child_node: Optional[int] = None
    if chosen_child_idx is not None and 0 <= int(chosen_child_idx) < len(out_edges[0]):
        chosen_edge_idx = out_edges[0][int(chosen_child_idx)]
        chosen_child_node = int(child[chosen_edge_idx])

    # traverse limited tree similar to old plotting code
    edges_to_draw: List[Tuple[int, int, bool]] = []
    stack: List[Tuple[int, int]] = [(0, 0)]
    seen = set()

    while stack:
        nid, depth = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        if depth >= int(max_depth):
            continue

        eidxs = out_edges[nid]
        # sort by visit count desc
        eidxs_sorted = sorted(eidxs, key=lambda ei: int(N_sa[ei]) if ei < len(N_sa) else 0, reverse=True)
        eidxs_sorted = eidxs_sorted[: int(top_k_per_node)]

        for ei in eidxs_sorted:
            c = int(child[ei])
            is_chosen = (nid == 0 and chosen_child_node is not None and c == chosen_child_node)
            edges_to_draw.append((nid, c, is_chosen))
            stack.append((c, depth + 1))

    # ----- plot edges -----
    if chosen_edge_color is None:
        chosen_edge_color = edge_color

    drew_chosen_label = False
    for p, c, is_chosen in edges_to_draw:
        x0, y0 = _node_xy_from_tree(tree, p, num_obstacles=num_obstacles)
        x1, y1 = _node_xy_from_tree(tree, c, num_obstacles=num_obstacles)
        if is_chosen:
            ax.plot([x0, x1], [y0, y1], color=chosen_edge_color, linewidth=3, alpha=0.9, zorder=4,
                    label=("Chosen edge" if not drew_chosen_label else None))
            drew_chosen_label = True
        else:
            ax.plot([x0, x1], [y0, y1], color=edge_color, linewidth=1, alpha=0.30, zorder=3)

    # ----- plot nodes -----
    node_is_terminal = tree.get("node_is_terminal", np.zeros((nn,), dtype=np.int8)).astype(bool, copy=False)
    node_is_unsafe   = tree.get("node_is_projected_unsafe", np.zeros((nn,), dtype=np.int8)).astype(bool, copy=False)

    xs_nt, ys_nt, xs_t, ys_t, xs_u, ys_u = [], [], [], [], [], []

    for _, c, _ in edges_to_draw:
        x, y = _node_xy_from_tree(tree, c, num_obstacles=num_obstacles)
        if node_is_unsafe[c]:
            xs_u.append(x); ys_u.append(y)
        elif node_is_terminal[c]:
            xs_t.append(x); ys_t.append(y)
        else:
            xs_nt.append(x); ys_nt.append(y)

    xr, yr = _node_xy_from_tree(tree, 0, num_obstacles=num_obstacles)
    ax.scatter([xr], [yr], s=90, marker="s", label="MCTS root", zorder=8)

    if xs_nt:
        ax.scatter(xs_nt, ys_nt, s=18, alpha=0.45, label="Tree nodes", zorder=4,
                   **({} if nonterminal_node_color is None else {"color": nonterminal_node_color}))
    if xs_t:
        ax.scatter(xs_t, ys_t, s=22, alpha=0.75, color=terminal_node_color, label="Terminal nodes", zorder=5)
    if xs_u:
        ax.scatter(xs_u, ys_u, s=26, alpha=0.85, color=unsafe_node_color, label="Projected unsafe", zorder=6)

    if chosen_child_node is not None:
        xc, yc = _node_xy_from_tree(tree, chosen_child_node, num_obstacles=num_obstacles)
        ax.scatter([xc], [yc], s=140, alpha=0.9, label="Chosen child", zorder=9)

    # bounds
    if L is None:
        pts = [goal_xy.reshape(1,2)]
        if coin_xy is not None:
            pts.append(coin_xy.reshape(1,2))
        for (p, c, _) in edges_to_draw:
            pts.append(np.asarray(_node_xy_from_tree(tree, p, num_obstacles=num_obstacles)).reshape(1,2))
            pts.append(np.asarray(_node_xy_from_tree(tree, c, num_obstacles=num_obstacles)).reshape(1,2))
        if len(pts):
            pts = np.concatenate(pts, axis=0)
            xmin, ymin = pts.min(axis=0)
            xmax, ymax = pts.max(axis=0)
            pad = 0.5
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(ymin - pad, ymax + pad)
    else:
        ax.set_xlim(-float(L), float(L))
        ax.set_ylim(-float(L), float(L))

    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")
    return ax


def plot_eval_step(
    slim: Dict[str, Any],
    tree_step: Dict[str, np.ndarray],
    k: int,
    *,
    num_obstacles: int,
    title: Optional[str] = None,
    L: Optional[float] = 10.0,
    max_depth: int = 6,
    top_k_per_node: int = 5,
    chosen_child_idx: Optional[int] = None,
):
    """
    Equivalent of old plot_dbg_step, but for eval.py artifacts.

    - Trajectory: uses slim.agent_pos up to step k
    - Tree: uses packed np arrays for the MCTS tree at step k
    """
    k = int(k)
    agent_pos = np.asarray(slim["agent_pos"], dtype=float)
    k = max(0, min(k, agent_pos.shape[0] - 2))  # step index (tree per action), agent_pos has T+1

    fig, ax = plt.subplots(figsize=(7, 7))

    # background + full trajectory (0..k)
    slim_k = dict(slim)
    slim_k["agent_pos"] = slim_k["agent_pos"][: k + 1]
    plot_seeker_trajectory_slim(slim_k, title=None, L=L, show_line=True, show_points=True)

    # overlay tree on top on a new axis for consistent legend? simplest: re-plot in same fig
    plt.close(fig)  # close previous to avoid duplicate output

    fig, ax = plt.subplots(figsize=(7, 7))

    # draw trajectory in this ax
    goal = np.asarray(slim["goal_pos"], dtype=float)[:2]
    obstacles = np.asarray(slim.get("obstacles", []), dtype=float)
    coin = slim.get("coin_pos", None)
    coin_xy = np.asarray(coin, dtype=float)[:2] if coin is not None else None

    if obstacles.size:
        for row in obstacles:
            _add_circle(ax, row[0], row[1], row[-1], color="red", alpha=0.2)

    ax.scatter(goal[0], goal[1], s=120, marker="*", label="Goal", zorder=6)
    if coin_xy is not None:
        ax.scatter(coin_xy[0], coin_xy[1], s=90, marker="o", label="Coin", zorder=6)

    traj = agent_pos[: k + 1]
    t = np.arange(traj.shape[0])
    ax.plot(traj[:, 0], traj[:, 1], linewidth=2, alpha=0.5, zorder=2)
    sc = ax.scatter(traj[:, 0], traj[:, 1], c=t, cmap="viridis", s=25, zorder=3)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="t")

    ax.scatter(traj[0, 0], traj[0, 1], s=80, label="Start", zorder=7)
    ax.scatter(traj[-1, 0], traj[-1, 1], s=90, label=f"Agent @ t={k}", zorder=8)

    # chosen idx from slim trace if present
    if chosen_child_idx is None and "chosen_idx" in slim and k < len(slim["chosen_idx"]):
        chosen_child_idx = int(slim["chosen_idx"][k])

    plot_mcts_tree_xy_limited_np(
        tree_step,
        num_obstacles=num_obstacles,
        slim=slim,
        L=L,
        title=title,
        ax=ax,
        max_depth=max_depth,
        top_k_per_node=top_k_per_node,
        chosen_child_idx=chosen_child_idx,
    )

    plt.show()
