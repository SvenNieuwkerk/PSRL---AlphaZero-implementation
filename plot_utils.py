from ipywidgets import interact, IntSlider
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def decode_obs(obs):
    obs = np.asarray(obs)
    agent = obs[0:2]
    goal = obs[2:4]
    obstacles_flat = obs[4:]
    num_obstacles = len(obstacles_flat)//3
    obstacles = obstacles_flat.reshape(num_obstacles, 3)  # (x, y, r) per obstacle
    return agent, goal, obstacles

def plot_seeker_obs(obs, info, title=None):
    agent, goal, obstacles = decode_obs(obs)
    L = info.get("boundary_size", 10)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Obstacles
    for (ox, oy, r) in obstacles:
        circ = plt.Circle((ox, oy), r, color="red", alpha=0.3)
        ax.add_patch(circ)

    # Goal
    ax.scatter(goal[0], goal[1], color="green", s=80, label="Goal")
    # (optional: draw goal radius if you know it somewhere else)

    # Agent
    ax.scatter(agent[0], agent[1], color="blue", s=80, label="Seeker")

    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    if title:
        ax.set_title(title)
    plt.show()

def plot_seeker_trajectory(states, title=None, cmap="viridis",
                           show_line=True, show_points=True, annotate=False):
    """
    states: list/iterable of obs vectors (each encodes agent, goal, obstacles)
    info: dict with boundary_size etc.
    """
    if len(states) == 0:
        return

    # Use first state for static elements (goal/obstacles/boundary)
    agent0, goal, obstacles = decode_obs(states[0])
    L = 10

    # Extract agent positions over time
    agents = np.array([decode_obs(s)[0] for s in states])  # shape (T, 2)
    T = len(agents)
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Obstacles
    for (ox, oy, r) in obstacles:
        ax.add_patch(plt.Circle((ox, oy), r, color="red", alpha=0.3))

    # Goal
    ax.scatter(goal[0], goal[1], color="green", s=120, marker="*", label="Goal", zorder=3)

    # Trajectory
    if show_line:
        ax.plot(agents[:, 0], agents[:, 1], linewidth=2, alpha=0.7, label="Path", zorder=2)

    if show_points:
        sc = ax.scatter(
            agents[:, 0], agents[:, 1],
            c=t, cmap=cmap, s=35, zorder=4
        )
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Step")

    # Start / End markers
    ax.scatter(agents[0, 0], agents[0, 1], color="cyan", s=90, label="Start", zorder=5)
    ax.scatter(agents[-1, 0], agents[-1, 1], color="black", s=90, label="End", zorder=5)

    if annotate:
        for i, (x, y) in enumerate(agents):
            ax.text(x, y, str(i), fontsize=8)

    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    if title:
        ax.set_title(title)
    plt.show()

def collect_tree_edges(root, depth_limit=None, max_nodes=5000):
    """
    Traverse the tree and collect (parent_node, child_node, child_edge, depth_child).
    """
    edges = []
    stack = [(root, 0)]
    seen = set()
    node_count = 0

    while stack:
        node, depth = stack.pop()
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)
        node_count += 1
        if node_count > max_nodes:
            break

        if depth_limit is not None and depth >= depth_limit:
            continue

        for ch in getattr(node, "children", []):
            child = ch.child_node
            edges.append((node, child, ch, depth + 1))
            stack.append((child, depth + 1))

    return edges


def plot_mcts_tree_xy(
    root,
    obs,
    *,
    L=10,
    title=None,
    ax=None,
    chosen_child_idx=None,
):
    """Full tree in XY, with optional highlight of the chosen root child."""
    agent, goal, obstacles = decode_obs(obs)

    def node_xy(node):
        a, _, _ = decode_obs(np.asarray(node.state))
        return float(a[0]), float(a[1])

    # identify chosen child node (optional)
    chosen_child_node = None
    if chosen_child_idx is not None and 0 <= int(chosen_child_idx) < len(getattr(root, "children", [])):
        chosen_child_node = root.children[int(chosen_child_idx)].child_node

    # traverse tree
    edges = []  # (parent, child, is_chosen_root_edge)
    stack = [root]
    seen = set()

    while stack:
        node = stack.pop()
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)

        for ch in getattr(node, "children", []):
            child = ch.child_node
            is_chosen = (node is root and chosen_child_node is child)
            edges.append((node, child, is_chosen))
            stack.append(child)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # obstacles
    for (ox, oy, r) in obstacles:
        ax.add_patch(plt.Circle((ox, oy), r, alpha=0.3))

    # goal and agent
    ax.scatter(goal[0], goal[1], s=120, marker="*", label="Goal", zorder=5)
    ax.scatter(agent[0], agent[1], s=90, label="Agent (obs)", zorder=6)

    # edges
    for parent, child, is_chosen in edges:
        x0, y0 = node_xy(parent)
        x1, y1 = node_xy(child)
        if is_chosen:
            ax.plot([x0, x1], [y0, y1], linewidth=3, alpha=0.9, label="Chosen edge", zorder=2)
        else:
            ax.plot([x0, x1], [y0, y1], linewidth=1, alpha=0.35, zorder=1)

    # nodes
    xs, ys = [], []
    for _, child, _ in edges:
        x, y = node_xy(child)
        xs.append(x); ys.append(y)

    xr, yr = node_xy(root)
    ax.scatter([xr], [yr], s=80, marker="s", label="MCTS root", zorder=7)
    if xs:
        ax.scatter(xs, ys, s=20, alpha=0.5, label="Tree nodes", zorder=3)

    # highlight chosen child node
    if chosen_child_node is not None:
        xc, yc = node_xy(chosen_child_node)
        ax.scatter([xc], [yc], s=110, alpha=0.9, label="Chosen child", zorder=8)

    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title)
    return ax


def plot_mcts_tree_xy_limited(
    root,
    obs,
    *,
    L=10,
    title=None,
    ax=None,
    max_depth=6,
    top_k_per_node=5,
    chosen_child_idx=None,
):
    """Tree in XY, limited by depth and top-K children per node, with chosen root child highlight."""
    agent, goal, obstacles = decode_obs(obs)

    def node_xy(node):
        a, _, _ = decode_obs(np.asarray(node.state))
        return float(a[0]), float(a[1])

    chosen_child_node = None
    if chosen_child_idx is not None and 0 <= int(chosen_child_idx) < len(getattr(root, "children", [])):
        chosen_child_node = root.children[int(chosen_child_idx)].child_node

    edges = []  # (parent, child, is_chosen_root_edge)
    stack = [(root, 0)]
    seen = set()

    while stack:
        node, depth = stack.pop()
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)

        if depth >= int(max_depth):
            continue

        children = list(getattr(node, "children", []))
        children.sort(key=lambda ch: ch.N_sa, reverse=True)
        children = children[: int(top_k_per_node)]

        for ch in children:
            child = ch.child_node
            is_chosen = (node is root and chosen_child_node is child)
            edges.append((node, child, is_chosen))
            stack.append((child, depth + 1))

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    for (ox, oy, r) in obstacles:
        ax.add_patch(plt.Circle((ox, oy), r, alpha=0.3))

    ax.scatter(goal[0], goal[1], s=120, marker="*", label="Goal", zorder=5)
    ax.scatter(agent[0], agent[1], s=90, label="Agent (obs)", zorder=6)

    for parent, child, is_chosen in edges:
        x0, y0 = node_xy(parent)
        x1, y1 = node_xy(child)
        if is_chosen:
            ax.plot([x0, x1], [y0, y1], linewidth=3, alpha=0.9, label="Chosen edge", zorder=2)
        else:
            ax.plot([x0, x1], [y0, y1], linewidth=1, alpha=0.35, zorder=1)

    xs, ys = [], []
    for _, child, _ in edges:
        x, y = node_xy(child)
        xs.append(x); ys.append(y)

    xr, yr = node_xy(root)
    ax.scatter([xr], [yr], s=80, marker="s", label="MCTS root", zorder=7)
    if xs:
        ax.scatter(xs, ys, s=20, alpha=0.5, label="Tree nodes", zorder=3)

    if chosen_child_node is not None:
        xc, yc = node_xy(chosen_child_node)
        ax.scatter([xc], [yc], s=110, alpha=0.9, label="Chosen child", zorder=8)

    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title)
    return ax

def inspect_debug_trace_xy(dbg, *, depth_limit=6, max_nodes=4000, show_edge_text=False, highlight_subtree=False):
    roots = dbg["roots"]
    chosen = dbg["chosen_idx"]
    info = dbg["info"]
    record_every = max(1, int(dbg.get("record_every", 1)))

    def view(k):
        env_t = k * record_every
        plot_mcts_tree_xy(
            roots[k],
            info,
            chosen_child_idx=chosen[k] if k < len(chosen) else None,
            depth_limit=depth_limit,
            max_nodes=max_nodes,
            show_edge_text=show_edge_text,
            highlight_subtree=highlight_subtree,
            title=f"Seed={dbg['seed']} | train_ep={dbg.get('train_episode')} | t={env_t}"
        )

    interact(view, k=IntSlider(min=0, max=len(roots)-1, step=1, value=0))

def plot_dbg_step(
    dbg,
    k: int,
    *,
    L=10,
    max_depth=6,
    top_k_per_node=5,
    title=None,
    ax=None,
    nsig=1.0,  # ellipse radius in "number of std devs"
):
    roots = dbg["roots"]
    states = dbg["states"]
    chosen_idx = dbg["chosen_idx"]
    net_out = dbg["network_outputs"]
    tgt_out = dbg["targets_from_root"]

    if k < 0 or k >= len(roots):
        raise IndexError(f"k={k} out of range, have {len(roots)} roots")

    root = roots[k]
    obs_k = states[k]
    agent, goal, obstacles = decode_obs(obs_k)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # --- environment ---
    for (ox, oy, r) in obstacles:
        ax.add_patch(plt.Circle((ox, oy), r, alpha=0.3))
    ax.scatter(goal[0], goal[1], s=120, marker="*", label="Goal", zorder=6)

    # --- past trajectory (0..k) ---
    agents = np.array([decode_obs(s)[0] for s in states[: k + 1]])
    t = np.arange(len(agents))
    ax.plot(agents[:, 0], agents[:, 1], linewidth=2, alpha=0.5, zorder=2)
    sc = ax.scatter(agents[:, 0], agents[:, 1], c=t, cmap="viridis", s=25, zorder=3)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="t")

    ax.scatter(agents[0, 0], agents[0, 1], s=80, label="Start", zorder=7)
    ax.scatter(agent[0], agent[1], s=90, label=f"Agent @ t={k}", zorder=8)

    # --- helpers for tree ---
    def node_xy(node):
        a, _, _ = decode_obs(np.asarray(node.state))
        return float(a[0]), float(a[1])

    # chosen child node
    chosen_child_node = None
    cidx = int(chosen_idx[k]) if k < len(chosen_idx) else None
    if cidx is not None and 0 <= cidx < len(getattr(root, "children", [])):
        chosen_child_node = root.children[cidx].child_node

    # traverse limited tree
    edges = []
    stack = [(root, 0)]
    seen = set()

    while stack:
        node, depth = stack.pop()
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)

        if depth >= int(max_depth):
            continue

        children = list(getattr(node, "children", []))
        children.sort(key=lambda ch: ch.N_sa, reverse=True)
        children = children[: int(top_k_per_node)]

        for ch in children:
            child = ch.child_node
            is_chosen = (node is root and chosen_child_node is child)
            edges.append((node, child, is_chosen))
            stack.append((child, depth + 1))

    for parent, child, is_chosen in edges:
        x0, y0 = node_xy(parent)
        x1, y1 = node_xy(child)
        if is_chosen:
            ax.plot([x0, x1], [y0, y1], linewidth=3, alpha=0.9, zorder=5)
        else:
            ax.plot([x0, x1], [y0, y1], linewidth=1, alpha=0.25, zorder=4)

    xs, ys = [], []
    for _, child, _ in edges:
        x, y = node_xy(child)
        xs.append(x); ys.append(y)

    xr, yr = node_xy(root)
    ax.scatter([xr], [yr], s=70, marker="s", label="MCTS root", zorder=9)
    if xs:
        ax.scatter(xs, ys, s=18, alpha=0.35, zorder=6)

    if chosen_child_node is not None:
        xc, yc = node_xy(chosen_child_node)
        ax.scatter([xc], [yc], s=120, alpha=0.9, label="Chosen child", zorder=10)

    # --- Gaussian ellipses (assumes action is delta-x, delta-y) ---
    mu, log_std, v = net_out[k]
    mu = np.asarray(mu).reshape(-1)
    std = np.exp(np.asarray(log_std).reshape(-1))

    mu_s, log_std_s, v_s = tgt_out[k]
    mu_s = np.asarray(mu_s).reshape(-1)
    std_s = np.exp(np.asarray(log_std_s).reshape(-1))

    c_net = np.array([agent[0], agent[1]]) + mu[:2]
    c_tgt = np.array([agent[0], agent[1]]) + mu_s[:2]

    # arrows (colored)
    ax.arrow(agent[0], agent[1], c_net[0]-agent[0], c_net[1]-agent[1],
             length_includes_head=True, head_width=0.15, alpha=0.8,
             color="steelblue", zorder=11)
    ax.arrow(agent[0], agent[1], c_tgt[0]-agent[0], c_tgt[1]-agent[1],
             length_includes_head=True, head_width=0.15, alpha=0.8,
             color="orange", zorder=11)

    w_net = 2.0 * nsig * float(std[0]);  h_net = 2.0 * nsig * float(std[1])
    w_tgt = 2.0 * nsig * float(std_s[0]); h_tgt = 2.0 * nsig * float(std_s[1])

    # ellipses (colored, not dashed)
    ax.add_patch(Ellipse((c_net[0], c_net[1]), w_net, h_net,
                         angle=0.0, fill=False, linewidth=2,
                         edgecolor="steelblue", alpha=0.9, label="Net (μ,σ)"))
    ax.add_patch(Ellipse((c_tgt[0], c_tgt[1]), w_tgt, h_tgt,
                         angle=0.0, fill=False, linewidth=2,
                         edgecolor="orange", alpha=0.9, label="Target (μ*,σ*)"))

    ax.text(0.02, 0.98, f"v={float(v):.3f}\nv*={float(v_s):.3f}",
            transform=ax.transAxes, va="top", fontsize=10)

    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title if title is not None else f"seed={dbg.get('seed')}  t={k}")

    ax.legend(loc="upper right")
    return ax