from ipywidgets import interact, IntSlider
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def decode_obs(obs, *, num_obstacles: int):
    obs = np.asarray(obs)
    n = int(obs.size)
    N = int(num_obstacles)

    # try dim in {2,3}
    for dim in (2, 3):
        base_no_coin = 2 * dim + N * (dim + 1)
        base_with_coin = 3 * dim + N * (dim + 1)

        if n == base_no_coin:
            has_coin = False
            break
        if n == base_with_coin:
            has_coin = True
            break
    else:
        raise ValueError(
            f"Cannot infer dim/coin from obs length {n} with num_obstacles={N}. "
            f"Expected lengths: "
            f"{2*2 + N*(2+1)} (2D no-coin), {3*2 + N*(2+1)} (2D coin), "
            f"{2*3 + N*(3+1)} (3D no-coin), {3*3 + N*(3+1)} (3D coin)."
        )

    agent = obs[0:dim]
    goal  = obs[dim:2*dim]

    start = 2 * dim
    end = start + N * (dim + 1)
    obstacles = obs[start:end].reshape(N, dim + 1)

    coin = obs[end:end+dim] if has_coin else None
    return agent, goal, obstacles, coin, dim

def get_seeker_radii(env):
    e = env.unwrapped if hasattr(env, "unwrapped") else env
    agent_r = getattr(e, "_agent_radius", None)
    goal_r  = getattr(e, "_goal_radius", None)
    coin_r  = getattr(e, "_coin_radius", None)  # only exists for exploration env
    return agent_r, goal_r, coin_r

def plot_seeker_obs(obs, info, *, num_obstacles: int, env=None, title=None):
    agent, goal, obstacles, coin, dim = decode_obs(obs, num_obstacles=num_obstacles)
    L = info.get("boundary_size", 10) if isinstance(info, dict) else 10

    agent_r = goal_r = coin_r = None
    if env is not None:
        agent_r, goal_r, coin_r = get_seeker_radii(env)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Obstacles: row = (x,y,(z),r) -> radius is last entry
    for row in obstacles:
        ox, oy = float(row[0]), float(row[1])
        r = float(row[-1])
        ax.add_patch(plt.Circle((ox, oy), r, color="red", alpha=0.3))

    # Goal + radius
    gx, gy = float(goal[0]), float(goal[1])
    ax.scatter(gx, gy, color="green", s=80, label="Goal")
    if goal_r is not None:
        ax.add_patch(plt.Circle((gx, gy), float(goal_r), fill=False, color="green", linewidth=2))

    # Coin + radius (only if present in obs AND env has coin radius)
    if coin is not None:
        cx, cy = float(coin[0]), float(coin[1])
        ax.scatter(cx, cy, color="orange", s=80, label="Coin")
        if coin_r is not None:
            ax.add_patch(plt.Circle((cx, cy), float(coin_r), fill=False, color="orange", linewidth=2))

    # Agent + radius
    ax.scatter(float(agent[0]), float(agent[1]), color="blue", s=80, label="Seeker")
    if agent_r is not None:
        ax.add_patch(plt.Circle((float(agent[0]), float(agent[1])), float(agent_r),
                                fill=False, color="blue", linewidth=1, alpha=0.8))

    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    if title:
        ax.set_title(title)
    plt.show()

def plot_seeker_trajectory(
    states,
    *,
    num_obstacles: int,
    env=None,
    title=None,
    cmap="viridis",
    show_line=True,
    show_points=True,
    annotate=False,
    L=10,
):
    if len(states) == 0:
        return

    # decode static elements from first state
    agent0, goal, obstacles, coin, dim = decode_obs(states[0], num_obstacles=num_obstacles)

    agent_r = goal_r = coin_r = None
    if env is not None:
        agent_r, goal_r, coin_r = get_seeker_radii(env)

    # agent positions over time (XY projection)
    agents = np.array([decode_obs(s, num_obstacles=num_obstacles)[0][:2] for s in states], dtype=np.float64)
    T = len(agents)
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(6, 6))

    # obstacles: (x,y,(z),r) -> last entry radius
    for row in obstacles:
        ox, oy = float(row[0]), float(row[1])
        r = float(row[-1])
        ax.add_patch(plt.Circle((ox, oy), r, color="red", alpha=0.3))

    # goal + radius
    gx, gy = float(goal[0]), float(goal[1])
    ax.scatter(gx, gy, color="green", s=120, marker="*", label="Goal", zorder=4)
    if goal_r is not None:
        ax.add_patch(plt.Circle((gx, gy), float(goal_r), fill=False, color="green", linewidth=2, zorder=3))

    # coin + radius (if present)
    if coin is not None:
        cx, cy = float(coin[0]), float(coin[1])
        ax.scatter(cx, cy, color="orange", s=110, marker="o", label="Coin", zorder=4)
        if coin_r is not None:
            ax.add_patch(plt.Circle((cx, cy), float(coin_r), fill=False, color="orange", linewidth=2, zorder=3))

    # trajectory line + points
    if show_line:
        ax.plot(agents[:, 0], agents[:, 1], linewidth=2, alpha=0.7, label="Path", zorder=2)

    if show_points:
        sc = ax.scatter(agents[:, 0], agents[:, 1], c=t, cmap=cmap, s=35, zorder=5)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Step")

    # start/end markers + optional agent radius at start/end
    ax.scatter(agents[0, 0], agents[0, 1], color="cyan", s=90, label="Start", zorder=6)
    ax.scatter(agents[-1, 0], agents[-1, 1], color="black", s=90, label="End", zorder=6)

    if agent_r is not None:
        ax.add_patch(plt.Circle((agents[0, 0], agents[0, 1]), float(agent_r),
                                fill=False, color="cyan", linewidth=1, alpha=0.9, zorder=5))
        ax.add_patch(plt.Circle((agents[-1, 0], agents[-1, 1]), float(agent_r),
                                fill=False, color="black", linewidth=1, alpha=0.9, zorder=5))

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


def plot_mcts_tree_xy_limited(
    root,
    *,
    num_obstacles: int,
    L=10,
    title=None,
    ax=None,
    max_depth=6,
    top_k_per_node=5,
    chosen_child_idx=None,
    goal_radius=None,
    coin_radius=None,
    show_goal_radius=True,
    show_coin_radius=True,
    edge_color="0.25",
    chosen_edge_color=None,
    nonterminal_node_color=None,
    terminal_node_color="tab:red",
    unsafe_node_color="tab:orange",
):
    obs = root.state
    agent, goal, obstacles, coin, dim = decode_obs(
        np.asarray(obs), num_obstacles=int(num_obstacles)
    )
    agent_xy = np.asarray(agent[:2], dtype=float)
    goal_xy = np.asarray(goal[:2], dtype=float)

    obstacles = np.asarray(obstacles, dtype=float)
    obs_xy = obstacles[:, :2] if obstacles.size else obstacles.reshape(0, 2)
    obs_r = obstacles[:, dim] if obstacles.size else np.zeros((0,), dtype=float)

    coin_xy = None
    if coin is not None:
        coin_xy = np.asarray(coin[:2], dtype=float)

    def node_xy(node):
        a, _, _, _, _ = decode_obs(
            np.asarray(node.state), num_obstacles=int(num_obstacles)
        )
        a = np.asarray(a, dtype=float)
        return float(a[0]), float(a[1])

    def is_terminal(node):
        return bool(getattr(node, "is_terminal", False))

    def is_projected_unsafe(node):
        return bool(getattr(node, "is_projected_unsafe", False))

    chosen_child_node = None
    if (
        chosen_child_idx is not None
        and 0 <= int(chosen_child_idx) < len(getattr(root, "children", []))
    ):
        chosen_child_node = root.children[int(chosen_child_idx)].child_node

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
            is_chosen = node is root and chosen_child_node is child
            edges.append((node, child, is_chosen))
            stack.append((child, depth + 1))

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    for (xy, r) in zip(obs_xy, obs_r):
        ax.add_patch(
            plt.Circle((float(xy[0]), float(xy[1])), float(r), alpha=0.25)
        )

    ax.scatter(goal_xy[0], goal_xy[1], s=120, marker="*", label="Goal", zorder=6)
    if show_goal_radius and goal_radius is not None:
        ax.add_patch(
            plt.Circle(
                (goal_xy[0], goal_xy[1]),
                float(goal_radius),
                fill=False,
                linewidth=2,
                alpha=0.8,
                zorder=5,
            )
        )

    if coin_xy is not None:
        ax.scatter(coin_xy[0], coin_xy[1], s=90, marker="o", label="Coin", zorder=6)
        if show_coin_radius and coin_radius is not None:
            ax.add_patch(
                plt.Circle(
                    (coin_xy[0], coin_xy[1]),
                    float(coin_radius),
                    fill=False,
                    linewidth=2,
                    alpha=0.8,
                    zorder=5,
                )
            )

    ax.scatter(agent_xy[0], agent_xy[1], s=90, label="Agent (obs)", zorder=7)

    drew_chosen_label = False
    if chosen_edge_color is None:
        chosen_edge_color = edge_color

    for parent, child, is_chosen in edges:
        x0, y0 = node_xy(parent)
        x1, y1 = node_xy(child)

        if is_chosen:
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=chosen_edge_color,
                linewidth=3,
                alpha=0.9,
                zorder=4,
                label=("Chosen edge" if not drew_chosen_label else None),
            )
            drew_chosen_label = True
        else:
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=edge_color,
                linewidth=1,
                alpha=0.30,
                zorder=3,
            )

    # --- classify nodes ---
    xs_nt, ys_nt = [], []
    xs_t, ys_t = [], []
    xs_u, ys_u = [], []

    for _, child, _ in edges:
        x, y = node_xy(child)

        # unsafe overrides terminal coloring
        if is_projected_unsafe(child):
            xs_u.append(x)
            ys_u.append(y)
        elif is_terminal(child):
            xs_t.append(x)
            ys_t.append(y)
        else:
            xs_nt.append(x)
            ys_nt.append(y)

    xr, yr = node_xy(root)
    ax.scatter([xr], [yr], s=90, marker="s", label="MCTS root", zorder=8)

    if xs_nt:
        ax.scatter(
            xs_nt,
            ys_nt,
            s=18,
            alpha=0.45,
            label="Tree nodes",
            zorder=4,
            **({} if nonterminal_node_color is None else {"color": nonterminal_node_color}),
        )

    if xs_t:
        ax.scatter(
            xs_t,
            ys_t,
            s=22,
            alpha=0.75,
            color=terminal_node_color,
            label="Terminal nodes",
            zorder=5,
        )

    if xs_u:
        ax.scatter(
            xs_u,
            ys_u,
            s=26,
            alpha=0.85,
            color=unsafe_node_color,
            label="Projected unsafe",
            zorder=6,
        )

    if chosen_child_node is not None:
        xc, yc = node_xy(chosen_child_node)
        ax.scatter([xc], [yc], s=140, alpha=0.9, label="Chosen child", zorder=9)

    if L is None:
        pts = [agent_xy, goal_xy]
        if coin_xy is not None:
            pts.append(coin_xy)

        for (xy, r) in zip(obs_xy, obs_r):
            pts.append([xy[0] - r, xy[1] - r])
            pts.append([xy[0] + r, xy[1] + r])

        for _, child, _ in edges:
            pts.append(node_xy(child))

        pts = np.asarray(pts, dtype=float)
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)

        pad = 0.5
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
    else:
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)

    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")
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
    num_obstacles: int,
    L=10,
    max_depth=6,
    top_k_per_node=5,
    title=None,
    ax=None,
    nsig=1.0,
    zoom=True,
    zoom_pad=0.5,      # extra margin around tree
    min_zoom_span=2.0  # minimum width/height so you don't over-zoom
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
    agent, goal, obstacles, coin, dim = decode_obs(obs_k, num_obstacles=num_obstacles)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # --- environment (XY projection) ---
    for row in obstacles:
        ox, oy = float(row[0]), float(row[1])
        r = float(row[-1])
        ax.add_patch(plt.Circle((ox, oy), r, alpha=0.3))
    ax.scatter(float(goal[0]), float(goal[1]), s=120, marker="*", label="Goal", zorder=6)
    if coin is not None:
        ax.scatter(float(coin[0]), float(coin[1]), s=90, marker="o", label="Coin", zorder=6)

    # --- past trajectory (0..k) ---
    agents = np.array([decode_obs(s, num_obstacles=num_obstacles)[0][:2] for s in states[: k + 1]])
    t = np.arange(len(agents))
    ax.plot(agents[:, 0], agents[:, 1], linewidth=2, alpha=0.5, zorder=2)
    sc = ax.scatter(agents[:, 0], agents[:, 1], c=t, cmap="viridis", s=25, zorder=3)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="t")

    ax.scatter(agents[0, 0], agents[0, 1], s=80, label="Start", zorder=7)
    ax.scatter(float(agent[0]), float(agent[1]), s=90, label=f"Agent @ t={k}", zorder=8)

    # --- helpers for tree ---
    def node_xy(node):
        a, _, _, _, _ = decode_obs(np.asarray(node.state), num_obstacles=num_obstacles)
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
    tree_pts = []  # for zoom

    while stack:
        node, depth = stack.pop()
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)

        x, y = node_xy(node)
        tree_pts.append((x, y))

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
        tree_pts.append((x, y))

    xr, yr = node_xy(root)
    ax.scatter([xr], [yr], s=70, marker="s", label="MCTS root", zorder=9)
    if xs:
        ax.scatter(xs, ys, s=18, alpha=0.35, zorder=6)

    if chosen_child_node is not None:
        xc, yc = node_xy(chosen_child_node)
        ax.scatter([xc], [yc], s=120, alpha=0.9, label="Chosen child", zorder=10)
        tree_pts.append((xc, yc))

    # --- Gaussian ellipses (uses first 2 dims of action as XY delta) ---
    mu, log_std, v = net_out[k]
    mu = np.asarray(mu).reshape(-1)
    std = np.exp(np.asarray(log_std).reshape(-1))

    mu_s, log_std_s, v_s = tgt_out[k]
    mu_s = np.asarray(mu_s).reshape(-1)
    std_s = np.exp(np.asarray(log_std_s).reshape(-1))

    axy = np.array([float(agent[0]), float(agent[1])])
    c_net = axy + mu[:2]
    c_tgt = axy + mu_s[:2]

    ax.arrow(axy[0], axy[1], c_net[0]-axy[0], c_net[1]-axy[1],
             length_includes_head=True, head_width=0.15, alpha=0.8,
             color="steelblue", zorder=11)
    ax.arrow(axy[0], axy[1], c_tgt[0]-axy[0], c_tgt[1]-axy[1],
             length_includes_head=True, head_width=0.15, alpha=0.8,
             color="orange", zorder=11)

    w_net = 2.0 * nsig * float(std[0]);  h_net = 2.0 * nsig * float(std[1])
    w_tgt = 2.0 * nsig * float(std_s[0]); h_tgt = 2.0 * nsig * float(std_s[1])

    ax.add_patch(Ellipse((c_net[0], c_net[1]), w_net, h_net,
                         angle=0.0, fill=False, linewidth=2,
                         edgecolor="steelblue", alpha=0.9, label="Net (μ,σ)"))
    ax.add_patch(Ellipse((c_tgt[0], c_tgt[1]), w_tgt, h_tgt,
                         angle=0.0, fill=False, linewidth=2,
                         edgecolor="orange", alpha=0.9, label="Target (μ*,σ*)"))

    # --- extra text: v, v*, reward, MC return-to-go (assumed present later) ---
    r_t = float(dbg.get("rewards", [np.nan]*len(roots))[k])
    g_mc = float(dbg.get("mc_return", [np.nan]*len(roots))[k])

    ax.text(
        0.02, 0.98,
        f"v={float(v):.3f}\nv*={float(v_s):.3f}\nr_t={r_t:.3f}\nG_mc={g_mc:.3f}",
        transform=ax.transAxes, va="top", fontsize=10
    )

    # --- zoom logic (focus on tree) ---
    if zoom and len(tree_pts) > 0:
        pts = np.asarray(tree_pts, dtype=float)
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)

        # ensure minimum span
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        span_x = max(xmax - xmin, min_zoom_span)
        span_y = max(ymax - ymin, min_zoom_span)

        xmin = cx - 0.5 * span_x - zoom_pad
        xmax = cx + 0.5 * span_x + zoom_pad
        ymin = cy - 0.5 * span_y - zoom_pad
        ymax = cy + 0.5 * span_y + zoom_pad

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title if title is not None else f"seed={dbg.get('seed')}  t={k}")
    ax.legend(loc="upper right")
    return ax