import os
import sys
import tracemalloc
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def make_initial_state(env, start_t=0):
    """Build initial search state (r, c, delivered_mask, t_mod)."""
    mask = 0
    for idx, stop in enumerate(env.stops):
        if stop in env.delivered or stop in env.stolen:
            mask |= (1 << idx)
    r, c = env.agent_pos
    return (r, c, mask, start_t % config.RUSH_HOUR_CYCLE)


def is_goal_state(state, num_stops):
    """Return True when all parcels have been collected/delivered."""
    goal_mask = (1 << num_stops) - 1
    return state[2] == goal_mask


def get_remaining_stops(state, stops):
    """Return list of stop coordinates not yet set in delivered_mask."""
    mask = state[2]
    return [stops[i] for i in range(len(stops)) if not (mask & (1 << i))]


def expand_search_state(env, state, actual_t):
    """
    Generate valid successor states (next_state, step_cost) at simulation step actual_t.
    """
    r, c, mask, _ = state
    next_t = actual_t + 1
    next_t_mod = next_t % config.RUSH_HOUR_CYCLE
    successors = []

    for next_pos, step_cost in env.get_valid_neighbors((r, c), actual_t):
        new_mask = mask
        for idx, stop in enumerate(env.stops):
            if next_pos == stop:
                new_mask |= (1 << idx)
        next_state = (next_pos[0], next_pos[1], new_mask, next_t_mod)
        successors.append((next_state, step_cost))

    return successors


def reconstruct_path(came_from, goal_state):
    """Reconstruct list of (r, c) grid coordinates from start to goal."""
    coords = []
    curr = goal_state
    while curr is not None:
        coords.append((curr[0], curr[1]))
        curr = came_from[curr]
    coords.reverse()
    return coords


# ------------------------------------------------------------------------------
# HEURISTICS (Custom Admissible MST + 2 Baselines)
# ------------------------------------------------------------------------------

def custom_mst_heuristic(state, stops):
    """
    Custom Admissible Heuristic:
    Minimum Spanning Tree (Prim's) over {agent_pos} U {remaining_stops} using
    Manhattan distance, scaled by MIN_TERRAIN_COST (3).
    """
    rem = get_remaining_stops(state, stops)
    if not rem:
        return 0.0

    nodes = [(state[0], state[1])] + rem
    n = len(nodes)
    in_mst = [False] * n
    min_edge = [float("inf")] * n
    min_edge[0] = 0.0
    mst_length = 0

    for _ in range(n):
        u = -1
        best = float("inf")
        for i in range(n):
            if not in_mst[i] and min_edge[i] < best:
                best = min_edge[i]
                u = i
        in_mst[u] = True
        mst_length += best

        ur, uc = nodes[u]
        for v in range(n):
            if not in_mst[v]:
                vr, vc = nodes[v]
                dist = abs(ur - vr) + abs(uc - vc)
                if dist < min_edge[v]:
                    min_edge[v] = dist

    return float(mst_length * config.MIN_TERRAIN_COST)


def baseline_nearest_manhattan(state, stops):
    """Baseline 1: Manhattan distance to the nearest undelivered stop * min cost."""
    rem = get_remaining_stops(state, stops)
    if not rem:
        return 0.0
    r, c = state[0], state[1]
    min_d = min(abs(r - sr) + abs(c - sc) for sr, sc in rem)
    return float(min_d * config.MIN_TERRAIN_COST)


def baseline_farthest_manhattan(state, stops):
    """Baseline 2: Manhattan distance to the farthest undelivered stop * min cost."""
    rem = get_remaining_stops(state, stops)
    if not rem:
        return 0.0
    r, c = state[0], state[1]
    max_d = max(abs(r - sr) + abs(c - sc) for sr, sc in rem)
    return float(max_d * config.MIN_TERRAIN_COST)


def verify_heuristic_consistency(env, samples=500):
    """
    Empirically verify consistency: h(s) <= step_cost(s, s') + h(s')
    across valid states on the grid.
    """
    free_cells = [(r, c) for r in range(env.rows) for c in range(env.cols) if env.grid[r, c] >= 0]
    violations = 0
    checked = 0

    for i in range(min(samples, len(free_cells))):
        r, c = free_cells[i]
        for mask in range(1 << len(env.stops)):
            state = (r, c, mask, i % config.RUSH_HOUR_CYCLE)
            h_s = custom_mst_heuristic(state, env.stops)
            for next_state, cost in expand_search_state(env, state, i):
                h_next = custom_mst_heuristic(next_state, env.stops)
                checked += 1
                if h_s > cost + h_next + 1e-9:
                    violations += 1

    return {"transitions_checked": checked, "violations": violations, "consistent": violations == 0}


def format_result(algorithm, path, cost, expanded, peak_kb, elapsed_ms, status="SUCCESS"):
    """Standardized dictionary returned by every planner."""
    return {
        "algorithm": algorithm,
        "status": status,
        "path": path,
        "steps": max(0, len(path) - 1),
        "cost": round(cost, 2) if cost < float("inf") else float("inf"),
        "nodes_expanded": expanded,
        "peak_memory_kb": round(peak_kb, 2),
        "runtime_ms": round(elapsed_ms, 3),
    }