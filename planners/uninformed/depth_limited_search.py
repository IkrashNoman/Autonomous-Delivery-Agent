import os
import sys
import time
import tracemalloc

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from planners.state import (
    make_initial_state,
    is_goal_state,
    expand_search_state,
    format_result,
)


def dls_recursive(env, curr, curr_t, g_cost, depth_limit, on_path, stats, memory_limit):
    """
    Recursive helper for Depth-Limited Search.
    Uses on_path set to avoid revisiting states on the active recursion stack.
    """
    if len(on_path) > memory_limit:
        stats["memory_exceeded"] = True
        return None, float("inf")

    stats["expanded"] += 1

    if is_goal_state(curr, len(env.stops)):
        return [(curr[0], curr[1])], g_cost

    if depth_limit <= 0:
        return None, float("inf")

    for next_state, step_cost in expand_search_state(env, curr, curr_t):
        if next_state not in on_path:
            on_path.add(next_state)
            sub_path, total_cost = dls_recursive(
                env,
                next_state,
                curr_t + 1,
                g_cost + step_cost,
                depth_limit - 1,
                on_path,
                stats,
                memory_limit,
            )
            on_path.remove(next_state)

            if sub_path is not None:
                return [(curr[0], curr[1])] + sub_path, total_cost
            if stats["memory_exceeded"]:
                return None, float("inf")

    return None, float("inf")


def dls(env, depth_limit=55, start_t=0, memory_limit=config.IDS_MEMORY_LIMIT_STATES):
    """Run Depth-Limited Search up to depth_limit."""
    tracemalloc.start()
    t0 = time.perf_counter()

    start_state = make_initial_state(env, start_t)
    on_path = {start_state}
    stats = {"expanded": 0, "memory_exceeded": False}

    path, cost = dls_recursive(env, start_state, start_t, 0.0, depth_limit, on_path, stats, memory_limit)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    if path is not None:
        return format_result(f"DLS(L={depth_limit})", path, cost, stats["expanded"], peak_bytes / 1024.0, elapsed_ms)

    status = "MEMORY_LIMIT" if stats["memory_exceeded"] else "CUTOFF"
    return format_result(f"DLS(L={depth_limit})", [], float("inf"), stats["expanded"], peak_bytes / 1024.0, elapsed_ms, status)