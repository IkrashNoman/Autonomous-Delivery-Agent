import os
import sys
import time
import heapq
import tracemalloc

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from planners.state import (
    make_initial_state,
    is_goal_state,
    expand_search_state,
    reconstruct_path,
    custom_mst_heuristic,
    format_result,
)


def weighted_a_star(env, heuristic_fn=custom_mst_heuristic, weight=config.INADMISSIBLE_HEURISTIC_WEIGHT if hasattr(config, "INADMISSIBLE_HEURISTIC_WEIGHT") else 1.8, start_t=0):
    """
    Inadmissible Weighted A*: f(n) = g(n) + w * h(n) with w > 1.
    Faster than optimal A*, but sacrifices optimality guarantee.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    start_state = make_initial_state(env, start_t)
    h0 = heuristic_fn(start_state, env.stops)
    pq = [(weight * h0, 0.0, start_t, start_state)]
    best_g = {start_state: 0.0}
    came_from = {start_state: None}
    expanded = 0

    while pq:
        _, g_cost, curr_t, curr = heapq.heappop(pq)

        if g_cost > best_g.get(curr, float("inf")):
            continue

        expanded += 1

        if is_goal_state(curr, len(env.stops)):
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            path = reconstruct_path(came_from, curr)
            return format_result(
                f"Weighted A* (w={weight})", path, g_cost, expanded, peak_bytes / 1024.0, (time.perf_counter() - t0) * 1000
            )

        for next_state, step_cost in expand_search_state(env, curr, curr_t):
            new_g = g_cost + step_cost
            if new_g < best_g.get(next_state, float("inf")):
                best_g[next_state] = new_g
                came_from[next_state] = curr
                f_val = new_g + weight * heuristic_fn(next_state, env.stops)
                heapq.heappush(pq, (f_val, new_g, curr_t + 1, next_state))

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return format_result(
        f"Weighted A* (w={weight})", [], float("inf"), expanded, peak_bytes / 1024.0, (time.perf_counter() - t0) * 1000, "FAILED"
    )