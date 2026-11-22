import os
import sys
import time
import heapq
import tracemalloc

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from planners.state import (
    make_initial_state,
    is_goal_state,
    expand_search_state,
    reconstruct_path,
    custom_mst_heuristic,
    format_result,
)


def greedy_best_first(env, heuristic_fn=custom_mst_heuristic, start_t=0):
    """From-scratch Greedy Best-First Search: f(n) = h(n)."""
    tracemalloc.start()
    t0 = time.perf_counter()

    start_state = make_initial_state(env, start_t)
    h0 = heuristic_fn(start_state, env.stops)
    pq = [(h0, 0.0, start_t, start_state)]
    visited = {start_state}
    came_from = {start_state: None}
    expanded = 0

    while pq:
        _, g_cost, curr_t, curr = heapq.heappop(pq)
        expanded += 1

        if is_goal_state(curr, len(env.stops)):
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            path = reconstruct_path(came_from, curr)
            return format_result(
                "Greedy Best-First", path, g_cost, expanded, peak_bytes / 1024.0, (time.perf_counter() - t0) * 1000
            )

        for next_state, step_cost in expand_search_state(env, curr, curr_t):
            if next_state not in visited:
                visited.add(next_state)
                came_from[next_state] = curr
                h_val = heuristic_fn(next_state, env.stops)
                heapq.heappush(pq, (h_val, g_cost + step_cost, curr_t + 1, next_state))

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return format_result(
        "Greedy Best-First", [], float("inf"), expanded, peak_bytes / 1024.0, (time.perf_counter() - t0) * 1000, "FAILED"
    )