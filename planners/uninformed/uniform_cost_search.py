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
    format_result,
)


def ucs(env, start_t=0, max_expansions=60000):
    """From-scratch Uniform-Cost Search with expansion budget (UI safety)."""
    tracemalloc.start()
    t0 = time.perf_counter()

    start_state = make_initial_state(env, start_t)
    pq = [(0.0, start_t, start_state)]
    best_g = {start_state: 0.0}
    came_from = {start_state: None}
    expanded = 0

    while pq:
        g_cost, curr_t, curr = heapq.heappop(pq)

        if g_cost > best_g.get(curr, float("inf")):
            continue

        expanded += 1

        if expanded >= max_expansions:
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return format_result(
                "UCS", [], float("inf"), expanded, peak_bytes / 1024.0,
                (time.perf_counter() - t0) * 1000, "BUDGET_LIMIT"
            )

        if is_goal_state(curr, len(env.stops)):
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            path = reconstruct_path(came_from, curr)
            return format_result(
                "UCS", path, g_cost, expanded, peak_bytes / 1024.0, (time.perf_counter() - t0) * 1000
            )

        for next_state, step_cost in expand_search_state(env, curr, curr_t):
            new_g = g_cost + step_cost
            if new_g < best_g.get(next_state, float("inf")):
                best_g[next_state] = new_g
                came_from[next_state] = curr
                heapq.heappush(pq, (new_g, curr_t + 1, next_state))

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return format_result("UCS", [], float("inf"), expanded, peak_bytes / 1024.0, (time.perf_counter() - t0) * 1000, "FAILED")