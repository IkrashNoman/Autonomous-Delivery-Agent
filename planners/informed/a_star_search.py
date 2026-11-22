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

def a_star(env, heuristic_fn=custom_mst_heuristic, start_t=0, label="A* (Custom MST)", track_memory=True):
    """From-scratch optimal A* Search: f(n) = g(n) + h(n)."""
    if track_memory:
        tracemalloc.start()
    t0 = time.perf_counter()

    start_state = make_initial_state(env, start_t)
    h0 = heuristic_fn(start_state, env.stops)
    pq = [(h0, 0.0, start_t, start_state)]
    best_g = {start_state: 0.0}
    came_from = {start_state: None}
    expanded = 0

    while pq:
        _, g_cost, curr_t, curr = heapq.heappop(pq)

        if g_cost > best_g.get(curr, float("inf")):
            continue

        expanded += 1

        if is_goal_state(curr, len(env.stops)):
            peak_kb = (tracemalloc.get_traced_memory()[1] / 1024.0) if track_memory else 0.0
            if track_memory:
                tracemalloc.stop()
            path = reconstruct_path(came_from, curr)
            return format_result(label, path, g_cost, expanded, peak_kb, (time.perf_counter() - t0) * 1000)

        for next_state, step_cost in expand_search_state(env, curr, curr_t):
            new_g = g_cost + step_cost
            if new_g < best_g.get(next_state, float("inf")):
                best_g[next_state] = new_g
                came_from[next_state] = curr
                f_val = new_g + heuristic_fn(next_state, env.stops)
                heapq.heappush(pq, (f_val, new_g, curr_t + 1, next_state))

    peak_kb = (tracemalloc.get_traced_memory()[1] / 1024.0) if track_memory else 0.0
    if track_memory:
        tracemalloc.stop()
    return format_result(label, [], float("inf"), expanded, peak_kb, (time.perf_counter() - t0) * 1000, "FAILED")