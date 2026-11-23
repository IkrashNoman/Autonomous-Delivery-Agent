import os
import sys
import time
import tracemalloc
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from planners.state import (
    make_initial_state,
    is_goal_state,
    expand_search_state,
    reconstruct_path,
    format_result,
)

def bfs(env, start_t=0, max_expansions=60000):
    """From-scratch Breadth-First Search (minimizes number of steps), with an expansion budget."""
    tracemalloc.start()
    t0 = time.perf_counter()

    start_state = make_initial_state(env, start_t)
    queue = deque([(start_state, start_t, 0.0)])
    came_from = {start_state: None}
    expanded = 0

    while queue:
        curr, curr_t, curr_cost = queue.popleft()
        expanded += 1

        if is_goal_state(curr, len(env.stops)):
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            path = reconstruct_path(came_from, curr)
            return format_result(
                "BFS", path, curr_cost, expanded, peak_bytes / 1024.0, (time.perf_counter() - t0) * 1000
            )

        if expanded >= max_expansions:
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return format_result(
                "BFS", [], float("inf"), expanded, peak_bytes / 1024.0,
                (time.perf_counter() - t0) * 1000, "BUDGET_LIMIT"
            )

        for next_state, step_cost in expand_search_state(env, curr, curr_t):
            if next_state not in came_from:
                came_from[next_state] = curr
                queue.append((next_state, curr_t + 1, curr_cost + step_cost))

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return format_result(
        "BFS", [], float("inf"), expanded, peak_bytes / 1024.0, (time.perf_counter() - t0) * 1000, "FAILED"
    )