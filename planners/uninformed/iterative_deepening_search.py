import os
import sys
import time
import tracemalloc

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from planners.state import make_initial_state, format_result
from planners.uninformed.depth_limited_search import dls_recursive


def ids(env, max_depth=65, start_t=0, memory_limit=config.IDS_MEMORY_LIMIT_STATES, max_expansions=25000):
    """
    Iterative Deepening Search (IDS) with current-path cycle check and memory/expansion cap.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    start_state = make_initial_state(env, start_t)
    total_expanded = 0

    for limit in range(1, max_depth + 1):
        on_path = {start_state}
        stats = {"expanded": 0, "limit_exceeded": False}

        path, cost = dls_recursive(
            env, start_state, start_t, 0.0, limit, on_path, stats, memory_limit, max_expansions - total_expanded
        )
        total_expanded += stats["expanded"]

        if path is not None:
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return format_result(
                "IDS", path, cost, total_expanded, peak_bytes / 1024.0, (time.perf_counter() - t0) * 1000
            )

        if stats["limit_exceeded"] or total_expanded >= max_expansions:
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return format_result(
                "IDS", [], float("inf"), total_expanded, peak_bytes / 1024.0, (time.perf_counter() - t0) * 1000, "BUDGET_LIMIT"
            )

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return format_result(
        "IDS", [], float("inf"), total_expanded, peak_bytes / 1024.0, (time.perf_counter() - t0) * 1000, "MAX_DEPTH_REACHED"
    )