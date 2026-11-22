import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from world.road_closures import DIRECTIONS


def is_rush_hour(t):
    """Return True if simulation step t is inside the rush-hour window."""
    if "rush_hour_time_varying_costs" not in config.TWISTS:
        return False
    phase = t % config.RUSH_HOUR_CYCLE
    start, end = config.RUSH_HOUR_WINDOW
    return start <= phase <= end


def build_cost_map_vectorized(grid, t=0, closure_mgr=None):
    """
    Build the 2D (28x28) cell cost matrix at step t using vectorized NumPy operations.
    Masks fully 'closed' cells from closure_mgr to np.inf.
    """
    cost_map = np.full(grid.shape, np.inf, dtype=float)
    free_mask = grid >= 0

    cost_map[free_mask] = config.COST_LOOKUP[grid[free_mask]]

    if is_rush_hour(t):
        cost_map[free_mask] *= config.RUSH_HOUR_MULTIPLIER[grid[free_mask]]

    if closure_mgr is not None:
        closed_cells = closure_mgr.get_closed_cells(t)
        if closed_cells:
            rows, cols = zip(*closed_cells)
            cost_map[rows, cols] = np.inf

    return cost_map


def build_directional_cost_tensor(grid, t=0, closure_mgr=None):
    """
    Build a 3D (28x28x4) directional entry cost tensor at step t.
    Axis 2 corresponds to DIRECTIONS = [UP, DOWN, LEFT, RIGHT].
    Disallowed entry directions on 'directional' maintenance cells are set to np.inf.
    """
    base_2d = build_cost_map_vectorized(grid, t=t, closure_mgr=closure_mgr)
    tensor = np.repeat(base_2d[:, :, np.newaxis], len(DIRECTIONS), axis=2)

    if closure_mgr is not None and "predictive_dynamic_road_closures" in config.TWISTS:
        for ev in closure_mgr.events:
            if ev["mode"] == "directional" and ev["active_start"] <= t < ev["active_end"]:
                r, c = ev["cell"]
                for d_idx, move_vec in enumerate(DIRECTIONS):
                    if move_vec not in ev["allowed_dirs"]:
                        tensor[r, c, d_idx] = np.inf

    return tensor


def build_cost_map_python(grid, t=0, closure_mgr=None):
    """Build the 28x28 cost matrix at step t using pure Python loops (baseline)."""
    rows, cols = grid.shape
    rush = is_rush_hour(t)
    base_costs = [float(x) for x in config.COST_LOOKUP]
    multipliers = [float(m) for m in config.RUSH_HOUR_MULTIPLIER]
    closed_cells = set(closure_mgr.get_closed_cells(t)) if closure_mgr is not None else set()

    cost_map = []
    for r in range(rows):
        row_costs = []
        for c in range(cols):
            cell = int(grid[r][c])
            if cell < 0 or (r, c) in closed_cells:
                row_costs.append(float("inf"))
            else:
                val = base_costs[cell]
                if rush:
                    val *= multipliers[cell]
                row_costs.append(val)
        cost_map.append(row_costs)

    return np.array(cost_map, dtype=float)


def get_step_cost(grid, next_pos, t):
    """Return the traversal cost of entering next_pos at simulation step t."""
    r, c = next_pos
    terrain = int(grid[r, c])
    if terrain < 0:
        return float("inf")
    cost = float(config.COST_LOOKUP[terrain])
    if is_rush_hour(t):
        cost *= float(config.RUSH_HOUR_MULTIPLIER[terrain])
    return cost


def get_wait_cost(grid, pos, t):
    """
    Shared wait-cost formula used identically by both planners and simulator.
    Clamped to >= MIN_TERRAIN_COST so admissible/consistent heuristics remain valid.
    """
    return max(float(config.MIN_TERRAIN_COST), get_step_cost(grid, pos, t) * 0.5)


def benchmark_cost_map(grid, iterations=2000):
    """Benchmark 2D vectorized NumPy cost map and 3D directional tensor vs pure Python loops."""
    assert np.allclose(build_cost_map_vectorized(grid, t=0), build_cost_map_python(grid, t=0))
    assert np.allclose(build_cost_map_vectorized(grid, t=8), build_cost_map_python(grid, t=8))

    t0 = time.perf_counter()
    for i in range(iterations):
        build_cost_map_vectorized(grid, t=i % 20)
    vec_time = time.perf_counter() - t0

    t_tensor = time.perf_counter()
    for i in range(iterations):
        build_directional_cost_tensor(grid, t=i % 20)
    tensor_time = time.perf_counter() - t_tensor

    t1 = time.perf_counter()
    for i in range(iterations):
        build_cost_map_python(grid, t=i % 20)
    py_time = time.perf_counter() - t1

    speedup = py_time / vec_time

    print(config.STAMP)
    print("-" * 80)
    print(f"Part A Benchmark: Cost Map Construction ({iterations} runs on {grid.shape[0]}x{grid.shape[1]} grid)")
    print(f"Pure Python Loop Time        : {py_time * 1000:.3f} ms")
    print(f"Vectorized 2D NumPy Time     : {vec_time * 1000:.3f} ms")
    print(f"3D Directional Tensor Time   : {tensor_time * 1000:.3f} ms")
    print(f"2D Speed-Up Factor           : {speedup:.2f}x")
    print("-" * 80)

    return {
        "python_ms": round(py_time * 1000, 3),
        "numpy_ms": round(vec_time * 1000, 3),
        "tensor_ms": round(tensor_time * 1000, 3),
        "speedup": round(speedup, 2),
    }


if __name__ == "__main__":
    grid = config.load_grid()
    benchmark_cost_map(grid)