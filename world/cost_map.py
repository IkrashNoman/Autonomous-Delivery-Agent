import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def is_rush_hour(t):
    """Return True if simulation step t is inside the rush-hour window."""
    if "rush_hour_time_varying_costs" not in config.TWISTS:
        return False
    phase = t % config.RUSH_HOUR_CYCLE
    start, end = config.RUSH_HOUR_WINDOW
    return start <= phase <= end


def build_cost_map_vectorized(grid, t=0):
    """Build the 28x28 cost matrix at step t using vectorized NumPy operations."""
    cost_map = np.full(grid.shape, np.inf, dtype=float)
    free_mask = grid >= 0

    cost_map[free_mask] = config.COST_LOOKUP[grid[free_mask]]

    if is_rush_hour(t):
        cost_map[free_mask] *= config.RUSH_HOUR_MULTIPLIER[grid[free_mask]]

    return cost_map


def build_cost_map_python(grid, t=0):
    """Build the 28x28 cost matrix at step t using pure Python loops (baseline)."""
    rows, cols = grid.shape
    rush = is_rush_hour(t)
    base_costs = [
        config.TERRAIN_COST["road"],
        config.TERRAIN_COST["grass"],
        config.TERRAIN_COST["mud"],
        config.TERRAIN_COST["ford"],
    ]
    multipliers = [2.0, 1.0, 1.25, 1.0]

    cost_map = []
    for r in range(rows):
        row_costs = []
        for c in range(cols):
            cell = int(grid[r][c])
            if cell < 0:
                row_costs.append(float("inf"))
            else:
                val = float(base_costs[cell])
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


def benchmark_cost_map(grid, iterations=2000):
    """Benchmark vectorized NumPy cost map vs pure Python loops."""
    assert np.allclose(build_cost_map_vectorized(grid, t=0), build_cost_map_python(grid, t=0))
    assert np.allclose(build_cost_map_vectorized(grid, t=8), build_cost_map_python(grid, t=8))

    t0 = time.perf_counter()
    for i in range(iterations):
        build_cost_map_vectorized(grid, t=i % 20)
    vec_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    for i in range(iterations):
        build_cost_map_python(grid, t=i % 20)
    py_time = time.perf_counter() - t1

    speedup = py_time / vec_time

    print(config.STAMP)
    print("-" * 80)
    print(f"Part A Benchmark: Cost Map Construction ({iterations} runs on {grid.shape[0]}x{grid.shape[1]} grid)")
    print(f"Pure Python Loop Time : {py_time * 1000:.3f} ms")
    print(f"Vectorized NumPy Time : {vec_time * 1000:.3f} ms")
    print(f"Speed-Up Factor       : {speedup:.2f}x")
    print("-" * 80)

    return {
        "python_ms": round(py_time * 1000, 3),
        "numpy_ms": round(vec_time * 1000, 3),
        "speedup": round(speedup, 2),
    }


if __name__ == "__main__":
    grid = config.load_grid()
    benchmark_cost_map(grid)