import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from world.environment import GridWorldEnv
from planners.state import (
    custom_mst_heuristic,
    baseline_nearest_manhattan,
    baseline_farthest_manhattan,
    verify_heuristic_consistency,
)
from planners.uninformed.breadth_first_search import bfs
from planners.uninformed.uniform_cost_search import ucs
from planners.uninformed.depth_limited_search import dls
from planners.uninformed.iterative_deepening_search import ids
from planners.informed.greedy_best_first import greedy_best_first
from planners.informed.a_star_search import a_star
from planners.informed.weighted_a_star import weighted_a_star
from planners.state import (
    custom_mst_heuristic,
    baseline_nearest_manhattan,
    baseline_farthest_manhattan,
    verify_heuristic_consistency,
    ablation_heuristic_no_nearest,
    ablation_heuristic_no_mst,
)
from planners.race_aware import plan_race_aware

def run_part_b_benchmarks():
    """Run all Part B search algorithms on the bound-closure instance and return structured metrics."""
    print(config.STAMP)
    print("=" * 96)
    print(f"PART B: SEARCH ALGORITHM BENCHMARK & HEURISTIC VERIFICATION (Instance {config.INSTANCE_ID})")
    print("=" * 96)

    env = GridWorldEnv(seed=config.SEED)
    cons_res = verify_heuristic_consistency(env)
    print(
        f"Heuristic Consistency Check (Custom MST-Path): Consistent={cons_res['consistent']} "
        f"(Transitions Checked={cons_res['transitions_checked']}, Violations={cons_res['violations']}, "
        f"Cells={cons_res['cells_checked']}, Time steps={cons_res['time_steps_checked']})"
    )

    # Bind dynamic closures to initial optimal A* corridor
    init_plan = a_star(env, custom_mst_heuristic, start_t=0)
    if init_plan.get("path"):
        env.closure_mgr.bind_route(init_plan["path"])

    w_eps = float(getattr(config, "INADMISSIBLE_HEURISTIC_WEIGHT", 1.8))

    algorithms = [
        ("BFS", lambda e: bfs(e, start_t=0)),
        ("UCS", lambda e: ucs(e, start_t=0)),
        ("DLS (L=55)", lambda e: dls(e, depth_limit=55, start_t=0)),
        ("IDS (max=65)", lambda e: ids(e, max_depth=65, start_t=0)),
        ("Greedy Best-First", lambda e: greedy_best_first(e, custom_mst_heuristic, start_t=0)),
        ("A* (Nearest-Manh)", lambda e: a_star(e, baseline_nearest_manhattan, start_t=0, label="A* (Nearest)")),
        ("A* (Farthest-Manh)", lambda e: a_star(e, baseline_farthest_manhattan, start_t=0, label="A* (Farthest)")),
        ("A* (Custom MST)", lambda e: a_star(e, custom_mst_heuristic, start_t=0, label="A* (Custom MST)")),
        ("A* (Custom MST, mem cap)", lambda e: a_star(
            e, custom_mst_heuristic, start_t=0, label="A* (mem cap)",
            memory_limit=config.IDS_MEMORY_LIMIT_STATES,
        )),
        ("A* (ablation: no nearest term)", lambda e: a_star(
            e, ablation_heuristic_no_nearest, start_t=0, label="A* (no nearest)",
        )),
        ("A* (ablation: no MST term)", lambda e: a_star(
            e, ablation_heuristic_no_mst, start_t=0, label="A* (no MST)",
        )),
        ("A* (race-aware, +penalty) *", lambda e: plan_race_aware(e, custom_mst_heuristic, start_t=0)),
        (f"Weighted A* (w={w_eps})", lambda e: weighted_a_star(e, custom_mst_heuristic, weight=w_eps, start_t=0)),
    ]

    print("\n1. ALL ALGORITHMS ON BOUND-CLOSURE INSTANCE")
    print("-" * 96)
    print(f"{'Algorithm':<30} {'Status':<14} {'Cost':>8} {'Steps':>7} {'Expanded':>10} {'PeakKB':>10} {'Time(ms)':>10}")
    print("-" * 96)

    rows = []
    for name, runner in algorithms:
        res = runner(env)
        cost_str = f"{res['cost']:.1f}" if res["cost"] != float("inf") else "INF"
        print(
            f"{name:<30} {res['status']:<14} {cost_str:>8} {res['steps']:>7} "
            f"{res['nodes_expanded']:>10} {res['peak_memory_kb']:>10.1f} {res['runtime_ms']:>10.2f}"
        )
        rows.append({
            "algorithm": name,
            "status": res["status"],
            "cost": res["cost"] if res["cost"] != float("inf") else None,
            "steps": res["steps"],
            "nodes_expanded": res["nodes_expanded"],
            "peak_memory_kb": res["peak_memory_kb"],
            "runtime_ms": res["runtime_ms"],
        })
    print("-" * 96)
    print("  * Race-aware cost includes RIVAL_STEAL_PENALTY per contested stop and plans only to winnable stops: NOT comparable to the other rows.")
    opt_row = next(r for r in rows if r["algorithm"] == "A* (Custom MST)")
    w_row = next(r for r in rows if r["algorithm"].startswith("Weighted A*"))
    ids_row = next(r for r in rows if r["algorithm"].startswith("IDS"))
    ablation_rows = [r for r in rows if r["algorithm"].startswith("A* (ablation")]

    print("\n2. HEURISTIC ABLATION (ADMISSIBLE VARIANTS: SAME OPTIMAL COST, DIFFERENT EFFORT)")
    print("-" * 96)
    print(f"  - Full custom heuristic : cost={opt_row['cost']} expanded={opt_row['nodes_expanded']} time={opt_row['runtime_ms']} ms")
    for r in ablation_rows:
        print(f"  - {r['algorithm']:<30}: cost={r['cost']} expanded={r['nodes_expanded']} time={r['runtime_ms']} ms")
    print("-" * 96)
    nearest_row = next(r for r in rows if r["algorithm"] == "A* (Nearest-Manh)")
    no_mst_row = next(r for r in rows if r["algorithm"] == "A* (ablation: no MST term)")
    if (nearest_row["nodes_expanded"], nearest_row["peak_memory_kb"]) == (no_mst_row["nodes_expanded"], no_mst_row["peak_memory_kb"]):
        print("  NOTE: 'no MST term' matches Nearest-Manhattan exactly (same expansions and memory), so it is the baseline, not new evidence.")
    inadmissible_gap = round((w_row["cost"] or 0.0) - (opt_row["cost"] or 0.0), 2)
    print(f"\n3. ADMISSIBLE VS INADMISSIBLE WEIGHTED A* (w={w_eps}) SUBOPTIMALITY GAP")
    print(f"  - Optimal A* Cost : {opt_row['cost']:.1f} ({opt_row['nodes_expanded']} expanded)")
    print(f"  - Weighted A* Cost: {w_row['cost']:.1f} ({w_row['nodes_expanded']} expanded) | Gap = +{inadmissible_gap:.1f}")

    print("\n4. MEMORY COMPARISON: IDS VS OPTIMAL A*")
    print(f"  - IDS Peak Memory : {ids_row['peak_memory_kb']:.1f} KB (Status={ids_row['status']}, Expanded={ids_row['nodes_expanded']})")
    print(f"  - A* Peak Memory  : {opt_row['peak_memory_kb']:.1f} KB (Status={opt_row['status']}, Expanded={opt_row['nodes_expanded']})")
    print("=" * 96)

    return {
        "consistency": cons_res,
        "algorithms": rows,
        "inadmissible_suboptimality_gap": inadmissible_gap,
        "ids_vs_astar_memory": {
            "ids_peak_kb": ids_row["peak_memory_kb"],
            "astar_peak_kb": opt_row["peak_memory_kb"],
        },
    }


if __name__ == "__main__":
    run_part_b_benchmarks()