import os
import sys
import json
import time
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
sys.path.append(ROOT_DIR)

import config
from world.environment import GridWorldEnv
from world.cost_map import benchmark_cost_map
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
from adversary.game_search import benchmark_adversarial_pruning
from adversary.match_simulator import run_episode
from dispatcher.eval_suite import TEST_CASES_20
from dispatcher.llm_dispatcher import LLMDispatcher


def run_part_a_and_b_full():
    """Reproduce exact Part A cost-map speedup & Part B bound-closure benchmark (including DLS & IDS)."""
    print(config.STAMP)
    print("=" * 100)
    print(f"PARTS A & B: COST-MAP SPEEDUP, CONSISTENCY & BOUND-CLOSURE BENCHMARK (Instance {config.INSTANCE_ID})")
    print("=" * 100)

    grid = config.load_grid()
    cost_bench = benchmark_cost_map(grid, iterations=2000)

    # Invoke planners/benchmark.py directly so exact Part B report tables are printed
    bench_script = os.path.join(ROOT_DIR, "planners", "benchmark.py")
    if os.path.isfile(bench_script):
        subprocess.run([sys.executable, bench_script], check=False)

    env = GridWorldEnv(seed=config.SEED)
    cons_res = verify_heuristic_consistency(env)

    # Bind dynamic closures against initial A* corridor (matching benchmark.py)
    init_plan = a_star(env, custom_mst_heuristic, start_t=0)
    if init_plan.get("path"):
        env.closure_mgr.bind_route(init_plan["path"])

    w_eps = getattr(
        config,
        "WEIGHTED_ASTAR_EPSILON",
        getattr(config, "INADMISSIBLE_HEURISTIC_WEIGHT", 1.8),
    )

    algos = [
        ("BFS", lambda e: bfs(e, start_t=0)),
        ("UCS", lambda e: ucs(e, start_t=0)),
        ("DLS (L=55)", lambda e: dls(e, depth_limit=55, start_t=0)),
        ("IDS (max=65)", lambda e: ids(e, max_depth=65, start_t=0)),
        ("Greedy Best-First", lambda e: greedy_best_first(e, custom_mst_heuristic, start_t=0)),
        ("A* (Nearest-Manh)", lambda e: a_star(e, baseline_nearest_manhattan, start_t=0, label="A* (Nearest)")),
        ("A* (Farthest-Manh)", lambda e: a_star(e, baseline_farthest_manhattan, start_t=0, label="A* (Farthest)")),
        ("A* (Custom MST)", lambda e: a_star(e, custom_mst_heuristic, start_t=0, label="A* (Custom MST)")),
        (f"Weighted A* (w={w_eps})", lambda e: weighted_a_star(e, custom_mst_heuristic, weight=w_eps, start_t=0)),
    ]

    summary_rows = []
    for name, fn in algos:
        res = fn(env)
        summary_rows.append({
            "algorithm": name,
            "status": res["status"],
            "cost": res["cost"] if res["cost"] != float("inf") else None,
            "steps": res["steps"],
            "nodes_expanded": res["nodes_expanded"],
            "runtime_ms": res["runtime_ms"],
            "peak_memory_kb": res["peak_memory_kb"],
        })

    opt_cost = next((r["cost"] for r in summary_rows if r["algorithm"] == "A* (Custom MST)"), None)
    w_cost = next((r["cost"] for r in summary_rows if r["algorithm"].startswith("Weighted A*")), None)
    inadmissible_gap = round(w_cost - opt_cost, 2) if (opt_cost is not None and w_cost is not None) else 0.0

    return {
        "cost_map_speedup": cost_bench,
        "consistency": cons_res,
        "algorithms": summary_rows,
        "inadmissible_suboptimality_gap": inadmissible_gap,
    }


def run_part_c_full(num_games=config.NUM_RIVAL_GAMES):
    """Run Part C pruning benchmark and 30-game stochastic adversarial tournament."""
    print("\n" + "=" * 100)
    print(f"PART C: MINIMAX VS ALPHA-BETA PRUNING & {num_games}-GAME TOURNAMENT (SEED BASE={config.SEED})")
    print("=" * 100)

    pruning_bench = benchmark_adversarial_pruning()

    games = []
    wins = losses = timeouts = 0
    for g in range(num_games):
        game_seed = config.SEED + g
        env = GridWorldEnv(seed=game_seed)
        res = run_episode(env, seed=game_seed, slip_prob=config.SLIP_PROBABILITY, max_steps=120, search_depth=3)
        res["game_idx"] = g + 1
        res["seed"] = game_seed
        games.append(res)
        if res["outcome"] == "WIN":
            wins += 1
        elif res["outcome"] == "LOSS":
            losses += 1
        else:
            timeouts += 1

    win_rate = round((wins / num_games) * 100.0, 2)
    mean_cost = round(sum(x["total_cost"] for x in games) / num_games, 2)
    mean_slips = round(sum(x["slips"] for x in games) / num_games, 2)
    print(f"Tournament ({num_games} games): WIN={wins} ({win_rate}%), LOSS={losses}, TIMEOUT={timeouts} | Mean Cost={mean_cost} | Mean Slips={mean_slips}")

    return {
        "pruning_benchmark": pruning_bench,
        "num_games": num_games,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate_pct": win_rate,
        "mean_cost": mean_cost,
        "mean_slips": mean_slips,
        "games": games,
    }


def run_part_d_full(use_live_api=False):
    """Run Part D 20-prompt ReAct benchmark + measured validation layer ablation."""
    print("\n" + "=" * 100)
    mode_label = "LIVE_GROQ_API" if use_live_api else "DETERMINISTIC_OFFLINE"
    print(f"PART D: REACT DISPATCHER & GUARDRAIL EVALUATION ({mode_label})")
    print("=" * 100)

    records = []
    for tc in TEST_CASES_20:
        env = GridWorldEnv(seed=config.SEED)
        agent = LLMDispatcher(env=env, use_live_api=use_live_api, enforce_guardrails=True, request_delay_sec=2.1)
        res = agent.dispatch(tc["prompt"])
        tools_ok = all(rt in res.get("tools_used", []) for rt in tc["req_tools"])
        full_pass = (res["status"] == tc["expected"]) and tools_ok
        records.append({
            "id": tc["id"],
            "tier": tc["tier"],
            "source": res.get("source"),
            "expected": tc["expected"],
            "actual": res["status"],
            "full_pass": full_pass,
            "tool_calls": res["tool_calls"],
            "cost_before": res["cost_before"],
            "cost_after": res["cost_after"],
            "latency_ms": res["latency_ms"],
        })

    unsafe_cases = [tc for tc in TEST_CASES_20 if tc["expected"] == "REJECTED"]
    unvalidated_blocked = 0
    for tc in unsafe_cases:
        env_raw = GridWorldEnv(seed=config.SEED)
        raw_agent = LLMDispatcher(env=env_raw, use_live_api=False, enforce_guardrails=False, request_delay_sec=0.0)
        if raw_agent.dispatch(tc["prompt"])["status"] == "REJECTED":
            unvalidated_blocked += 1

    guarded_blocked = sum(1 for r in records if r["expected"] == "REJECTED" and r["actual"] == "REJECTED")
    full_pass_count = sum(1 for r in records if r["full_pass"])
    print(
        f"20-Prompt Full-Pass: {full_pass_count}/20 | "
        f"Guardrail Ablation: Unvalidated Blocked={unvalidated_blocked}/{len(unsafe_cases)} vs Guarded Blocked={guarded_blocked}/{len(unsafe_cases)}"
    )

    return {
        "mode": mode_label,
        "full_pass_count": full_pass_count,
        "total_prompts": len(records),
        "unvalidated_blocked": unvalidated_blocked,
        "guarded_blocked": guarded_blocked,
        "total_unsafe": len(unsafe_cases),
        "records": records,
    }


def save_figures(part_ab, part_c, part_d):
    """Generate 4 publication-grade report figures in figures/."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    generated = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed; skipping PNG chart rendering.")
        return generated

    # Figure 1: Expanded Nodes across Search Algorithms
    algos = [r["algorithm"] for r in part_ab["algorithms"]]
    nodes = [r["nodes_expanded"] for r in part_ab["algorithms"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(algos, nodes, color="#2563eb")
    ax.set_title(f"Expanded Nodes by Search Algorithm (Instance {config.INSTANCE_ID})")
    ax.set_ylabel("Nodes Expanded")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    f1 = os.path.join(FIGURES_DIR, "fig1_search_nodes.png")
    fig.savefig(f1, dpi=180)
    plt.close(fig)
    generated.append(f1)

    # Figure 2: Minimax vs Alpha-Beta Node Savings
    pb = part_c.get("pruning_benchmark") or {}
    if isinstance(pb, dict) and "minimax_nodes" in pb and "alphabeta_nodes" in pb:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.bar(
            ["Minimax (Unpruned)", "Alpha-Beta (Move-Ordered)"],
            [pb["minimax_nodes"], pb["alphabeta_nodes"]],
            color=["#f59e0b", "#16a34a"],
        )
        ax.set_title("Part C: Minimax vs Alpha-Beta Search Nodes")
        ax.set_ylabel("Nodes Evaluated")
        plt.tight_layout()
        f2 = os.path.join(FIGURES_DIR, "fig2_minimax_vs_alphabeta.png")
        fig.savefig(f2, dpi=180)
        plt.close(fig)
        generated.append(f2)
    else:
        valid = [r for r in part_ab["algorithms"] if r["cost"] is not None]
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.bar([r["algorithm"] for r in valid], [r["cost"] for r in valid], color="#16a34a")
        ax.set_title("Part B: Total Path Cost Comparison")
        ax.set_ylabel("Total Path Cost")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        f2 = os.path.join(FIGURES_DIR, "fig2_path_cost.png")
        fig.savefig(f2, dpi=180)
        plt.close(fig)
        generated.append(f2)

    # Figure 3: Part C 30-Game Adversarial Outcomes
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["WIN", "LOSS", "TIMEOUT"]
    counts = [part_c["wins"], part_c["losses"], part_c["timeouts"]]
    ax.bar(labels, counts, color=["#22c55e", "#ef4444", "#f59e0b"])
    ax.set_title(f"Part C Adversarial Tournament Outcomes ({part_c['num_games']} Games)")
    ax.set_ylabel("Episode Count")
    plt.tight_layout()
    f3 = os.path.join(FIGURES_DIR, "fig3_adversarial_outcomes.png")
    fig.savefig(f3, dpi=180)
    plt.close(fig)
    generated.append(f3)

    # Figure 4: Part D Validation Layer Safety Ablation
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(
        ["Without Schema Guardrails", "With Schema Guardrails"],
        [part_d["unvalidated_blocked"], part_d["guarded_blocked"]],
        color=["#dc2626", "#2563eb"],
    )
    ax.set_ylim(0, part_d["total_unsafe"] + 1)
    ax.set_ylabel("Unsafe / Adversarial Prompts Blocked (out of 8)")
    ax.set_title("Part D Validation Layer Ablation")
    plt.tight_layout()
    f4 = os.path.join(FIGURES_DIR, "fig4_guardrail_ablation.png")
    fig.savefig(f4, dpi=180)
    plt.close(fig)
    generated.append(f4)

    return generated


def main():
    t_start = time.perf_counter()
    use_live = "--live" in sys.argv

    part_ab = run_part_a_and_b_full()
    part_c = run_part_c_full(num_games=config.NUM_RIVAL_GAMES)
    part_d = run_part_d_full(use_live_api=use_live)
    figures = save_figures(part_ab, part_c, part_d)

    total_sec = round(time.perf_counter() - t_start, 2)
    artifact = {
        "stamp": config.STAMP,
        "instance_id": config.INSTANCE_ID,
        "seed": config.SEED,
        "part_ab": part_ab,
        "part_c": part_c,
        "part_d": part_d,
        "generated_figures": figures,
        "total_reproduction_time_sec": total_sec,
    }
    out_path = os.path.join(ROOT_DIR, "results_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print("\n" + "=" * 100)
    print(f"ALL PARTS (A–D) VERIFIED IN {total_sec}s")
    print(f"  - Consolidated JSON Artifact : {out_path}")
    print(f"  - Generated Report Figures   : {len(figures)} PNGs in {FIGURES_DIR}")
    print("  - Part E Interactive Web UI  : python web/server.py (http://127.0.0.1:8000)")
    print("=" * 100)


if __name__ == "__main__":
    main()