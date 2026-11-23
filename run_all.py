import os
import sys
import json
import time
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
sys.path.append(ROOT_DIR)

import config
from world.environment import GridWorldEnv
from world.cost_map import benchmark_cost_map
from planners.benchmark import run_part_b_benchmarks
from adversary.match_simulator import run_part_c_experiments, run_episode, _wilson_ci_95, _mean_ci_95
from dispatcher.eval_suite import run_part_d_evaluation


def run_part_e_sweep_and_failure_analysis(part_b, part_c, part_d):
    """
    Part E: 30-scenario sensitivity sweep per configuration (seed = config.SEED + i, i=0..29)
    reporting Win, Draw, Loss rates and Mean Cost +/- 95% CIs, plus a failure analysis whose
    every number and claim is computed from this run (no fixed text).
    """
    num_scenarios = 30
    depth_val = part_c["search_depth"]
    best_weights = part_c["best_weights"]

    print("\n" + "=" * 116)
    print(config.STAMP)
    print(f"PART E: 30-SCENARIO SENSITIVITY SWEEP & FAILURE ANALYSIS (Instance {config.INSTANCE_ID}, Depth={depth_val})")
    print("=" * 116)
    print(
        f"{'Slip':<6} {'N':<4} {'Win% (95% CI)':<22} {'Draw% (95% CI)':<22} {'Loss% (95% CI)':<22} "
        f"{'Mean Cost +/- 95% CI':<24} {'Slips':>7}"
    )
    print("-" * 116)

    slip_levels = [0.0, 0.13, 0.26, 0.40]
    sweep_rows = []
    for sp in slip_levels:
        episodes = []
        wins = draws = losses = 0
        for i in range(num_scenarios):
            sc_seed = config.SEED + i
            env = GridWorldEnv(seed=sc_seed)
            ep = run_episode(
                env,
                seed=sc_seed,
                slip_prob=sp,
                search_depth=depth_val,
                weights=best_weights,
            )
            episodes.append(ep)
            if ep["outcome"] == "WIN":
                wins += 1
            elif ep["outcome"] == "DRAW":
                draws += 1
            else:
                losses += 1

        wr = round((wins / num_scenarios) * 100.0, 2)
        dr = round((draws / num_scenarios) * 100.0, 2)
        lr = round((losses / num_scenarios) * 100.0, 2)

        w_ci = _wilson_ci_95(wins, num_scenarios)
        d_ci = _wilson_ci_95(draws, num_scenarios)
        l_ci = _wilson_ci_95(losses, num_scenarios)

        m_cost, c_half = _mean_ci_95([x["total_cost"] for x in episodes])
        m_slips, s_half = _mean_ci_95([x["slips"] for x in episodes])
        m_rep, _ = _mean_ci_95([x["replans"] for x in episodes])

        row = {
            "slip_prob": sp,
            "num_scenarios": num_scenarios,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate_pct": wr,
            "win_wilson_ci95": list(w_ci),
            "draw_rate_pct": dr,
            "draw_wilson_ci95": list(d_ci),
            "loss_rate_pct": lr,
            "loss_wilson_ci95": list(l_ci),
            "mean_cost": m_cost,
            "mean_cost_ci95_half": c_half,
            "mean_slips": m_slips,
            "mean_slips_ci95_half": s_half,
            "mean_replans": m_rep,
            "seed0_episode": episodes[0],
        }
        sweep_rows.append(row)

        w_str = f"{wr:.1f}% [{w_ci[0]:.1f},{w_ci[1]:.1f}]"
        d_str = f"{dr:.1f}% [{d_ci[0]:.1f},{d_ci[1]:.1f}]"
        l_str = f"{lr:.1f}% [{l_ci[0]:.1f},{l_ci[1]:.1f}]"
        c_str = f"{m_cost:.1f} +/- {c_half:.1f}"
        print(f"{sp:<6.2f} {num_scenarios:<4} {w_str:<22} {d_str:<22} {l_str:<22} {c_str:<24} {m_slips:>7.2f}")

    print("-" * 116)

    # ---------- Computed failure analysis (every figure comes from this run) ----------
    t_res = part_c["tournament"]
    n_games = t_res["num_games"]
    s0 = sweep_rows[0]

    opt_row = next((r for r in part_b["algorithms"] if r["algorithm"] == "A* (Custom MST)"), None)
    wa_row = next((r for r in part_b["algorithms"] if r["algorithm"].startswith("Weighted A*")), None)
    dls_row = next((r for r in part_b["algorithms"] if r["algorithm"].startswith("DLS")), None)
    ids_row = next((r for r in part_b["algorithms"] if r["algorithm"].startswith("IDS")), None)
    cap_row = next((r for r in part_b["algorithms"] if "mem cap" in r["algorithm"]), None)

    def _status_text(r):
        if r is None:
            return "not run"
        return f"{r['status']} after {r['nodes_expanded']} expansions"

    if opt_row and wa_row and opt_row["nodes_expanded"]:
        wa_saving = (1.0 - wa_row["nodes_expanded"] / opt_row["nodes_expanded"]) * 100.0
        wa_text = (
            f"{wa_row['algorithm']} expanded {wa_row['nodes_expanded']} nodes "
            f"({wa_saving:.1f}% fewer than optimal A* with {opt_row['nodes_expanded']}), "
            f"cost {wa_row['cost']} vs optimal {opt_row['cost']} (gap +{part_b['inadmissible_suboptimality_gap']})."
        )
    else:
        wa_text = "weighted A* comparison unavailable."
    opt_text = (
        f"Optimal A* found a {opt_row['steps']}-step plan of cost {opt_row['cost']}."
        if opt_row and opt_row["cost"] is not None
        else "Optimal A* result unavailable."
    )

    part_b_text = (
        f"{opt_text} {wa_text} Uninformed search: DLS is {_status_text(dls_row)}; "
        f"IDS is {_status_text(ids_row)}. "
        f"Optimal A* under the memory cap of {config.IDS_MEMORY_LIMIT_STATES} stored states: {_status_text(cap_row)}."
    )

    draws = t_res["draws"]
    part_c_text = (
        f"Across {n_games} stochastic games at depth {depth_val}: "
        f"WIN={t_res['wins']}/{n_games} ({t_res['win_rate_pct']}%, Wilson CI {t_res['win_wilson_ci95_pct']}), "
        f"DRAW={draws}/{n_games} ({t_res['draw_rate_pct']}%, Wilson CI {t_res['draw_wilson_ci95_pct']}), "
        f"LOSS={t_res['losses']}/{n_games} ({t_res['loss_rate_pct']}%, Wilson CI {t_res['loss_wilson_ci95_pct']}). "
        f"Mean cost {t_res['mean_cost']} +/- {t_res['mean_cost_ci95_half']}. "
        f"Deterministic (slip=0.00) wins in this sweep: {s0['wins']}/{num_scenarios}."
    )

    win_series = [r["win_rate_pct"] for r in sweep_rows]
    cost_series = [r["mean_cost"] for r in sweep_rows]
    peak = sweep_rows[int(np.argmax(win_series))]
    win_monotone = (
        all(b >= a for a, b in zip(win_series, win_series[1:]))
        or all(b <= a for a, b in zip(win_series, win_series[1:]))
    )
    cost_monotone = all(b >= a for a, b in zip(cost_series, cost_series[1:]))
    sweep_text = (
        "Win rate by slip: " + ", ".join(f"{r['slip_prob']:.2f} -> {r['win_rate_pct']}%" for r in sweep_rows)
        + f". Peak at slip={peak['slip_prob']:.2f}. "
        + ("Win rate is monotone in slip. " if win_monotone else "Win rate is NOT monotone in slip. ")
        + "Mean cost by slip: " + ", ".join(f"{r['mean_cost']}" for r in sweep_rows)
        + (". Mean cost increases monotonically with slip." if cost_monotone else ". Mean cost is NOT monotone in slip.")
    )

    failed = [r for r in part_d["records"] if not r["full_pass"]]
    failed_text = (
        "; ".join(
            f"{r['id']} (expected {r['expected']}, actual {r['actual']}, source {r['source']})" for r in failed
        ) or "none"
    )
    offline_ids = [r["id"] for r in part_d["records"] if r.get("source") == "OFFLINE_FALLBACK"]
    part_d_text = (
        f"Mode={part_d['mode']} | Full-pass={part_d['full_pass_count']}/{part_d['total_prompts']} | "
        f"Live API calls={part_d['total_api_calls']} | Offline fallbacks={len(offline_ids)} {offline_ids}. "
        f"Failed prompts: {failed_text}. "
        f"With schema guardrails: {part_d['guarded_blocked']}/{part_d['total_unsafe']} schema-relevant unsafe prompts blocked "
        f"(unsafe actions executed: {part_d.get('guarded_leaks', 'n/a')}). "
        f"Without schema guardrails: {part_d['unvalidated_blocked']}/{part_d['total_unsafe']} blocked "
        f"(unsafe actions executed: {part_d.get('unguarded_leaks', 'n/a')})."
    )

    failure_analysis = {
        "part_b_search_failures": part_b_text,
        "part_c_adversarial_failures": part_c_text,
        "part_d_llm_and_guardrail_failures": part_d_text,
        "part_e_sensitivity_findings": sweep_text,
    }

    print("\nDETAILED MULTI-PART FAILURE ANALYSIS:")
    print(config.STAMP)
    for section, narrative in failure_analysis.items():
        print(f"  [{section}] {narrative}")
    print("=" * 116)

    return {"slip_sensitivity_sweep": sweep_rows, "failure_analysis": failure_analysis}


    

def save_figures(part_b, part_c, part_d, part_e):
    """Generate 4 stamped, publication-grade report figures in figures/."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    generated = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed; skipping PNG chart rendering.")
        return generated

    # Figure 1: Expanded Nodes across Search Algorithms (Part B)
    algos = [r["algorithm"] for r in part_b["algorithms"]]
    nodes = [r["nodes_expanded"] for r in part_b["algorithms"]]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(algos, nodes, color="#2563eb")
    ax.set_title(f"Part B: Expanded Nodes by Search Algorithm (Instance {config.INSTANCE_ID})\n{config.STAMP}", fontsize=9.5)
    ax.set_ylabel("Nodes Expanded")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    f1 = os.path.join(FIGURES_DIR, "fig1_search_nodes.png")
    fig.savefig(f1, dpi=180)
    plt.close(fig)
    generated.append(f1)

    # Figure 2: Minimax vs Alpha-Beta Pruning across Depths 2..5 (Part C)
    pb = part_c.get("pruning_benchmark") or []
    if pb:
        depths = [f"Depth {r['depth']}" for r in pb]
        mm_nodes = [r["minimax_nodes"] for r in pb]
        ab_nodes = [r["alphabeta_nodes"] for r in pb]
        x = range(len(depths))
        fig, ax = plt.subplots(figsize=(7.8, 4.5))
        ax.bar([i - 0.18 for i in x], mm_nodes, width=0.35, label="Minimax (Unpruned)", color="#f59e0b")
        ax.bar([i + 0.18 for i in x], ab_nodes, width=0.35, label="Alpha-Beta (Ordered)", color="#16a34a")
        ax.set_xticks(list(x))
        ax.set_xticklabels(depths)
        ax.set_ylabel("Nodes Evaluated")
        ax.set_title(f"Part C: Minimax vs Alpha-Beta Pruning (Depths 2–{max(r['depth'] for r in pb)})\n{config.STAMP}", fontsize=9.5)
        ax.legend()
        plt.tight_layout()
        f2 = os.path.join(FIGURES_DIR, "fig2_minimax_vs_alphabeta.png")
        fig.savefig(f2, dpi=180)
        plt.close(fig)
        generated.append(f2)

    # Figure 3: Part E 30-Scenario Slip Sensitivity Sweep (Mean Cost +/- 95% CI)
    sw = part_e["slip_sensitivity_sweep"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    slips = [f"slip={r['slip_prob']:.2f}" for r in sw]
    means = [r["mean_cost"] for r in sw]
    errs = [r["mean_cost_ci95_half"] for r in sw]
    ax.bar(slips, means, yerr=errs, capsize=5, color="#0284c7")
    ax.set_ylabel("Mean Episode Cost (+/- 95% CI)")
    ax.set_title(f"Part E: 30-Scenario Slip Sensitivity Sweep (Depth={part_c['search_depth']})\n{config.STAMP}", fontsize=9.5)
    plt.tight_layout()
    f3 = os.path.join(FIGURES_DIR, "fig3_slip_sensitivity_sweep.png")
    fig.savefig(f3, dpi=180)
    plt.close(fig)
    generated.append(f3)

    # Figure 4: Part D Validation Layer Safety Ablation
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(
        ["Without Schema Guardrails", "With Schema Guardrails"],
        [part_d["unvalidated_blocked"], part_d["guarded_blocked"]],
        color=["#dc2626", "#2563eb"],
    )
    ax.set_ylim(0, part_d["total_unsafe"] + 1)
    ax.set_ylabel(f"Unsafe Prompts Blocked (out of {part_d['total_unsafe']})")
    ax.set_title(f"Part D: Validation Layer Safety Ablation\n{config.STAMP}", fontsize=9.5)
    plt.tight_layout()
    f4 = os.path.join(FIGURES_DIR, "fig4_guardrail_ablation.png")
    fig.savefig(f4, dpi=180)
    plt.close(fig)
    generated.append(f4)

    return generated


def main():
    t_start = time.perf_counter()
    use_live = "--live" in sys.argv

    part_a = benchmark_cost_map(config.load_grid(), iterations=2000)
    part_b = run_part_b_benchmarks()
    part_c = run_part_c_experiments(num_games=config.NUM_RIVAL_GAMES, search_depth=config.ADVERSARY_SEARCH_DEPTH)
    part_d = run_part_d_evaluation(use_live_api=use_live)
    part_e = run_part_e_sweep_and_failure_analysis(part_b, part_c, part_d)
    figures = save_figures(part_b, part_c, part_d, part_e)

    total_sec = round(time.perf_counter() - t_start, 2)
    artifact = {
        "stamp": config.STAMP,
        "instance_id": config.INSTANCE_ID,
        "seed": config.SEED,
        "part_a": part_a,
        "part_b": part_b,
        "part_c": part_c,
        "part_d": part_d,
        "part_e": part_e,
        "generated_figures": figures,
        "total_reproduction_time_sec": total_sec,
    }
    out_path = os.path.join(ROOT_DIR, "results_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print("\n" + "=" * 116)
    print(config.STAMP)
    print(f"ALL PARTS (A–E) VERIFIED IN {total_sec}s")
    print(f"  - Consolidated JSON Artifact : {out_path}")
    print(f"  - Generated Stamped Figures  : {len(figures)} PNGs in {FIGURES_DIR}")
    print("  - Part E Interactive Web UI  : python web/server.py (http://127.0.0.1:8000)")
    print("=" * 116)


if __name__ == "__main__":
    main()
