import os
import sys
import math
import heapq
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from world.environment import GridWorldEnv
from planners.state import canonical_time
from adversary.evaluation import tune_evaluation_weights, DEFAULT_WEIGHTS
from adversary.game_search import compare_minimax_vs_alphabeta, choose_agent_and_rival_moves


def _plan_single_goal_astar(env, start_pos, goal_pos, start_t):
    """Fast time-aware A* to a single target stop (used for rapid slip recovery)."""
    start_state = (start_pos[0], start_pos[1], canonical_time(start_t))
    h0 = (abs(start_pos[0] - goal_pos[0]) + abs(start_pos[1] - goal_pos[1])) * config.MIN_TERRAIN_COST
    pq = [(h0, 0.0, start_t, start_state)]
    best_g = {start_state: 0.0}
    came_from = {start_state: None}

    while pq:
        _, g_cost, curr_t, curr = heapq.heappop(pq)
        if g_cost > best_g.get(curr, float("inf")):
            continue

        if (curr[0], curr[1]) == goal_pos:
            path = []
            node = curr
            while node is not None:
                path.append((node[0], node[1]))
                node = came_from[node]
            path.reverse()
            return path

        for nxt_pos, step_cost in env.get_valid_neighbors((curr[0], curr[1]), curr_t):
            nxt_state = (nxt_pos[0], nxt_pos[1], canonical_time(curr_t + 1))
            new_g = g_cost + step_cost
            if new_g < best_g.get(nxt_state, float("inf")):
                best_g[nxt_state] = new_g
                came_from[nxt_state] = curr
                h = (abs(nxt_pos[0] - goal_pos[0]) + abs(nxt_pos[1] - goal_pos[1])) * config.MIN_TERRAIN_COST
                heapq.heappush(pq, (new_g + h, new_g, curr_t + 1, nxt_state))

    return [start_pos]

def _plan_multi_stop_tour(env, start_t):
    """
    Plan an efficient multi-stop route prioritizing stops that are close to the agent
    and well-separated from the rival chaser, stitched via time-aware single-goal A*.
    """
    remaining = [s for s in env.stops if s not in env.delivered and s not in env.stolen]
    if not remaining:
        return [env.agent_pos]

    curr_pos = env.agent_pos
    curr_t = start_t
    full_path = [curr_pos]

    while remaining:
        nxt_stop = min(
            remaining,
            key=lambda s: (abs(curr_pos[0] - s[0]) + abs(curr_pos[1] - s[1]))
            - 0.6 * (abs(env.rival_pos[0] - s[0]) + abs(env.rival_pos[1] - s[1])),
        )
        seg = _plan_single_goal_astar(env, curr_pos, nxt_stop, curr_t)
        if len(seg) > 1:
            full_path.extend(seg[1:])
            curr_t += len(seg) - 1
            curr_pos = nxt_stop
        remaining.remove(nxt_stop)

    return full_path

def run_episode(
    env,
    seed,
    slip_prob=config.SLIP_PROBABILITY,
    weights=None,
    max_steps=140,
    search_depth=config.ADVERSARY_SEARCH_DEPTH,
):
    """
    Execute one full episode against the Chaser rival.
    Re-seeds both slip RNG and dynamic road closures per game via env.reset(seed=seed).
    Replans after every slip or route invalidation.
    """
    w = DEFAULT_WEIGHTS if weights is None else weights
    env.slip_prob = slip_prob
    env.reset(seed=seed)

    planned_path = _plan_multi_stop_tour(env, start_t=0)
    if len(planned_path) > 1:
        env.closure_mgr.bind_route(planned_path)
        planned_path = _plan_multi_stop_tour(env, start_t=0)

    path_idx = 1
    replans = 0

    for _ in range(max_steps):
        rival_dist = abs(env.agent_pos[0] - env.rival_pos[0]) + abs(env.agent_pos[1] - env.rival_pos[1])

        # Tactical Alpha-Beta search when rival enters local window
        if rival_dist <= config.LOCAL_GAME_WINDOW_RADIUS:
            tactical_agent_pos, rival_next_pos = choose_agent_and_rival_moves(
                env, depth=search_depth, weights=w
            )
            # Symmetric tactical engagement within depth-5 horizon (rival_dist <= 5)
            next_target = tactical_agent_pos if (rival_dist <= 5 and env.rival_cooldown == 0) else None
        else:
            rival_next_pos = env.rival_pos if env.rival_cooldown > 0 else None
            next_target = None

        if next_target is None:
            if path_idx >= len(planned_path) or planned_path[path_idx - 1] != env.agent_pos:
                replans += 1
                planned_path = _plan_multi_stop_tour(env, start_t=env.t)
                path_idx = 1

            next_target = planned_path[path_idx] if path_idx < len(planned_path) else env.agent_pos

        action = (next_target[0] - env.agent_pos[0], next_target[1] - env.agent_pos[1])
        _, info = env.step(action, rival_next_pos=rival_next_pos)

        if info["slipped"] or info["stole_parcel"]:
            replans += 1
            planned_path = _plan_multi_stop_tour(env, start_t=env.t)
            path_idx = 1
        else:
            path_idx += 1

        if info["done"]:
            break

    delivered_cnt = len(env.delivered)
    stolen_cnt = len(env.stolen)
    if delivered_cnt > stolen_cnt and delivered_cnt >= 2:
        outcome = "WIN"
    elif delivered_cnt == stolen_cnt:
        outcome = "DRAW"
    else:
        outcome = "LOSS"

    return {
        "outcome": outcome,
        "delivered": delivered_cnt,
        "stolen": stolen_cnt,
        "total_cost": round(env.total_cost, 2),
        "steps": env.t,
        "slips": env.slip_count,
        "replans": replans,
    }


def proportion_ci_95(count, n):
    """Return percentage and 95% confidence interval half-width."""
    p = count / n
    se = math.sqrt((p * (1.0 - p)) / n)
    return round(p * 100.0, 2), round(1.96 * se * 100.0, 2)


def run_part_c_experiments(num_games=config.NUM_RIVAL_GAMES):
    """Run all Part C benchmarks and print stamped tables for the report."""
    env = GridWorldEnv(seed=config.SEED)

    print(config.STAMP)
    print("=" * 92)
    print(f"PART C: ADVERSARIAL RIVAL MODELING & STOCHASTIC REPLANNING (Instance {config.INSTANCE_ID})")
    print("=" * 92)

    # 1. Minimax vs Alpha-Beta Pruning Node Counts (Depths 2..5)
    pruning_rows = compare_minimax_vs_alphabeta(env)
    print("\n1. MINIMAX VS ALPHA-BETA PRUNING (REDUCED LOCAL WINDOW)")
    print("-" * 82)
    print(f"{'Depth':>6} {'Minimax Nodes':>15} {'Alpha-Beta Nodes':>18} {'Nodes Saved %':>15} {'AB Time(ms)':>14}")
    print("-" * 82)
    for r in pruning_rows:
        print(
            f"{r['depth']:>6} {r['minimax_nodes']:>15} {r['alphabeta_nodes']:>18} "
            f"{r['saved_pct']:>14.2f}% {r['alphabeta_ms']:>14.2f}"
        )
    print("-" * 82)

    # 2. Evaluation Function Weight Tuning (Grid Search)
    best_weights, tuning_rows = tune_evaluation_weights(env, run_episode)
    print("\n2. EVALUATION FUNCTION GRID SEARCH TUNING")
    print("-" * 82)
    for tr in tuning_rows:
        print(
            f"Candidate {tr['candidate_id']}: {tr['weights']} -> "
            f"Wins={tr['wins']}/{tr['trials']}, AvgUtility={tr['avg_utility']}"
        )
    print(f"Selected Tuned Weights: {best_weights}")

    # 3. Deterministic vs Stochastic Slip Replanning Overhead
    det_res = run_episode(env, seed=config.SEED, slip_prob=0.0, weights=best_weights)
    stoch_res = run_episode(env, seed=config.SEED, slip_prob=config.SLIP_PROBABILITY, weights=best_weights)
    extra_cost = round(stoch_res["total_cost"] - det_res["total_cost"], 2)
    print("\n3. SLIP REPLANNING IMPACT (Deterministic slip=0.0 vs Stochastic slip=0.26)")
    print(
        f"  - Deterministic Baseline : Cost = {det_res['total_cost']}, Steps = {det_res['steps']}, "
        f"Slips = {det_res['slips']}, Replans = {det_res['replans']}"
    )
    print(
        f"  - Stochastic Execution   : Cost = {stoch_res['total_cost']}, Steps = {stoch_res['steps']}, "
        f"Slips = {stoch_res['slips']}, Replans = {stoch_res['replans']} (Extra Cost = +{extra_cost})"
    )

    # 4. 30-Game Independent Tournament with 95% Confidence Intervals
    game_results = [
        run_episode(env, seed=config.SEED + g, slip_prob=config.SLIP_PROBABILITY, weights=best_weights)
        for g in range(num_games)
    ]

    wins = sum(1 for r in game_results if r["outcome"] == "WIN")
    draws = sum(1 for r in game_results if r["outcome"] == "DRAW")
    losses = sum(1 for r in game_results if r["outcome"] == "LOSS")

    win_pct, win_ci = proportion_ci_95(wins, num_games)
    draw_pct, draw_ci = proportion_ci_95(draws, num_games)
    loss_pct, loss_ci = proportion_ci_95(losses, num_games)

    costs = [r["total_cost"] for r in game_results]
    replans_list = [r["replans"] for r in game_results]
    mean_cost = float(np.mean(costs))
    ci_cost = 1.96 * float(np.std(costs, ddof=1)) / math.sqrt(num_games)
    mean_replans = float(np.mean(replans_list))

    print(f"\n4. {num_games}-GAME TOURNAMENT RESULTS AGAINST DEPTH-{config.ADVERSARY_SEARCH_DEPTH} CHASER")
    print("-" * 82)
    print(f"  - Win Rate  : {wins}/{num_games} ({win_pct:.2f}% ± {win_ci:.2f}%)")
    print(f"  - Draw Rate : {draws}/{num_games} ({draw_pct:.2f}% ± {draw_ci:.2f}%)")
    print(f"  - Loss Rate : {losses}/{num_games} ({loss_pct:.2f}% ± {loss_ci:.2f}%)")
    print(f"  - Mean Episode Cost   : {mean_cost:.2f} ± {ci_cost:.2f}")
    print(f"  - Mean Replans / Game : {mean_replans:.2f}")
    print("=" * 92)


if __name__ == "__main__":
    run_part_c_experiments()