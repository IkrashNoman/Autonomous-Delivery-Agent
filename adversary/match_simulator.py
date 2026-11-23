import os
import sys
import math
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from world.environment import GridWorldEnv
from planners.state import custom_mst_heuristic
from planners.informed.a_star_search import a_star
from adversary.game_search import choose_agent_and_rival_moves, benchmark_adversarial_pruning
from adversary.evaluation import tune_evaluation_weights


def _wilson_ci_95(count, n):
    """
    Wilson score 95% binomial confidence interval (z = 1.96).
    Never collapses to zero width at 0% or 100%.
    """
    if n <= 0:
        return 0.0, 0.0
    z = 1.96
    phat = count / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * n)) / n)) / denom
    return round(max(0.0, center - half) * 100.0, 2), round(min(1.0, center + half) * 100.0, 2)


def _mean_ci_95(values):
    """Compute sample mean and 95% confidence half-width (1.96 * sigma / sqrt(n))."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    mean_val = float(np.mean(arr))
    if n == 1:
        return round(mean_val, 2), 0.0
    std_val = float(np.std(arr, ddof=1))
    half = 1.96 * (std_val / math.sqrt(n))
    return round(mean_val, 2), round(half, 2)


def _select_contested_target(env):
    """
    Choose the remaining delivery stop with the strongest race margin against the rival:
    minimizes (agent_dist - 0.65 * (rival_dist + rival_cooldown)) so the agent prioritizes
    reachable stops before the rival can steal them.
    """
    rem = [s for s in env.stops if s not in env.delivered and s not in env.stolen]
    if not rem:
        return None

    def race_score(stop):
        d_a = abs(env.agent_pos[0] - stop[0]) + abs(env.agent_pos[1] - stop[1])
        d_r = abs(env.rival_pos[0] - stop[0]) + abs(env.rival_pos[1] - stop[1]) + env.rival_cooldown
        return d_a - 0.65 * d_r

    return min(rem, key=race_score)


def _plan_multi_stop_tour(env):
    """Plan optimal A* leg to the highest-priority contested stop."""
    target = _select_contested_target(env)
    if target is None:
        return []

    orig_stops = env.stops
    env.stops = [target]
    try:
        res = a_star(env, custom_mst_heuristic, start_t=env.t, track_memory=False)
    finally:
        env.stops = orig_stops
    return res.get("path", [])


def run_episode(
    env,
    seed=config.SEED,
    slip_prob=config.SLIP_PROBABILITY,
    max_steps=120,
    search_depth=config.ADVERSARY_SEARCH_DEPTH,
    weights=None,
):
    """
    Simulate a full adversarial multi-stop delivery episode.
    Outcome classification (majority parcel contest):
      - WIN  : delivered > stolen
      - DRAW : delivered == stolen (including tied splits or unresolved stalemates)
      - LOSS : delivered < stolen
    """
    env.slip_prob = slip_prob
    env.reset(seed=seed)

    active_path = _plan_multi_stop_tour(env)
    if active_path:
        env.closure_mgr.bind_route(active_path)
        active_path = _plan_multi_stop_tour(env)

    replans = 0
    recent_positions = [env.agent_pos]

    while env.t < max_steps:
        if len(env.delivered) + len(env.stolen) == len(env.stops):
            break

        need_replan = (
            not active_path
            or len(active_path) < 2
            or active_path[0] != env.agent_pos
            or not env.closure_mgr.is_move_allowed(env.agent_pos, active_path[1], env.t)
        )
        if need_replan:
            active_path = _plan_multi_stop_tour(env)
            replans += 1

        next_target = active_path[1] if len(active_path) >= 2 else env.agent_pos
        rival_dist = abs(env.agent_pos[0] - env.rival_pos[0]) + abs(env.agent_pos[1] - env.rival_pos[1])

        if rival_dist <= config.LOCAL_GAME_WINDOW_RADIUS:
            tactical_a, rival_next = choose_agent_and_rival_moves(env, depth=search_depth, weights=weights)
            # Engage Alpha-Beta tactical move when rival threatens within 4 cells and wouldn't cause a 2-step loop
            if rival_dist <= 4 and tactical_a != env.agent_pos:
                if len(recent_positions) < 2 or tactical_a != recent_positions[-2]:
                    next_target = tactical_a
        else:
            rival_next = None

        action = (next_target[0] - env.agent_pos[0], next_target[1] - env.agent_pos[1])
        _, info = env.step(action, rival_next_pos=rival_next)
        recent_positions.append(env.agent_pos)

        if info.get("slipped") or info.get("stole_parcel"):
            active_path = _plan_multi_stop_tour(env)
            replans += 1
        elif len(active_path) >= 2 and active_path[1] == env.agent_pos:
            active_path = active_path[1:]
        else:
            active_path = _plan_multi_stop_tour(env)
            replans += 1

    n_del = len(env.delivered)
    n_sto = len(env.stolen)
    if n_del > n_sto:
        outcome = "WIN"
    elif n_del == n_sto:
        outcome = "DRAW"
    else:
        outcome = "LOSS"

    return {
        "outcome": outcome,
        "delivered": n_del,
        "stolen": n_sto,
        "total_cost": round(env.total_cost, 2),
        "steps": env.t,
        "slips": env.slip_count,
        "replans": replans,
    }

def run_replanning_cost_analysis(num_games, search_depth, weights, slip_prob=None):
    """
    Paired comparison on identical seeds (same closure schedule, same rival):
    episode with slips (replanning after every slip) minus the same episode with slip = 0.
    Reports how often slip-triggered replanning fired and the extra cost it causes.
    """
    if slip_prob is None:
        slip_prob = config.SLIP_PROBABILITY
    diffs, replans, slips = [], [], []
    games_with_slip = 0
    for g in range(num_games):
        seed = config.SEED + g
        ep_det = run_episode(
            GridWorldEnv(seed=seed), seed=seed, slip_prob=0.0, search_depth=search_depth, weights=weights
        )
        ep_sto = run_episode(
            GridWorldEnv(seed=seed), seed=seed, slip_prob=slip_prob, search_depth=search_depth, weights=weights
        )
        diffs.append(ep_sto["total_cost"] - ep_det["total_cost"])
        replans.append(ep_sto["replans"])
        slips.append(ep_sto["slips"])
        if ep_sto["slips"] > 0:
            games_with_slip += 1

    extra_mean, extra_half = _mean_ci_95(diffs)
    rep_mean, _ = _mean_ci_95(replans)
    slip_mean, _ = _mean_ci_95(slips)
    return {
        "num_games": num_games,
        "slip_prob": slip_prob,
        "games_with_slip": games_with_slip,
        "games_with_slip_pct": round(games_with_slip / max(1, num_games) * 100.0, 2),
        "mean_replans": rep_mean,
        "mean_slips": slip_mean,
        "extra_cost_mean": extra_mean,
        "extra_cost_ci95_half": extra_half,
        "paired_differences": [round(d, 2) for d in diffs],
    }


def run_part_c_experiments(num_games=config.NUM_RIVAL_GAMES, search_depth=config.ADVERSARY_SEARCH_DEPTH):
    """Run all Part C experiments at the rival's full depth limit (ADVERSARY_SEARCH_DEPTH = 5)."""
    depth_val = int(search_depth)
    print(config.STAMP)
    print("=" * 98)
    print(f"PART C: ADVERSARIAL CHASER EXPERIMENTS & TOURNAMENT (Instance {config.INSTANCE_ID}, Depth={depth_val})")
    print("=" * 98)

    pruning_rows = benchmark_adversarial_pruning(depths=tuple(range(2, depth_val + 1)))

    env_tune = GridWorldEnv(seed=config.SEED)
    best_weights, tuning_table = tune_evaluation_weights(
        env_tune,
        lambda e, *args, **kwargs: run_episode(
            e,
            *args,
            **{
                "slip_prob": config.SLIP_PROBABILITY,
                "search_depth": depth_val,
                **kwargs,
            },
        ),
    )

    print(f"\n1. EVALUATION WEIGHT GRID SEARCH ({len(tuning_table)} CANDIDATES x {tuning_table[0]['trials']} TRIALS AT DEPTH={depth_val})")
    print("-" * 98)
    for row in tuning_table:
        print(
            f"  Candidate {row['candidate_id']}: wins={row['wins']}/{row['trials']}, draws={row['draws']}/{row['trials']}, "
            f"avg_utility={row['avg_utility']} | weights={row['weights']}"
        )
    print(f"  Selected Best Weights: {best_weights}")

    print(f"\n2. DETERMINISTIC (slip=0.0) VS STOCHASTIC (slip={config.SLIP_PROBABILITY}) BASELINE (DEPTH={depth_val})")
    print("-" * 98)
    det_res = run_episode(
        GridWorldEnv(seed=config.SEED),
        seed=config.SEED,
        slip_prob=0.0,
        search_depth=depth_val,
        weights=best_weights,
    )
    sto_res = run_episode(
        GridWorldEnv(seed=config.SEED),
        seed=config.SEED,
        slip_prob=config.SLIP_PROBABILITY,
        search_depth=depth_val,
        weights=best_weights,
    )
    print(f"  - Deterministic (slip=0.00): {det_res}")
    print(f"  - Stochastic    (slip={config.SLIP_PROBABILITY}): {sto_res}")

    print(f"\n3. {num_games}-GAME STOCHASTIC TOURNAMENT (DEPTH={depth_val}, WIN/DRAW/LOSS & MEAN COST 95% CIs)")
    print("-" * 98)
    games = []
    wins = draws = losses = 0
    for g in range(num_games):
        g_seed = config.SEED + g
        ep = run_episode(
            GridWorldEnv(seed=g_seed),
            seed=g_seed,
            slip_prob=config.SLIP_PROBABILITY,
            search_depth=depth_val,
            weights=best_weights,
        )
        ep["game"] = g + 1
        ep["seed"] = g_seed
        games.append(ep)
        if ep["outcome"] == "WIN":
            wins += 1
        elif ep["outcome"] == "DRAW":
            draws += 1
        else:
            losses += 1
        if (g + 1) % 10 == 0:
            print(f"  ... completed {g + 1}/{num_games} games (WIN={wins}, DRAW={draws}, LOSS={losses})")

    win_rate = round((wins / max(1, num_games)) * 100.0, 2)
    draw_rate = round((draws / max(1, num_games)) * 100.0, 2)
    loss_rate = round((losses / max(1, num_games)) * 100.0, 2)

    win_ci = _wilson_ci_95(wins, num_games)
    draw_ci = _wilson_ci_95(draws, num_games)
    loss_ci = _wilson_ci_95(losses, num_games)

    mean_cost, cost_ci_half = _mean_ci_95([x["total_cost"] for x in games])
    mean_slips, slips_ci_half = _mean_ci_95([x["slips"] for x in games])

    print(f"  - Win Rate  : {wins}/{num_games} ({win_rate}%, Wilson 95% CI=[{win_ci[0]}%, {win_ci[1]}%])")
    print(f"  - Draw Rate : {draws}/{num_games} ({draw_rate}%, Wilson 95% CI=[{draw_ci[0]}%, {draw_ci[1]}%])")
    print(f"  - Loss Rate : {losses}/{num_games} ({loss_rate}%, Wilson 95% CI=[{loss_ci[0]}%, {loss_ci[1]}%])")
    print(f"  - Mean Cost : {mean_cost} +/- {cost_ci_half} (95% CI=[{round(mean_cost - cost_ci_half, 2)}, {round(mean_cost + cost_ci_half, 2)}])")
    print(f"  - Mean Slips: {mean_slips} +/- {slips_ci_half}")

    replan = run_replanning_cost_analysis(
        num_games=num_games, search_depth=depth_val, weights=best_weights,
    )
    print(f"\n4. REPLANNING COST (paired seeds: slip={replan['slip_prob']} vs slip=0.00, {replan['num_games']} games)")
    print("-" * 98)
    print(f"  - Games with at least one slip (replan triggered): {replan['games_with_slip']}/{replan['num_games']} ({replan['games_with_slip_pct']}%)")
    print(f"  - Mean replans per game: {replan['mean_replans']} | Mean slips per game: {replan['mean_slips']}")
    print(f"  - Extra cost from slips + replanning (paired): {replan['extra_cost_mean']} +/- {replan['extra_cost_ci95_half']} (95% CI)")
    print("=" * 98)

    return {
        "search_depth": depth_val,
        "pruning_benchmark": pruning_rows,
        "weight_tuning": tuning_table,
        "best_weights": best_weights,
        "deterministic_vs_stochastic": {"deterministic": det_res, "stochastic": sto_res},
        "replanning_cost": replan,
        "tournament": {
            "num_games": num_games,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate_pct": win_rate,
            "win_wilson_ci95_pct": list(win_ci),
            "draw_rate_pct": draw_rate,
            "draw_wilson_ci95_pct": list(draw_ci),
            "loss_rate_pct": loss_rate,
            "loss_wilson_ci95_pct": list(loss_ci),
            "mean_cost": mean_cost,
            "mean_cost_ci95_half": cost_ci_half,
            "mean_slips": mean_slips,
            "mean_slips_ci95_half": slips_ci_half,
            "games": games,
        },
    }

if __name__ == "__main__":
    run_part_c_experiments()    