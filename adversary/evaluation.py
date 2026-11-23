import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


DEFAULT_EVAL_WEIGHTS = {
    "delivered": 140.0,
    "stolen": -160.0,
    "agent_stop_dist": -3.5,
    "rival_stop_dist": 4.5,
    "cost_penalty": -0.25,
}


def _remaining_tour_estimate(agent_pos, remaining_stops):
    """Greedy nearest-neighbor Manhattan lower bound across remaining stops."""
    if not remaining_stops:
        return 0.0
    curr = agent_pos
    rem = list(remaining_stops)
    total = 0.0
    while rem:
        nxt = min(rem, key=lambda s: abs(curr[0] - s[0]) + abs(curr[1] - s[1]))
        total += abs(curr[0] - nxt[0]) + abs(curr[1] - nxt[1])
        curr = nxt
        rem.remove(nxt)
    return float(total)


def extract_features(env, agent_pos, rival_pos, delivered, stolen, t, rival_cooldown):
    """Extract 5 interpretable domain features from an adversarial game state."""
    rem_stops = [s for s in env.stops if s not in delivered and s not in stolen]
    agent_tour = _remaining_tour_estimate(agent_pos, rem_stops)

    if rem_stops:
        rival_min_dist = min(abs(rival_pos[0] - s[0]) + abs(rival_pos[1] - s[1]) for s in rem_stops)
    else:
        rival_min_dist = 0.0

    effective_rival_dist = float(rival_min_dist + rival_cooldown)

    return {
        "delivered": float(len(delivered)),
        "stolen": float(len(stolen)),
        "agent_stop_dist": agent_tour,
        "rival_stop_dist": effective_rival_dist,
        "cost_penalty": float(t * config.MIN_TERRAIN_COST),
    }


def evaluate_game_state(env, state_tuple, weights=None):
    """Linear heuristic evaluation function V(s) = w^T * phi(s) for Alpha-Beta leaf nodes."""
    w = DEFAULT_EVAL_WEIGHTS if weights is None else weights
    agent_pos, rival_pos, delivered, stolen, t, rival_cooldown = state_tuple
    feats = extract_features(env, agent_pos, rival_pos, delivered, stolen, t, rival_cooldown)
    return float(sum(w.get(k, 0.0) * feats[k] for k in feats))


def build_weight_grid():
    """Grid over four evaluation weights (cost_penalty fixed at -0.25): 2 x 2 x 2 x 2 = 16 candidates."""
    grid = []
    for delivered in (120.0, 180.0):
        for stolen in (-120.0, -200.0):
            for agent_stop in (-2.0, -5.0):
                for rival_stop in (2.0, 6.0):
                    grid.append({
                        "delivered": delivered,
                        "stolen": stolen,
                        "agent_stop_dist": agent_stop,
                        "rival_stop_dist": rival_stop,
                        "cost_penalty": -0.25,
                    })
    return grid


def tune_evaluation_weights(env, episode_runner):
    """
    Grid search over build_weight_grid() at the target adversarial depth.
    6 seeded trials per candidate on tuning seeds (SEED + 1000 + trial), disjoint from the
    tournament and Part E seeds, so the tuned weights are not tested on the games they were tuned on.
    Ranked by (wins * 2 + draws), then average utility.
    """
    candidates = build_weight_grid()
    trials_per_candidate = 6
    tuning_table = []
    best_weights = candidates[0]
    best_score = (-1, -float("inf"))

    for idx, cand in enumerate(candidates, start=1):
        wins = 0
        draws = 0
        utilities = []
        for tr in range(trials_per_candidate):
            ep = episode_runner(env, seed=config.SEED + 1000 + tr, weights=cand)
            if ep["outcome"] == "WIN":
                wins += 1
            elif ep["outcome"] == "DRAW":
                draws += 1
            util = (ep["delivered"] * 120.0) - (ep["stolen"] * 150.0) - (0.2 * ep["total_cost"])
            utilities.append(util)

        avg_util = round(float(np.mean(utilities)), 2)
        row = {
            "candidate_id": idx,
            "weights": cand,
            "wins": wins,
            "draws": draws,
            "trials": trials_per_candidate,
            "avg_utility": avg_util,
        }
        tuning_table.append(row)

        score_key = (wins * 2 + draws, avg_util)
        if score_key > best_score:
            best_score = score_key
            best_weights = cand

    return best_weights, tuning_table


