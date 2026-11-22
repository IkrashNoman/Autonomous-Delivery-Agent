import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from world.road_closures import DIRECTIONS

DEFAULT_WEIGHTS = {
    "w1_progress": 8.0,
    "w2_rival_dist": 1.5,
    "w3_parcel_diff": 150.0,
    "w4_mobility": 1.2,
}


def _remaining_tour_estimate(agent_pos, remaining_stops):
    """Estimate remaining Manhattan tour length over {agent_pos} U remaining_stops."""
    if not remaining_stops:
        return 0.0

    nearest_dist = min(abs(agent_pos[0] - sr) + abs(agent_pos[1] - sc) for sr, sc in remaining_stops)
    if len(remaining_stops) == 1:
        return float(nearest_dist)

    n = len(remaining_stops)
    in_mst = [False] * n
    min_edge = [float("inf")] * n
    min_edge[0] = 0.0
    mst_sum = 0.0

    for _ in range(n):
        u = min((i for i in range(n) if not in_mst[i]), key=lambda i: min_edge[i])
        in_mst[u] = True
        mst_sum += min_edge[u]
        ur, uc = remaining_stops[u]
        for v in range(n):
            if not in_mst[v]:
                vr, vc = remaining_stops[v]
                d = abs(ur - vr) + abs(uc - vc)
                if d < min_edge[v]:
                    min_edge[v] = d

    return float(nearest_dist + 0.5 * mst_sum)


def extract_features(env, agent_pos, rival_pos, delivered, stolen, t, rival_cooldown):
    """Extract the 4 state features used by the adversarial evaluation function."""
    remaining = [s for s in env.stops if s not in delivered and s not in stolen]

    # Feature 1: Negative remaining tour estimate (drives decisive progress toward stops)
    f1_progress = -_remaining_tour_estimate(agent_pos, remaining)

    # Feature 2: Immediate collision buffer from rival (capped at 5 so agent doesn't flee endlessly)
    raw_dist = abs(agent_pos[0] - rival_pos[0]) + abs(agent_pos[1] - rival_pos[1])
    f2_rival_dist = float(min(raw_dist, 5)) if rival_cooldown == 0 else 5.0

    # Feature 3: Net parcel score (delivered minus stolen)
    f3_parcel_diff = float(len(delivered) - len(stolen))

    # Feature 4: Local escape mobility (valid unblocked exits at step t)
    exits = 0
    for dr, dc in DIRECTIONS:
        nxt = (agent_pos[0] + dr, agent_pos[1] + dc)
        if env.in_bounds_and_free(nxt) and env.closure_mgr.is_move_allowed(agent_pos, nxt, t):
            exits += 1
    f4_mobility = float(exits)

    return {
        "f1_progress": f1_progress,
        "f2_rival_dist": f2_rival_dist,
        "f3_parcel_diff": f3_parcel_diff,
        "f4_mobility": f4_mobility,
    }


def evaluate_game_state(env, state_tuple, weights=None):
    """
    Evaluate adversarial state tuple from the delivery agent's (MAX) perspective:
    state_tuple = (agent_pos, rival_pos, delivered_frozenset, stolen_frozenset, t, rival_cooldown)
    """
    w = DEFAULT_WEIGHTS if weights is None else weights
    agent_pos, rival_pos, delivered, stolen, t, rival_cooldown = state_tuple

    feats = extract_features(env, agent_pos, rival_pos, delivered, stolen, t, rival_cooldown)
    return float(
        w["w1_progress"] * feats["f1_progress"]
        + w["w2_rival_dist"] * feats["f2_rival_dist"]
        + w["w3_parcel_diff"] * feats["f3_parcel_diff"]
        + w["w4_mobility"] * feats["f4_mobility"]
    )


def tune_evaluation_weights(env, episode_runner):
    """
    Grid search over candidate weight vectors using independent seeded episodes.
    Returns (best_weights, summary_rows).
    """
    candidates = [
        {"w1_progress": 6.0, "w2_rival_dist": 1.0, "w3_parcel_diff": 120.0, "w4_mobility": 1.0},
        {"w1_progress": 8.0, "w2_rival_dist": 1.5, "w3_parcel_diff": 150.0, "w4_mobility": 1.2},
        {"w1_progress": 10.0, "w2_rival_dist": 2.5, "w3_parcel_diff": 180.0, "w4_mobility": 1.5},
        {"w1_progress": 7.0, "w2_rival_dist": 3.5, "w3_parcel_diff": 150.0, "w4_mobility": 1.0},
    ]

    results = []
    best_weights = dict(candidates[1])
    best_score = -float("inf")
    trials = 4

    for idx, w in enumerate(candidates):
        total_utility = 0.0
        wins = 0

        for g in range(trials):
            ep = episode_runner(
                env,
                seed=config.SEED + 500 + g,
                slip_prob=config.SLIP_PROBABILITY,
                weights=w,
                max_steps=130,
                search_depth=3,
            )
            if ep["outcome"] == "WIN":
                wins += 1
            total_utility += ep["delivered"] * 100.0 - ep["stolen"] * 100.0 - ep["total_cost"] * 0.25

        avg_utility = round(total_utility / trials, 2)
        results.append({
            "candidate_id": idx + 1,
            "weights": dict(w),
            "wins": wins,
            "trials": trials,
            "avg_utility": avg_utility,
        })
        if avg_utility > best_score:
            best_score = avg_utility
            best_weights = dict(w)

    return best_weights, results