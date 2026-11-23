import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from planners.state import custom_mst_heuristic
from planners.informed.a_star_search import a_star


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def race_split(env):
    """
    Split undelivered stops into winnable (agent reaches it no later than the rival) and contested.
    Uses the rival's position at planning time: a static snapshot, not a prediction of rival moves.
    """
    rem = [s for s in env.stops if s not in env.delivered and s not in env.stolen]
    winnable, contested = [], []
    for s in rem:
        d_agent = _manhattan(env.agent_pos, s)
        d_rival = _manhattan(env.rival_pos, s) + env.rival_cooldown
        (winnable if d_agent <= d_rival else contested).append(s)
    return winnable, contested


def plan_race_aware(env, heuristic_fn=custom_mst_heuristic, start_t=0):
    """
    Rival-aware plan: optimal A* over winnable stops only, plus RIVAL_STEAL_PENALTY per contested stop.
    The search and heuristic are unchanged; only the set of stops it must visit differs.
    """
    winnable, contested = race_split(env)
    penalty = float(config.RIVAL_STEAL_PENALTY) * len(contested)

    if not winnable:
        return {
            "algorithm": "A* (race-aware)",
            "status": "SUCCESS",
            "path": [env.agent_pos],
            "cost": round(penalty, 2),
            "steps": 0,
            "nodes_expanded": 0,
            "peak_memory_kb": 0.0,
            "runtime_ms": 0.0,
            "contested": contested,
        }

    orig_stops = env.stops
    env.stops = winnable
    try:
        res = a_star(env, heuristic_fn, start_t=start_t, label="A* (race-aware)", track_memory=False)
    finally:
        env.stops = orig_stops

    res["contested"] = contested
    if res["cost"] != float("inf"):
        res["cost"] = round(res["cost"] + penalty, 2)
    return res