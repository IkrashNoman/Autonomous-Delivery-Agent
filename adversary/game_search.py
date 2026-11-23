import os
import sys
import time
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from adversary.evaluation import evaluate_game_state


def _order_moves(moves, target_pos, maximizing=True):
    """Move ordering heuristic: sort candidate positions by Manhattan distance to target_pos."""
    return sorted(
        moves,
        key=lambda pos: abs(pos[0] - target_pos[0]) + abs(pos[1] - target_pos[1]),
        reverse=not maximizing,
    )


def minimax_search(env, agent_pos, rival_pos, delivered, stolen, t, rival_cooldown, depth, maximizing, stats, weights=None):
    """Unpruned depth-limited Minimax search over the local tactical window."""
    stats["nodes"] += 1
    rem_count = len(env.stops) - len(delivered) - len(stolen)
    if depth == 0 or rem_count == 0:
        state_tuple = (agent_pos, rival_pos, delivered, stolen, t, rival_cooldown)
        return evaluate_game_state(env, state_tuple, weights=weights), agent_pos if maximizing else rival_pos

    if maximizing:
        best_val = -math.inf
        best_move = agent_pos
        for nxt_a, _ in env.get_valid_neighbors(agent_pos, t):
            new_del = set(delivered)
            if nxt_a in env.stops and nxt_a not in stolen:
                new_del.add(nxt_a)
            val, _ = minimax_search(
                env, nxt_a, rival_pos, new_del, stolen, t, rival_cooldown, depth - 1, False, stats, weights
            )
            if val > best_val:
                best_val = val
                best_move = nxt_a
        return best_val, best_move
    else:
        best_val = math.inf
        best_move = rival_pos
        for nxt_r in env.get_valid_rival_moves(rival_pos, t):
            new_stolen = set(stolen)
            new_cd = max(0, rival_cooldown - 1)
            if rival_cooldown == 0:
                active = [s for s in env.stops if s not in delivered and s not in new_stolen]
                if nxt_r in active:
                    new_stolen.add(nxt_r)
                    new_cd = 3
            val, _ = minimax_search(
                env, agent_pos, nxt_r, delivered, new_stolen, t + 1, new_cd, depth - 1, True, stats, weights
            )
            if val < best_val:
                best_val = val
                best_move = nxt_r
        return best_val, best_move


def alphabeta_search(
    env, agent_pos, rival_pos, delivered, stolen, t, rival_cooldown, depth, alpha, beta, maximizing, stats, weights=None
):
    """Alpha-Beta pruned depth-limited game tree search with Manhattan move ordering."""
    stats["nodes"] += 1
    rem_count = len(env.stops) - len(delivered) - len(stolen)
    if depth == 0 or rem_count == 0:
        state_tuple = (agent_pos, rival_pos, delivered, stolen, t, rival_cooldown)
        return evaluate_game_state(env, state_tuple, weights=weights), agent_pos if maximizing else rival_pos

    rem_stops = [s for s in env.stops if s not in delivered and s not in stolen]

    if maximizing:
        best_val = -math.inf
        best_move = agent_pos
        target = min(rem_stops, key=lambda s: abs(s[0] - agent_pos[0]) + abs(s[1] - agent_pos[1])) if rem_stops else agent_pos
        raw_moves = [nxt for nxt, _ in env.get_valid_neighbors(agent_pos, t)]
        ordered_moves = _order_moves(raw_moves, target, maximizing=True)

        for nxt_a in ordered_moves:
            new_del = set(delivered)
            if nxt_a in env.stops and nxt_a not in stolen:
                new_del.add(nxt_a)
            val, _ = alphabeta_search(
                env, nxt_a, rival_pos, new_del, stolen, t, rival_cooldown, depth - 1, alpha, beta, False, stats, weights
            )
            if val > best_val:
                best_val = val
                best_move = nxt_a
            alpha = max(alpha, best_val)
            if beta <= alpha:
                stats["cutoffs"] += 1
                break
        return best_val, best_move
    else:
        best_val = math.inf
        best_move = rival_pos
        target = min(rem_stops, key=lambda s: abs(s[0] - rival_pos[0]) + abs(s[1] - rival_pos[1])) if rem_stops else agent_pos
        raw_moves = env.get_valid_rival_moves(rival_pos, t)
        ordered_moves = _order_moves(raw_moves, target, maximizing=False)

        for nxt_r in ordered_moves:
            new_stolen = set(stolen)
            new_cd = max(0, rival_cooldown - 1)
            if rival_cooldown == 0:
                active = [s for s in env.stops if s not in delivered and s not in new_stolen]
                if nxt_r in active:
                    new_stolen.add(nxt_r)
                    new_cd = 3
            val, _ = alphabeta_search(
                env, agent_pos, nxt_r, delivered, new_stolen, t + 1, new_cd, depth - 1, alpha, beta, True, stats, weights
            )
            if val < best_val:
                best_val = val
                best_move = nxt_r
            beta = min(beta, best_val)
            if beta <= alpha:
                stats["cutoffs"] += 1
                break
        return best_val, best_move


def choose_agent_and_rival_moves(env, depth=config.ADVERSARY_SEARCH_DEPTH, weights=None):
    """
    Compute best tactical move for Agent (MAX root) and Rival (MIN root) via Alpha-Beta pruning.
    Because GridWorldEnv.step() executes simultaneous moves within step t, the MIN root
    evaluates the rival's counter-response at step t with depth max(1, depth - 1) so both
    searches terminate on the same terminal ply.
    """
    stats_a = {"nodes": 0, "cutoffs": 0}
    _, best_a = alphabeta_search(
        env,
        env.agent_pos,
        env.rival_pos,
        set(env.delivered),
        set(env.stolen),
        env.t,
        env.rival_cooldown,
        depth,
        -math.inf,
        math.inf,
        True,
        stats_a,
        weights,
    )

    stats_r = {"nodes": 0, "cutoffs": 0}
    _, best_r = alphabeta_search(
        env,
        env.agent_pos,
        env.rival_pos,
        set(env.delivered),
        set(env.stolen),
        env.t,
        env.rival_cooldown,
        max(1, depth - 1),
        -math.inf,
        math.inf,
        False,
        stats_r,
        weights,
    )
    return best_a, best_r


def benchmark_adversarial_pruning(env=None, depths=None):
    """Benchmark unpruned Minimax vs Alpha-Beta pruning for depths 2 through ADVERSARY_SEARCH_DEPTH (5)."""
    from world.environment import GridWorldEnv

    if env is None:
        env = GridWorldEnv(seed=config.SEED)
    if depths is None:
        depths = tuple(range(2, int(config.ADVERSARY_SEARCH_DEPTH) + 1))

    print(config.STAMP)
    print("-" * 92)
    print(f"Part C Benchmark: Unpruned Minimax vs Alpha-Beta Pruning (Instance {config.INSTANCE_ID}, Max Depth={max(depths)})")
    print(f"{'Depth':<8} {'Minimax Nodes':>15} {'Alpha-Beta Nodes':>18} {'Reduction':>12} {'MM Time(ms)':>14} {'AB Time(ms)':>14}")
    print("-" * 92)

    rows = []
    for d in depths:
        s_mm = {"nodes": 0, "cutoffs": 0}
        t0 = time.perf_counter()
        minimax_search(
            env, env.agent_pos, env.rival_pos, set(env.delivered), set(env.stolen), env.t, env.rival_cooldown, d, True, s_mm
        )
        t_mm = (time.perf_counter() - t0) * 1000

        s_ab = {"nodes": 0, "cutoffs": 0}
        t1 = time.perf_counter()
        alphabeta_search(
            env,
            env.agent_pos,
            env.rival_pos,
            set(env.delivered),
            set(env.stolen),
            env.t,
            env.rival_cooldown,
            d,
            -math.inf,
            math.inf,
            True,
            s_ab,
        )
        t_ab = (time.perf_counter() - t1) * 1000

        red = (1.0 - s_ab["nodes"] / max(1, s_mm["nodes"])) * 100.0
        print(f"{d:<8} {s_mm['nodes']:>15} {s_ab['nodes']:>18} {red:>11.1f}% {t_mm:>14.2f} {t_ab:>14.2f}")
        rows.append({
            "depth": d,
            "minimax_nodes": s_mm["nodes"],
            "alphabeta_nodes": s_ab["nodes"],
            "reduction_pct": round(red, 2),
            "minimax_ms": round(t_mm, 2),
            "alphabeta_ms": round(t_ab, 2),
        })
    print("-" * 92)
    return rows