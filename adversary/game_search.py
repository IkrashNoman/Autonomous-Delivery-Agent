import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from adversary.evaluation import evaluate_game_state, DEFAULT_WEIGHTS


def _in_local_window(pos, center, radius=config.LOCAL_GAME_WINDOW_RADIUS):
    """Restrict adversarial lookahead to a local Manhattan window around center."""
    return abs(pos[0] - center[0]) + abs(pos[1] - center[1]) <= radius


def _simulate_joint_ply(env, state_tuple, agent_next, rival_next):
    """Deterministic joint transition used inside Minimax and Alpha-Beta trees."""
    _, _, delivered, stolen, t, rival_cooldown = state_tuple
    new_delivered = set(delivered)
    new_stolen = set(stolen)

    if agent_next in env.stops and agent_next not in new_delivered and agent_next not in new_stolen:
        new_delivered.add(agent_next)

    new_cooldown = rival_cooldown
    if new_cooldown > 0:
        new_cooldown -= 1
    elif "rival_steals_parcels" in config.TWISTS:
        active_stops = [s for s in env.stops if s not in new_delivered and s not in new_stolen]
        if rival_next in active_stops:
            new_stolen.add(rival_next)
            new_cooldown = 3
        elif rival_next == agent_next and active_stops:
            nearest = min(active_stops, key=lambda s: abs(s[0] - agent_next[0]) + abs(s[1] - agent_next[1]))
            new_stolen.add(nearest)
            new_cooldown = 3

    return (agent_next, rival_next, frozenset(new_delivered), frozenset(new_stolen), t + 1, new_cooldown)


def _get_ordered_agent_moves(env, state_tuple, center, weights):
    """MAX move ordering: sort legal agent moves by static evaluation descending."""
    agent_pos, rival_pos, delivered, stolen, t, _ = state_tuple
    candidates = []
    for nxt, _ in env.get_valid_neighbors(agent_pos, t):
        if _in_local_window(nxt, center):
            preview = _simulate_joint_ply(env, state_tuple, nxt, rival_pos)
            candidates.append((evaluate_game_state(env, preview, weights), nxt))

    if not candidates:
        candidates.append((0.0, agent_pos))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [nxt for _, nxt in candidates]


def _get_ordered_rival_moves(env, state_tuple, center):
    """
    MIN move ordering: uses env.get_valid_rival_moves (which always includes rival_pos stay)
    within the rival's local radius and orders moves closing distance to agent/active stops first.
    """
    agent_pos, rival_pos, delivered, stolen, t, rival_cooldown = state_tuple
    if rival_cooldown > 0:
        return [rival_pos]

    active_stops = [s for s in env.stops if s not in delivered and s not in stolen]
    targets = [agent_pos] + active_stops

    candidates = []
    for nxt in env.get_valid_rival_moves(rival_pos, t):
        if _in_local_window(nxt, rival_pos, radius=config.LOCAL_GAME_WINDOW_RADIUS):
            min_target_dist = min(abs(nxt[0] - tr) + abs(nxt[1] - tc) for tr, tc in targets)
            candidates.append((min_target_dist, nxt))

    if not candidates:
        candidates.append((0, rival_pos))

    candidates.sort(key=lambda item: item[0])
    return [nxt for _, nxt in candidates]


def _apply_agent_half_ply(env, state_tuple, a_nxt):
    """Apply MAX's move and immediately credit parcel delivery if a_nxt is an active stop."""
    _, rival_pos, delivered, stolen, t, rival_cooldown = state_tuple
    new_delivered = set(delivered)
    if a_nxt in env.stops and a_nxt not in new_delivered and a_nxt not in stolen:
        new_delivered.add(a_nxt)
    return (a_nxt, rival_pos, frozenset(new_delivered), stolen, t + 1, rival_cooldown)


def minimax_search(env, state_tuple, depth, maximizing, center, weights, stats):
    """From-scratch Minimax search without pruning."""
    stats["nodes"] += 1
    agent_pos, rival_pos, delivered, stolen, t, _ = state_tuple
    if depth == 0 or len(delivered) + len(stolen) == len(env.stops):
        return evaluate_game_state(env, state_tuple, weights), agent_pos

    if maximizing:
        best_val = -float("inf")
        best_move = agent_pos
        for a_nxt in _get_ordered_agent_moves(env, state_tuple, center, weights):
            mid_state = _apply_agent_half_ply(env, state_tuple, a_nxt)
            val, _ = minimax_search(env, mid_state, depth - 1, False, center, weights, stats)
            if val > best_val:
                best_val = val
                best_move = a_nxt
        return best_val, best_move
    else:
        best_val = float("inf")
        best_move = rival_pos
        for r_nxt in _get_ordered_rival_moves(env, state_tuple, center):
            # Restore t before joint rival resolution so time increments once per full round
            pre_rival_state = (agent_pos, rival_pos, delivered, stolen, max(0, t - 1), state_tuple[5])
            next_state = _simulate_joint_ply(env, pre_rival_state, agent_pos, r_nxt)
            val, _ = minimax_search(env, next_state, depth - 1, True, center, weights, stats)
            if val < best_val:
                best_val = val
                best_move = r_nxt
        return best_val, best_move


def alphabeta_search(env, state_tuple, depth, alpha, beta, maximizing, center, weights, stats):
    """From-scratch Alpha-Beta Pruning search with domain move ordering."""
    stats["nodes"] += 1
    agent_pos, rival_pos, delivered, stolen, t, _ = state_tuple
    if depth == 0 or len(delivered) + len(stolen) == len(env.stops):
        return evaluate_game_state(env, state_tuple, weights), agent_pos

    if maximizing:
        best_val = -float("inf")
        best_move = agent_pos
        for a_nxt in _get_ordered_agent_moves(env, state_tuple, center, weights):
            mid_state = _apply_agent_half_ply(env, state_tuple, a_nxt)
            val, _ = alphabeta_search(env, mid_state, depth - 1, alpha, beta, False, center, weights, stats)
            if val > best_val:
                best_val = val
                best_move = a_nxt
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break
        return best_val, best_move
    else:
        best_val = float("inf")
        best_move = rival_pos
        for r_nxt in _get_ordered_rival_moves(env, state_tuple, center):
            pre_rival_state = (agent_pos, rival_pos, delivered, stolen, max(0, t - 1), state_tuple[5])
            next_state = _simulate_joint_ply(env, pre_rival_state, agent_pos, r_nxt)
            val, _ = alphabeta_search(env, next_state, depth - 1, alpha, beta, True, center, weights, stats)
            if val < best_val:
                best_val = val
                best_move = r_nxt
            beta = min(beta, best_val)
            if beta <= alpha:
                break
        return best_val, best_move

def choose_agent_and_rival_moves(env, depth=config.ADVERSARY_SEARCH_DEPTH, weights=None):
    """
    Select next grid coordinates for MAX (agent) and MIN (rival) via Alpha-Beta.
    Always returns a valid rival coordinate (returning env.rival_pos during cooldown).
    """
    w = DEFAULT_WEIGHTS if weights is None else weights
    state_tuple = (
        env.agent_pos,
        env.rival_pos,
        frozenset(env.delivered),
        frozenset(env.stolen),
        env.t,
        env.rival_cooldown,
    )
    center = env.agent_pos

    stats_max = {"nodes": 0}
    _, best_agent_pos = alphabeta_search(
        env, state_tuple, depth, -float("inf"), float("inf"), True, center, w, stats_max
    )

    if env.rival_cooldown > 0:
        best_rival_pos = env.rival_pos
    else:
        stats_min = {"nodes": 0}
        _, best_rival_pos = alphabeta_search(
            env, state_tuple, depth, -float("inf"), float("inf"), False, center, w, stats_min
        )

    return best_agent_pos, best_rival_pos


def compare_minimax_vs_alphabeta(env, weights=None):
    """Measure node expansions for depths 2 to ADVERSARY_SEARCH_DEPTH and pruning savings."""
    w = DEFAULT_WEIGHTS if weights is None else weights

    # Position rival on a valid traversable cell within tactical engagement distance (~4 steps)
    contested_rival = env.rival_pos
    for r in range(env.rows):
        for c in range(env.cols):
            if env.grid[r, c] >= 0 and (r, c) != env.agent_pos:
                if abs(r - env.agent_pos[0]) + abs(c - env.agent_pos[1]) == 4:
                    contested_rival = (r, c)
                    break
        if contested_rival != env.rival_pos:
            break

    state_tuple = (
        env.agent_pos,
        contested_rival,
        frozenset(env.delivered),
        frozenset(env.stolen),
        env.t,
        0,
    )
    center = env.agent_pos

    rows = []
    for d in range(2, config.ADVERSARY_SEARCH_DEPTH + 1):
        t0 = time.perf_counter()
        mm_stats = {"nodes": 0}
        minimax_search(env, state_tuple, d, True, center, w, mm_stats)
        mm_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        ab_stats = {"nodes": 0}
        alphabeta_search(env, state_tuple, d, -float("inf"), float("inf"), True, center, w, ab_stats)
        ab_ms = (time.perf_counter() - t1) * 1000

        saved_pct = ((mm_stats["nodes"] - ab_stats["nodes"]) / max(1, mm_stats["nodes"])) * 100.0
        rows.append({
            "depth": d,
            "minimax_nodes": mm_stats["nodes"],
            "alphabeta_nodes": ab_stats["nodes"],
            "saved_pct": round(saved_pct, 2),
            "minimax_ms": round(mm_ms, 2),
            "alphabeta_ms": round(ab_ms, 2),
        })
    return rows