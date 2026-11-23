import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import copy
import signal
import threading
import mimetypes
import numpy as np
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote
from adversary.evaluation import DEFAULT_EVAL_WEIGHTS
from oel_instance import reachable
from planners.race_aware import plan_race_aware

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(WEB_DIR, "assests")
sys.path.append(ROOT_DIR)

import config
from world.environment import GridWorldEnv
from world.road_closures import DIR_NAMES, MAINTENANCE_CORRIDOR_MODES
from planners.state import baseline_nearest_manhattan, custom_mst_heuristic
from planners.uninformed.uniform_cost_search import ucs
from planners.informed.greedy_best_first import greedy_best_first
from planners.informed.a_star_search import a_star
from planners.informed.weighted_a_star import weighted_a_star
from adversary.game_search import choose_agent_and_rival_moves
from dispatcher.llm_dispatcher import LLMDispatcher


def _json_default(obj):
    """Convert NumPy arrays and scalar types into native Python JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def _safe_asset_path(rel):
    """Map an /assets/ request to a file inside ASSETS_DIR; return None if it escapes that folder."""
    base = os.path.realpath(ASSETS_DIR)
    candidate = os.path.realpath(os.path.join(base, unquote(rel)))
    try:
        if os.path.commonpath([base, candidate]) != base:
            return None
    except ValueError:  # different drive on Windows
        return None
    return candidate

def _load_tuned_weights():
    """Use the Part C tuned weights from results_summary.json when present; otherwise the defaults."""
    path = os.path.join(ROOT_DIR, "results_summary.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return dict(json.load(f)["part_c"]["best_weights"])
    except (OSError, KeyError, ValueError):
        return dict(DEFAULT_EVAL_WEIGHTS)
    
class SimulationSession:
    """Holds live simulation state, runtime config overrides, and active planner selection."""

    def __init__(self):
        default_eps = config.WEIGHTED_ASTAR_EPSILON
        rw_start, rw_end = int(config.RUSH_HOUR_WINDOW[0]), int(config.RUSH_HOUR_WINDOW[1])
        orig_mult = config.RUSH_HOUR_MULTIPLIER
        self._base_rush_mult = np.asarray(orig_mult, dtype=float).copy()
        road_mult = float(self._base_rush_mult[0])

        self._orig_config_snapshot = {
            "SEED": int(config.SEED),
            "SLIP_PROBABILITY": float(config.SLIP_PROBABILITY),
            "RUSH_HOUR_CYCLE": int(config.RUSH_HOUR_CYCLE),
            "RUSH_HOUR_WINDOW": copy.deepcopy(config.RUSH_HOUR_WINDOW),
            "RUSH_HOUR_MULTIPLIER": copy.deepcopy(orig_mult),
            "WEIGHTED_ASTAR_EPSILON": float(default_eps),
            "ADVERSARY_SEARCH_DEPTH": int(config.ADVERSARY_SEARCH_DEPTH),
            "LOCAL_GAME_WINDOW_RADIUS": int(config.LOCAL_GAME_WINDOW_RADIUS),
            "DELIVERY_STOPS": copy.deepcopy(config.DELIVERY_STOPS),
            "TWISTS": copy.deepcopy(config.TWISTS),
        }

        self._default_overrides = {
            "SEED": int(config.SEED),
            "SLIP_PROBABILITY": float(config.SLIP_PROBABILITY),
            "RUSH_HOUR_CYCLE": int(config.RUSH_HOUR_CYCLE),
            "RUSH_HOUR_START": rw_start,
            "RUSH_HOUR_END": rw_end,
            "RUSH_HOUR_MULTIPLIER": round(road_mult, 2),
            "WEIGHTED_ASTAR_EPSILON": float(default_eps),
            "ADVERSARY_SEARCH_DEPTH": int(config.ADVERSARY_SEARCH_DEPTH),
            "LOCAL_GAME_WINDOW_RADIUS": int(config.LOCAL_GAME_WINDOW_RADIUS),
            "TURN_TIME_LIMIT_SEC": 10.0,
            "STOPS": [[int(r), int(c)] for r, c in config.DELIVERY_STOPS],
            "TWIST_RIVAL": config.RIVAL_TWIST in config.TWISTS,
            "TWIST_RUSH": config.RUSH_TWIST in config.TWISTS,
            "TWIST_CLOSURE": config.THIRD_TWIST in config.TWISTS,
        }
        self.overrides = copy.deepcopy(self._default_overrides)
        self.best_weights = _load_tuned_weights()
        self.algorithm = "astar_custom"
        self.env = GridWorldEnv(seed=self.overrides["SEED"])
        self.dispatcher = LLMDispatcher(env=self.env, use_live_api=True, request_delay_sec=2.1)
        self.current_plan = None
        self.agent_trail = [self.env.agent_pos]
        self.rival_trail = [self.env.rival_pos]
        self.last_event_log = f"Initialized instance {config.INSTANCE_ID}."
        self.recompute_plan(bind_closures=True)

    def _twists_from_flags(self):
        """Build config.TWISTS from the three UI twist flags."""
        twists = []
        if self.overrides["TWIST_RIVAL"]:
            twists.append(config.RIVAL_TWIST)
        if self.overrides["TWIST_RUSH"]:
            twists.append(config.RUSH_TWIST)
        if self.overrides["TWIST_CLOSURE"]:
            twists.append(config.THIRD_TWIST)
        return twists

    def validate_stops(self, stops):
        """Raise ValueError if the requested stops cannot be used on the current grid."""
        expected = len(self._orig_config_snapshot["DELIVERY_STOPS"])
        if len(stops) != expected:
            raise ValueError(f"The instance has {expected} delivery stops; viva changes move stops, they do not add or remove them.")
        if len(set(stops)) != len(stops):
            raise ValueError("Delivery stops must be distinct.")
        start = tuple(config.START)
        for r, c in stops:
            if not (0 <= r < self.env.rows and 0 <= c < self.env.cols):
                raise ValueError(f"Stop {(r, c)} is outside the {self.env.rows}x{self.env.cols} grid.")
            if self.env.grid[r, c] < 0:
                raise ValueError(f"Stop {(r, c)} is on an obstacle.")
            if (r, c) == start:
                raise ValueError("A stop cannot be on the depot.")
        if not reachable(self.env.grid.tolist(), start, stops):
            raise ValueError("At least one stop cannot be reached from the depot.")

    def apply_overrides(self, new_vals):
        """
        Update overrides and config. Returns True when the world must be rebuilt
        (stop locations or twist flags changed).
        """
        rebuild = False
        for k, v in new_vals.items():
            if k == "STOPS":
                stops = [(int(s[0]), int(s[1])) for s in v]
                if stops != [tuple(s) for s in config.DELIVERY_STOPS]:
                    self.validate_stops(stops)
                    config.DELIVERY_STOPS = stops
                    self.overrides["STOPS"] = [[r, c] for r, c in stops]
                    rebuild = True
                continue

            if k in ("TWIST_RIVAL", "TWIST_RUSH", "TWIST_CLOSURE"):
                flag = bool(v)
                if self.overrides[k] != flag:
                    self.overrides[k] = flag
                    rebuild = True
                continue

            if k in self.overrides:
                if isinstance(self.overrides[k], int) and not isinstance(self.overrides[k], bool):
                    self.overrides[k] = int(v)
                else:
                    self.overrides[k] = float(v)

                if k == "RUSH_HOUR_MULTIPLIER" and hasattr(config, "RUSH_HOUR_MULTIPLIER"):
                    updated = self._base_rush_mult.copy()
                    updated[0] = float(self.overrides[k])
                    config.RUSH_HOUR_MULTIPLIER = updated
                elif k not in ("RUSH_HOUR_START", "RUSH_HOUR_END") and hasattr(config, k):
                    setattr(config, k, self.overrides[k])

        config.TWISTS = self._twists_from_flags()
        config.RUSH_HOUR_WINDOW = (int(self.overrides["RUSH_HOUR_START"]), int(self.overrides["RUSH_HOUR_END"]))
        self.env.slip_prob = float(self.overrides["SLIP_PROBABILITY"])
        return rebuild


    def restore_defaults(self):
        for k, orig_val in self._orig_config_snapshot.items():
            if hasattr(config, k):
                setattr(config, k, copy.deepcopy(orig_val))
        self.overrides = copy.deepcopy(self._default_overrides)
        self.reset()
        self.last_event_log = f"Restored default parameters for instance {config.INSTANCE_ID}."

    
    def reset(self, new_overrides=None):
        if new_overrides:
            self.apply_overrides(new_overrides)
        self.env = GridWorldEnv(seed=int(self.overrides["SEED"]))
        self.env.slip_prob = float(self.overrides["SLIP_PROBABILITY"])
        self.dispatcher = LLMDispatcher(env=self.env, use_live_api=True, request_delay_sec=2.1)
        self.agent_trail = [self.env.agent_pos]
        self.rival_trail = [self.env.rival_pos]
        self.last_event_log = f"Reset environment with seed={self.overrides['SEED']}."
        self.recompute_plan(bind_closures=True)



    def _run_selected_planner(self):
        """
        Run active planner from current (agent_pos, t, remaining stops) with:
          1. Strict closure & directional maintenance enforcement
          2. Lookahead penalty for closures scheduled within 5 steps (known schedule)
          3. Dynamic Rival Repulsion Field (scaled down 10x for smooth interactive paths)
          4. Expansion budget for UCS/Greedy with automatic A* fallback (UI safety)
        """
        orig_neighbors = self.env.get_valid_neighbors
        remaining_stops = {
            s for s in self.env.stops if s not in self.env.delivered and s not in self.env.stolen
        }
        rival_pos = self.env.rival_pos
        window_rad = int(self.overrides["LOCAL_GAME_WINDOW_RADIUS"])
        rival_dist_now = abs(self.env.agent_pos[0] - rival_pos[0]) + abs(self.env.agent_pos[1] - rival_pos[1])

        def constrained_neighbors(pos, t):
            res = []
            imminent_warning_cells = {
                ev["cell"]: ev["mode"]
                for ev in self.env.closure_mgr.events
                if 0 <= (ev["active_start"] - t) <= 5
            }

            for nxt, cost in orig_neighbors(pos, t):
                if nxt != pos and nxt in self.dispatcher.toolkit.avoid_cells:
                    continue

                adj_cost = cost * (
                    self.dispatcher.toolkit.rush_hour_boost if self.env.get_state()["rush_hour"] else 1.0
                )

                # 1. Lookahead penalty for closures scheduled within 5 steps (known schedule)
                if nxt != pos and nxt in imminent_warning_cells:
                    adj_cost += 4.0 if imminent_warning_cells[nxt] == "closed" else 2.2

                # 2. Rival evasion field
                if nxt != pos and nxt not in remaining_stops and rival_dist_now <= window_rad:
                    d_rival = abs(nxt[0] - rival_pos[0]) + abs(nxt[1] - rival_pos[1])
                    if d_rival == 0:
                        adj_cost += 8.0
                    elif d_rival == 1:
                        adj_cost += 4.5
                    elif d_rival == 2:
                        adj_cost += 2.4
                    elif d_rival == 3:
                        adj_cost += 1.0

                res.append((nxt, adj_cost))
            # No fallback wait: a trapped agent gets no successors, and the planner reports FAILED.
            return res

        def priority_custom_h(state, stops):
            base_h = custom_mst_heuristic(state, stops)
            p_stop = self.dispatcher.toolkit.priority_stop
            if p_stop is not None and p_stop in stops:
                idx = stops.index(p_stop)
                if not (state[2] & (1 << idx)):
                    dist_p = abs(state[0] - p_stop[0]) + abs(state[1] - p_stop[1])
                    # max() of admissible bounds stays admissible; 3 * dist is the true minimum cost to reach p
                    return max(base_h, dist_p * float(config.MIN_TERRAIN_COST))
            return base_h

        self.env.get_valid_neighbors = constrained_neighbors
        try:
            if self.algorithm == "ucs":
                res = ucs(self.env, start_t=self.env.t, max_expansions=40000)
            elif self.algorithm == "greedy":
                res = greedy_best_first(self.env, custom_mst_heuristic, start_t=self.env.t, max_expansions=40000)
            elif self.algorithm == "astar_manhattan":
                res = a_star(self.env, baseline_nearest_manhattan, start_t=self.env.t, track_memory=False)
            elif self.algorithm == "weighted_astar":
                eps = float(self.overrides["WEIGHTED_ASTAR_EPSILON"])
                res = weighted_a_star(self.env, custom_mst_heuristic, weight=eps, start_t=self.env.t)
            elif self.algorithm == "race_aware":
                res = plan_race_aware(self.env, priority_custom_h, start_t=self.env.t)
            else:
                res = a_star(self.env, priority_custom_h, start_t=self.env.t, track_memory=False)
        finally:
            self.env.get_valid_neighbors = orig_neighbors

        # Fallback to A* if UCS/Greedy hit budget or failed -> never leaves the agent stuck
        if res.get("status") in ("BUDGET_LIMIT", "FAILED") or not res.get("path"):
            fallback_label = f"{self.algorithm.upper()} (fallback A*)"
            self.env.get_valid_neighbors = constrained_neighbors
            try:
                res = a_star(
                    self.env, priority_custom_h, start_t=self.env.t,
                    track_memory=False, label=fallback_label,
                )
            finally:
                self.env.get_valid_neighbors = orig_neighbors

        return res
    
    def recompute_plan(self, bind_closures=False):
        plan = self._run_selected_planner()
        if bind_closures and plan.get("path"):
            self.env.closure_mgr.bind_route(plan["path"])
            plan = self._run_selected_planner()
        self.current_plan = plan
        return plan

    def step(self):
        """Advance simulation 1 step with live replanning around closures, maintenance, and rival."""
        if len(self.env.delivered) + len(self.env.stolen) == len(self.env.stops):
            self.last_event_log = "Episode complete: all parcels resolved."
            return

        # Recompute route at current t
        prev_path = list(self.current_plan.get("path", [])) if self.current_plan else []
        self.recompute_plan(bind_closures=False)
        path = self.current_plan.get("path", [])
        next_target = path[1] if len(path) >= 2 else self.env.agent_pos

        # Anti-oscillation guard
        if len(self.agent_trail) >= 2 and next_target == self.agent_trail[-2] and len(path) >= 3:
            alt_moves = [
                nxt
                for nxt, _ in self.env.get_valid_neighbors(self.env.agent_pos, self.env.t)
                if nxt != self.env.agent_pos and nxt != self.agent_trail[-2]
            ]
            if alt_moves:
                rem = [s for s in self.env.stops if s not in self.env.delivered and s not in self.env.stolen]
                if rem:
                    goal = min(rem, key=lambda s: abs(s[0] - self.env.agent_pos[0]) + abs(s[1] - self.env.agent_pos[1]))
                    next_target = min(alt_moves, key=lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1]))

        rival_dist = abs(self.env.agent_pos[0] - self.env.rival_pos[0]) + abs(
            self.env.agent_pos[1] - self.env.rival_pos[1]
        )
        rival_next = None
        if rival_dist <= int(self.overrides["LOCAL_GAME_WINDOW_RADIUS"]):
            tactical_a, rival_next = choose_agent_and_rival_moves(
                self.env,
                depth=int(self.overrides["ADVERSARY_SEARCH_DEPTH"]),
                weights=self.best_weights,
            )
            # Same engagement rule as adversary/match_simulator.run_episode
            if rival_dist <= 4 and tactical_a != self.env.agent_pos:
                if len(self.agent_trail) < 2 or tactical_a != self.agent_trail[-2]:
                    next_target = tactical_a

        prev_delivered = len(self.env.delivered)
        prev_stolen = len(self.env.stolen)

        prev_active_ids = {
            id(ev) for ev in self.env.closure_mgr.events
            if ev["active_start"] <= self.env.t < ev["active_end"]
        }

        action = (next_target[0] - self.env.agent_pos[0], next_target[1] - self.env.agent_pos[1])
        _, info = self.env.step(action, rival_next_pos=rival_next)

        self.agent_trail.append(self.env.agent_pos)
        self.rival_trail.append(self.env.rival_pos)

        new_active_ids = {
            id(ev) for ev in self.env.closure_mgr.events
            if ev["active_start"] <= self.env.t < ev["active_end"]
        }
        world_changed = (
            info.get("slipped", False)
            or info.get("stole_parcel", False)
            or len(self.env.delivered) > prev_delivered
            or len(self.env.stolen) > prev_stolen
            or new_active_ids != prev_active_ids
            or rival_dist <= int(self.overrides["LOCAL_GAME_WINDOW_RADIUS"])
        )

        if world_changed or len(prev_path) < 2:
            self.recompute_plan(bind_closures=False)
        else:
            self.current_plan = dict(self.current_plan)
            self.current_plan["path"] = prev_path[1:]
            self.current_plan["steps"] = max(0, len(prev_path) - 2)

        new_path = self.current_plan.get("path", [])
        new_cost = self.current_plan.get("cost", 0)

        active_events = [
            ev for ev in self.env.closure_mgr.events if ev["active_start"] <= self.env.t < ev["active_end"]
        ]
        rerouted = len(prev_path) >= 3 and len(new_path) >= 2 and prev_path[1:] != new_path

        events = [f"t={self.env.t}"]
        if info.get("slipped", False):
            events.append("SLIP! Rerouted")
        if rerouted and active_events:
            events.append(f"CLOSURE/MAINTENANCE REROUTE! New plan cost={new_cost}")
        elif rerouted and rival_dist <= int(self.overrides["LOCAL_GAME_WINDOW_RADIUS"]):
            events.append(f"RIVAL EVASION REROUTE (dist={rival_dist})! Plan cost={new_cost}")
        if len(self.env.delivered) > prev_delivered:
            events.append(f"DELIVERED parcel at {self.env.agent_pos}!")
        if len(self.env.stolen) > prev_stolen:
            events.append("RIVAL STOLE PARCEL!")
        if len(events) == 1:
            events.append(f"Moved to {self.env.agent_pos} | Plan cost={new_cost}")
        self.last_event_log = " | ".join(events)

    def serialize(self):
        state = self.env.get_state()
        closures = []
        for ev in self.env.closure_mgr.events:
            if ev["active_start"] <= self.env.t < ev["active_end"]:
                status = "ACTIVE"
                steps_until = 0
                steps_left = int(ev["active_end"] - self.env.t)
            elif self.env.t < ev["active_start"] <= self.env.t + 12:
                status = "UPCOMING"
                steps_until = int(ev["active_start"] - self.env.t)
                steps_left = int(ev["active_end"] - ev["active_start"])
            else:
                continue

            dir_labels = [DIR_NAMES[d] for d in ev.get("allowed_dirs", [])]
            closures.append({
                "cell": [int(ev["cell"][0]), int(ev["cell"][1])],
                "mode": str(ev["mode"]),
                "allowed_dirs": dir_labels,
                "active_start": int(ev["active_start"]),
                "active_end": int(ev["active_end"]),
                "status": status,
                "steps_until": steps_until,
                "steps_left": steps_left,
            })

        return {
            "stamp": config.STAMP,
            "instance_id": config.INSTANCE_ID,
            "rows": int(self.env.rows),
            "cols": int(self.env.cols),
            "grid": self.env.grid.tolist(),
            "start": [int(self.env.start[0]), int(self.env.start[1])],
            "stops": [[int(s[0]), int(s[1])] for s in self.env.stops],
            "delivered": [[int(s[0]), int(s[1])] for s in self.env.delivered],
            "stolen": [[int(s[0]), int(s[1])] for s in self.env.stolen],
            "agent_pos": [int(self.env.agent_pos[0]), int(self.env.agent_pos[1])],
            "rival_pos": [int(self.env.rival_pos[0]), int(self.env.rival_pos[1])],
            "rival_cooldown": int(self.env.rival_cooldown),
            "t": int(self.env.t),
            "rush_hour": bool(state["rush_hour"]),
            "total_cost": round(float(self.env.total_cost), 2),
            "slip_count": int(self.env.slip_count),
            "closures": closures,
            "avoid_cells": [[int(c[0]), int(c[1])] for c in self.dispatcher.toolkit.avoid_cells],
            "priority_stop": [int(self.dispatcher.toolkit.priority_stop[0]), int(self.dispatcher.toolkit.priority_stop[1])]
            if self.dispatcher.toolkit.priority_stop
            else None,
            "algorithm": self.algorithm,
            "plan": {
                "status": self.current_plan.get("status", "N/A"),
                "cost": float(self.current_plan.get("cost", 0)),
                "steps": int(self.current_plan.get("steps", 0)),
                "nodes_expanded": int(self.current_plan.get("nodes_expanded", 0)),
                "runtime_ms": float(self.current_plan.get("runtime_ms", 0)),
                "path": [[int(p[0]), int(p[1])] for p in self.current_plan.get("path", [])],
            },
            "agent_trail": [[int(p[0]), int(p[1])] for p in self.agent_trail[-40:]],
            "rival_trail": [[int(p[0]), int(p[1])] for p in self.rival_trail[-40:]],
            "last_event_log": self.last_event_log,
            "overrides": self.overrides,
        }


SESSION = SimulationSession()
SESSION_LOCK = threading.Lock()


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, payload, status=200):
        raw = json.dumps(payload, default=_json_default).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def _send_file(self, filepath):
        if not os.path.isfile(filepath):
            self.send_error(404, "File not found")
            return
        mime, _ = mimetypes.guess_type(filepath)
        with open(filepath, "rb") as f:
            content = f.read()
        try:
            self.send_response(200)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def do_GET(self):
        parsed = urlparse(self.path).path
        if parsed == "/api/state":
            with SESSION_LOCK:
                data = SESSION.serialize()
            self._send_json(data)
        elif parsed.startswith("/assets/"):
            rel = parsed[len("/assets/"):]
            safe_path = _safe_asset_path(rel)
            if safe_path is None:
                self.send_error(403, "Forbidden")
            else:
                self._send_file(safe_path)
        elif parsed == "/" or parsed == "/index.html":
            self._send_file(os.path.join(WEB_DIR, "index.html"))
        elif parsed == "/style.css":
            self._send_file(os.path.join(WEB_DIR, "style.css"))
        elif parsed == "/app.js":
            self._send_file(os.path.join(WEB_DIR, "app.js"))
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
        except json.JSONDecodeError:
            self.send_error(400, "Malformed JSON body")
            return

        if parsed == "/api/dispatch":
            self._handle_dispatch(body)
            return

        try:
            with SESSION_LOCK:
                if parsed == "/api/plan":
                    overrides = body.get("overrides", {})
                    if "algorithm" in body:
                        SESSION.algorithm = body["algorithm"]
                    seed_changed = "SEED" in overrides and int(overrides["SEED"]) != SESSION.overrides["SEED"]
                    if seed_changed:
                        SESSION.reset(overrides)
                        SESSION.last_event_log = (
                            f"Seed changed to {SESSION.overrides['SEED']} — rebuilt environment & closures."
                        )
                    else:
                        rebuild = SESSION.apply_overrides(overrides) if overrides else False
                        if rebuild:
                            SESSION.reset()
                            SESSION.last_event_log = "Stops/twists applied: world rebuilt at t=0."
                        else:
                            SESSION.recompute_plan(bind_closures=False)
                            SESSION.last_event_log = (
                                f"Switched/updated planner [{SESSION.algorithm}] -> cost={SESSION.current_plan['cost']}, "
                                f"nodes={SESSION.current_plan['nodes_expanded']}."
                            )
                    self._send_json(SESSION.serialize())

                elif parsed == "/api/step":
                    rebuild = SESSION.apply_overrides(body["overrides"]) if "overrides" in body else False
                    if rebuild:
                        SESSION.reset()
                        SESSION.last_event_log = "Stops/twists applied: world rebuilt at t=0."
                    else:
                        SESSION.step()
                    self._send_json(SESSION.serialize())

                elif parsed == "/api/reset":
                    SESSION.reset(body.get("overrides"))
                    self._send_json(SESSION.serialize())

                elif parsed == "/api/restore_defaults":
                    SESSION.restore_defaults()
                    self._send_json(SESSION.serialize())

                else:
                    self.send_error(404, "Not found")
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _handle_dispatch(self, body):
        """Run the ReAct dispatcher on a snapshot so the Groq calls do not block the UI."""
        instruction = str(body.get("instruction", "")).strip()

        # 1. Snapshot the world and constraints under the lock (fast).
        with SESSION_LOCK:
            env_copy = copy.deepcopy(SESSION.env)
            env_copy.__dict__.pop("get_valid_neighbors", None)  # drop any stale bound-method override
            worker = LLMDispatcher(env=env_copy, use_live_api=True, request_delay_sec=2.1)
            worker.toolkit.avoid_cells = set(SESSION.dispatcher.toolkit.avoid_cells)
            worker.toolkit.priority_stop = SESSION.dispatcher.toolkit.priority_stop
            worker.toolkit.rush_hour_boost = SESSION.dispatcher.toolkit.rush_hour_boost

        # 2. Slow Groq / ReAct loop runs WITHOUT the lock.
        disp_res = worker.dispatch(instruction, reset_before=False)

        # 3. Apply the resulting constraints back to the live session (fast).
        with SESSION_LOCK:
            SESSION.dispatcher.toolkit.avoid_cells = set(worker.toolkit.avoid_cells)
            SESSION.dispatcher.toolkit.priority_stop = worker.toolkit.priority_stop
            SESSION.dispatcher.toolkit.rush_hour_boost = worker.toolkit.rush_hour_boost
            SESSION.recompute_plan(bind_closures=False)
            SESSION.last_event_log = f"Dispatcher [{disp_res['status']}]: {instruction}"
            payload = SESSION.serialize()

        payload["dispatch_result"] = disp_res
        self._send_json(payload)


def run_server(port=8000):
    print(config.STAMP)
    print(f"Starting Autonomous GridWorld Web Dashboard at http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.")
    server = ThreadingHTTPServer(
        ("127.0.0.1", port),
        DashboardHandler
    )

    # Allow the server to stop without waiting indefinitely
    server.daemon_threads = True
    server.block_on_close = False

    stop_event = threading.Event()

    def handle_shutdown(signum, frame):
        if not stop_event.is_set():
            print("\nCtrl+C detected! Shutting down server...")
            stop_event.set()

            # shutdown() must run from a thread other than
            # the one executing serve_forever().
            threading.Thread(
                target=server.shutdown,
                daemon=True
            ).start()

    # Register Ctrl+C handler
    signal.signal(signal.SIGINT, handle_shutdown)

    try:
        server.serve_forever(poll_interval=0.2)

    except KeyboardInterrupt:
        handle_shutdown(signal.SIGINT, None)

    finally:
        print("\nClosing server...")
        server.server_close()
        print("Server stopped successfully.")


if __name__ == "__main__":
    run_server(8000)