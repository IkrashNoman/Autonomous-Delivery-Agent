import os
import sys
import json
import copy
import threading
import mimetypes
import numpy as np
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

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


class SimulationSession:
    """Holds live simulation state, runtime config overrides, and active planner selection."""

    def __init__(self):
        default_eps = getattr(
            config,
            "WEIGHTED_ASTAR_EPSILON",
            getattr(config, "INADMISSIBLE_HEURISTIC_WEIGHT", 1.8),
        )
        rw = getattr(config, "RUSH_HOUR_WINDOW", (6, 12))
        rw_start = int(rw[0]) if isinstance(rw, (tuple, list, np.ndarray)) else 6
        rw_end = int(rw[1]) if isinstance(rw, (tuple, list, np.ndarray)) else 12

        orig_mult = getattr(config, "RUSH_HOUR_MULTIPLIER", np.array([2.0, 1.0, 1.25, 1.0]))
        if isinstance(orig_mult, np.ndarray):
            self._base_rush_mult = orig_mult.astype(float).copy()
            road_mult = float(self._base_rush_mult[0])
        elif isinstance(orig_mult, (list, tuple)):
            self._base_rush_mult = np.asarray(orig_mult, dtype=float)
            road_mult = float(self._base_rush_mult[0])
        else:
            self._base_rush_mult = float(orig_mult)
            road_mult = float(orig_mult)

        self._orig_config_snapshot = {
            "SEED": int(config.SEED),
            "SLIP_PROBABILITY": float(config.SLIP_PROBABILITY),
            "RUSH_HOUR_CYCLE": int(getattr(config, "RUSH_HOUR_CYCLE", 20)),
            "RUSH_HOUR_WINDOW": copy.deepcopy(getattr(config, "RUSH_HOUR_WINDOW", (6, 12))),
            "RUSH_HOUR_MULTIPLIER": copy.deepcopy(orig_mult),
            "WEIGHTED_ASTAR_EPSILON": float(default_eps),
            "ADVERSARY_SEARCH_DEPTH": int(config.ADVERSARY_SEARCH_DEPTH),
            "LOCAL_GAME_WINDOW_RADIUS": int(config.LOCAL_GAME_WINDOW_RADIUS),
        }

        self._default_overrides = {
            "SEED": int(config.SEED),
            "SLIP_PROBABILITY": float(config.SLIP_PROBABILITY),
            "RUSH_HOUR_CYCLE": int(getattr(config, "RUSH_HOUR_CYCLE", 20)),
            "RUSH_HOUR_START": rw_start,
            "RUSH_HOUR_END": rw_end,
            "RUSH_HOUR_MULTIPLIER": round(road_mult, 2),
            "WEIGHTED_ASTAR_EPSILON": float(default_eps),
            "ADVERSARY_SEARCH_DEPTH": int(config.ADVERSARY_SEARCH_DEPTH),
            "LOCAL_GAME_WINDOW_RADIUS": int(config.LOCAL_GAME_WINDOW_RADIUS),
            "TURN_TIME_LIMIT_SEC": 10.0,
        }
        self.overrides = copy.deepcopy(self._default_overrides)
        self.algorithm = "astar_custom"
        self.env = GridWorldEnv(seed=self.overrides["SEED"])
        self.dispatcher = LLMDispatcher(env=self.env, use_live_api=True, request_delay_sec=2.1)
        self.current_plan = None
        self.agent_trail = [self.env.agent_pos]
        self.rival_trail = [self.env.rival_pos]
        self.last_event_log = f"Initialized instance {config.INSTANCE_ID}."
        self.recompute_plan(bind_closures=True)

    def apply_overrides(self, new_vals):
        """Update config constants cleanly without collapsing per-terrain rush-hour multipliers."""
        for k, v in new_vals.items():
            if k in self.overrides:
                if isinstance(self.overrides[k], int) and not isinstance(self.overrides[k], bool):
                    self.overrides[k] = int(v)
                else:
                    self.overrides[k] = float(v)

                if k == "RUSH_HOUR_MULTIPLIER" and hasattr(config, "RUSH_HOUR_MULTIPLIER"):
                    if isinstance(self._base_rush_mult, np.ndarray):
                        updated = self._base_rush_mult.copy()
                        updated[0] = float(self.overrides[k])
                        config.RUSH_HOUR_MULTIPLIER = updated
                    else:
                        config.RUSH_HOUR_MULTIPLIER = float(self.overrides[k])
                elif k not in ("RUSH_HOUR_START", "RUSH_HOUR_END") and hasattr(config, k):
                    setattr(config, k, self.overrides[k])

        if hasattr(config, "RUSH_HOUR_WINDOW"):
            config.RUSH_HOUR_WINDOW = (
                int(self.overrides["RUSH_HOUR_START"]),
                int(self.overrides["RUSH_HOUR_END"]),
            )
        self.env.slip_prob = float(self.overrides["SLIP_PROBABILITY"])

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

    def _schedule_visible_corridor_closures(self, base_path):
        """
        Ensure dynamic road closures & directional maintenance actually occur along the
        planned corridor early in the episode (t=3..18) so live rerouting is guaranteed.
        """
        self.env.closure_mgr.bind_route(base_path)
        protected = {self.env.start, *self.env.stops}
        corridor_cells = [c for c in base_path[3:22] if c not in protected]

        # Ensure at least 3 early on-route events (mix of full closure and 1-way/2-way maintenance)
        if len(corridor_cells) >= 3:
            sample_indices = [
                max(0, len(corridor_cells) // 5),
                max(1, len(corridor_cells) // 2),
                max(2, (4 * len(corridor_cells)) // 5),
            ]
            for idx_num, path_idx in enumerate(sample_indices):
                cell = corridor_cells[path_idx]
                # Replace or insert early event on this corridor cell
                self.env.closure_mgr.events = [
                    ev for ev in self.env.closure_mgr.events if ev["cell"] != cell
                ]
                act_start = 3 + idx_num * 4  # t=3, t=7, t=11
                mode = "closed" if idx_num % 2 == 0 else "directional"
                allowed_dirs = (
                    []
                    if mode == "closed"
                    else list(MAINTENANCE_CORRIDOR_MODES[(self.overrides["SEED"] + idx_num) % len(MAINTENANCE_CORRIDOR_MODES)])
                )
                self.env.closure_mgr.events.append({
                    "event_id": 100 + idx_num,
                    "cell": cell,
                    "zone": "on_route",
                    "mode": mode,
                    "allowed_dirs": allowed_dirs,
                    "warn_start": max(0, act_start - config.CLOSURE_WARNING_LEAD),
                    "active_start": act_start,
                    "active_end": act_start + config.CLOSURE_DURATION + 3,
                })
            self.env.closure_mgr.events.sort(key=lambda e: e["active_start"])

    def _run_selected_planner(self):
        """
        Run active planner from current (agent_pos, t, remaining stops) with:
          1. Strict closure & directional maintenance enforcement
          2. Predictive warning-zone detour penalty so the agent reroutes BEFORE a cell closes
          3. Dynamic Rival Repulsion Field when rival is within LOCAL_GAME_WINDOW_RADIUS
             (causes A* to curve smoothly away from the rival without up-down ping-ponging)
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
            # Cells entering closure or maintenance within the next 5 steps
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

                # 1. Predictive warning detour: reroute BEFORE a road closes or enters maintenance
                if nxt != pos and nxt in imminent_warning_cells:
                    adj_cost += 40.0 if imminent_warning_cells[nxt] == "closed" else 22.0

                # 2. Smooth Rival Evasion Field: when rival is close, penalize nearby cells so A* curves away
                if nxt != pos and nxt not in remaining_stops and rival_dist_now <= window_rad:
                    d_rival = abs(nxt[0] - rival_pos[0]) + abs(nxt[1] - rival_pos[1])
                    if d_rival == 0:
                        adj_cost += 80.0
                    elif d_rival == 1:
                        adj_cost += 45.0
                    elif d_rival == 2:
                        adj_cost += 24.0
                    elif d_rival == 3:
                        adj_cost += 10.0

                res.append((nxt, adj_cost))
            return res if res else [(pos, float(config.MIN_TERRAIN_COST))]

        def priority_custom_h(state, stops):
            base_h = custom_mst_heuristic(state, stops)
            p_stop = self.dispatcher.toolkit.priority_stop
            if p_stop is not None and p_stop in stops:
                idx = stops.index(p_stop)
                if not (state[2] & (1 << idx)):
                    dist_p = abs(state[0] - p_stop[0]) + abs(state[1] - p_stop[1])
                    return base_h + dist_p * 1.5
            return base_h

        self.env.get_valid_neighbors = constrained_neighbors
        try:
            if self.algorithm == "ucs":
                res = ucs(self.env, start_t=self.env.t)
            elif self.algorithm == "greedy":
                res = greedy_best_first(self.env, custom_mst_heuristic, start_t=self.env.t)
            elif self.algorithm == "astar_manhattan":
                res = a_star(self.env, baseline_nearest_manhattan, start_t=self.env.t, track_memory=False)
            elif self.algorithm == "weighted_astar":
                eps = float(self.overrides["WEIGHTED_ASTAR_EPSILON"])
                res = weighted_a_star(self.env, custom_mst_heuristic, weight=eps, start_t=self.env.t)
            else:
                res = a_star(self.env, priority_custom_h, start_t=self.env.t, track_memory=False)
        finally:
            self.env.get_valid_neighbors = orig_neighbors

        return res

    def recompute_plan(self, bind_closures=False):
        plan = self._run_selected_planner()
        if bind_closures and plan.get("path"):
            self._schedule_visible_corridor_closures(plan["path"])
            plan = self._run_selected_planner()
        self.current_plan = plan
        return plan

    def step(self):
        """Advance simulation 1 step with live replanning around closures, maintenance, and rival."""
        if len(self.env.delivered) + len(self.env.stolen) == len(self.env.stops):
            self.last_event_log = "Episode complete: all parcels resolved."
            return

        # Recompute route at current t (respecting active closures, imminent warnings, and rival proximity)
        prev_path = list(self.current_plan.get("path", [])) if self.current_plan else []
        self.recompute_plan(bind_closures=False)
        path = self.current_plan.get("path", [])
        next_target = path[1] if len(path) >= 2 else self.env.agent_pos

        # Anti-oscillation guard: never step back into immediately previous cell if forward progress exists
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
        if rival_dist <= int(self.overrides["LOCAL_GAME_WINDOW_RADIUS"]):
            _, rival_next = choose_agent_and_rival_moves(
                self.env, depth=int(self.overrides["ADVERSARY_SEARCH_DEPTH"])
            )
        else:
            rival_next = None

        prev_delivered = len(self.env.delivered)
        prev_stolen = len(self.env.stolen)

        action = (next_target[0] - self.env.agent_pos[0], next_target[1] - self.env.agent_pos[1])
        _, info = self.env.step(action, rival_next_pos=rival_next)

        self.agent_trail.append(self.env.agent_pos)
        self.rival_trail.append(self.env.rival_pos)

        # Recompute route for new step t so UI shows updated detour immediately
        self.recompute_plan(bind_closures=False)
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
            self._send_file(os.path.join(ASSETS_DIR, rel))
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
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

        with SESSION_LOCK:
            if parsed == "/api/plan":
                overrides = body.get("overrides", {})
                if "algorithm" in body:
                    SESSION.algorithm = body["algorithm"]
                if "SEED" in overrides and int(overrides["SEED"]) != SESSION.overrides["SEED"]:
                    SESSION.reset(overrides)
                    SESSION.last_event_log = (
                        f"Seed changed to {SESSION.overrides['SEED']} — rebuilt environment & closures."
                    )
                else:
                    if overrides:
                        SESSION.apply_overrides(overrides)
                    SESSION.recompute_plan(bind_closures=False)
                    SESSION.last_event_log = (
                        f"Switched/updated planner [{SESSION.algorithm}] -> cost={SESSION.current_plan['cost']}, "
                        f"nodes={SESSION.current_plan['nodes_expanded']}."
                    )
                self._send_json(SESSION.serialize())

            elif parsed == "/api/step":
                if "overrides" in body:
                    SESSION.apply_overrides(body["overrides"])
                SESSION.step()
                self._send_json(SESSION.serialize())

            elif parsed == "/api/reset":
                SESSION.reset(body.get("overrides"))
                self._send_json(SESSION.serialize())

            elif parsed == "/api/restore_defaults":
                SESSION.restore_defaults()
                self._send_json(SESSION.serialize())

            elif parsed == "/api/dispatch":
                instruction = str(body.get("instruction", "")).strip()
                disp_res = SESSION.dispatcher.dispatch(instruction, reset_before=False)
                SESSION.recompute_plan(bind_closures=False)
                SESSION.last_event_log = f"Dispatcher [{disp_res['status']}]: {instruction}"
                payload = SESSION.serialize()
                payload["dispatch_result"] = disp_res
                self._send_json(payload)
            else:
                self.send_error(404, "Not found")


def run_server(port=8000):
    print(config.STAMP)
    print(f"Starting Autonomous GridWorld Web Dashboard at http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.")
    server = ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    run_server(8000)