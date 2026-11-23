import os
import sys
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from planners.state import custom_mst_heuristic
from planners.informed.a_star_search import a_star
from adversary.match_simulator import run_episode
from dispatcher.schema import validate_dispatch_command, validate_simulation_args, validate_inspect_args

class DispatcherToolkit:
    """
    Tool suite exposed to the ReAct Dispatcher Agent:
      1. plan_route       - Time-aware A* route planner with priority & avoidance constraints
      2. simulate         - Non-destructive stochastic + adversarial episode simulator
      3. inspect_closures - Custom twist tool inspecting dynamic road closures & rush-hour phase
      4. add_avoid_zone   - Guardrailed hazard avoidance registration
    """

    def __init__(self, env, enforce_guardrails=True):
        self.env = env
        self.enforce_guardrails = enforce_guardrails
        self.priority_stop = None
        self.avoid_cells = set()
        self.rush_hour_boost = 1.0
        self.last_plan = None

    def reset_constraints(self):
        self.priority_stop = None
        self.avoid_cells.clear()
        self.rush_hour_boost = 1.0
        self.last_plan = None

    def inspect_closures(self, t=None):
        """Custom Twist Tool: inspect active/upcoming road closures, rush-hour status, and rival distance."""
        if self.enforce_guardrails:
            t = validate_inspect_args(t)
        query_t = self.env.t if t is None else int(t)
        active = []
        upcoming = []
        for ev in self.env.closure_mgr.events:
            if ev["active_start"] <= query_t < ev["active_end"]:
                active.append({"cell": ev["cell"], "mode": ev["mode"], "ends_at": ev["active_end"]})
            elif query_t < ev["active_start"] <= query_t + 10:
                upcoming.append({"cell": ev["cell"], "mode": ev["mode"], "starts_at": ev["active_start"]})

        state = self.env.get_state()
        rival_dist = abs(self.env.agent_pos[0] - self.env.rival_pos[0]) + abs(
            self.env.agent_pos[1] - self.env.rival_pos[1]
        )
        return {
            "t": query_t,
            "rush_hour": state["rush_hour"],
            "active_closures": active,
            "upcoming_closures": upcoming,
            "rival_pos": self.env.rival_pos,
            "rival_dist": rival_dist,
            "remaining_stops": [s for s in self.env.stops if s not in self.env.delivered and s not in self.env.stolen],
        }
    
    def add_avoid_zone(self, row=None, col=None, radius=0):
        """Validate and register an avoidance zone via schema guardrails."""
        if self.enforce_guardrails:
            cmd = validate_dispatch_command(
                {"action": "avoid_zone", "target": [row, col], "radius": int(radius)},
                self.env,
            )
            self.avoid_cells.update(cmd["blocked_cells"])
            return {"added_cells": cmd["blocked_cells"], "total_avoid_cells": len(self.avoid_cells)}

        # Unvalidated ablation mode: bypasses schema checks (used only in Section 3 ablation)
        if row is None or col is None:
            raise ValueError("Unvalidated execution crashed on missing coordinates (None, None)")
        r, c, rad = int(row), int(col), int(radius)
        added = []
        for rr in range(r - rad, r + rad + 1):
            for cc in range(c - rad, c + rad + 1):
                if abs(rr - r) + abs(cc - c) <= rad:
                    self.avoid_cells.add((rr, cc))
                    added.append((rr, cc))
        return {"added_cells": added, "total_avoid_cells": len(self.avoid_cells)}

    def plan_route(self, priority_stop=None, avoid_rush_hour=False):
        """Compute time-aware A* route under active dispatch constraints."""
        if priority_stop is not None:
            if self.enforce_guardrails:
                cmd = validate_dispatch_command(
                    {"action": "prioritize_stop", "target": list(priority_stop)},
                    self.env,
                )
                self.priority_stop = cmd["target"]
            else:
                self.priority_stop = (int(priority_stop[0]), int(priority_stop[1]))

        if avoid_rush_hour:
            self.rush_hour_boost = 1.5

        orig_neighbors = self.env.get_valid_neighbors

        def constrained_neighbors(pos, t):
            res = []
            for nxt, cost in orig_neighbors(pos, t):
                if nxt != pos and nxt in self.avoid_cells:
                    continue
                adj_cost = cost * (self.rush_hour_boost if self.env.get_state()["rush_hour"] else 1.0)
                res.append((nxt, adj_cost))
            # No fallback wait: a trapped agent gets no successors, and the planner reports FAILED.
            return res

        def priority_heuristic(state, stops):
            base_h = custom_mst_heuristic(state, stops)
            if self.priority_stop is not None and self.priority_stop in stops:
                idx = stops.index(self.priority_stop)
                if not (state[2] & (1 << idx)):
                    dist_p = abs(state[0] - self.priority_stop[0]) + abs(state[1] - self.priority_stop[1])
                    return max(base_h, dist_p * float(config.MIN_TERRAIN_COST))
            return base_h

        self.env.get_valid_neighbors = constrained_neighbors
        try:
            plan = a_star(self.env, priority_heuristic, start_t=self.env.t, track_memory=False)
        finally:
            self.env.get_valid_neighbors = orig_neighbors

        self.last_plan = plan
        return {
            "status": plan["status"],
            "cost": plan["cost"],
            "steps": plan["steps"],
            "nodes_expanded": plan["nodes_expanded"],
            "priority_stop": self.priority_stop,
            "avoid_count": len(self.avoid_cells),
        }

    def simulate(self, seed=config.SEED, slip_prob=config.SLIP_PROBABILITY, max_steps=120):
        """
        Non-destructive rollout simulator: validates arguments (guardrails on), snapshots the
        env, runs the episode, and restores exact state afterward (including slip_prob and planned_route).
        """
        if self.enforce_guardrails:
            seed, slip_prob, max_steps = validate_simulation_args(
                config.SEED if seed is None else seed,
                config.SLIP_PROBABILITY if slip_prob is None else slip_prob,
                120 if max_steps is None else max_steps,
            )

        snapshot = {
            "seed": self.env.seed,
            "rng_state": copy.deepcopy(self.env.rng.bit_generator.state),
            "agent_pos": self.env.agent_pos,
            "rival_pos": self.env.rival_pos,
            "delivered": set(self.env.delivered),
            "stolen": set(self.env.stolen),
            "t": self.env.t,
            "total_cost": self.env.total_cost,
            "slip_count": self.env.slip_count,
            "rival_cooldown": self.env.rival_cooldown,
            "slip_prob": self.env.slip_prob,
            "closure_seed": self.env.closure_mgr.seed,
            "closure_rng_state": copy.deepcopy(self.env.closure_mgr.rng.bit_generator.state),
            "closure_events": copy.deepcopy(self.env.closure_mgr.events),
            "planned_route": list(self.env.closure_mgr.planned_route),
        }
        try:
            sim_res = run_episode(
                self.env,
                seed=seed,
                slip_prob=slip_prob,
                max_steps=max_steps,
                search_depth=int(config.ADVERSARY_SEARCH_DEPTH),
            )
        finally:
            self.env.seed = snapshot["seed"]
            self.env.rng.bit_generator.state = snapshot["rng_state"]
            self.env.agent_pos = snapshot["agent_pos"]
            self.env.rival_pos = snapshot["rival_pos"]
            self.env.delivered = snapshot["delivered"]
            self.env.stolen = snapshot["stolen"]
            self.env.t = snapshot["t"]
            self.env.total_cost = snapshot["total_cost"]
            self.env.slip_count = snapshot["slip_count"]
            self.env.rival_cooldown = snapshot["rival_cooldown"]
            self.env.slip_prob = snapshot["slip_prob"]
            self.env.closure_mgr.seed = snapshot["closure_seed"]
            self.env.closure_mgr.rng.bit_generator.state = snapshot["closure_rng_state"]
            self.env.closure_mgr.events = snapshot["closure_events"]
            self.env.closure_mgr.planned_route = snapshot["planned_route"]

        return sim_res
    