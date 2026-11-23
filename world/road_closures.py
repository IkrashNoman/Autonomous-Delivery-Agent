import os
import sys
from collections import deque
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_NAMES = {
    (-1, 0): "UP",
    (1, 0): "DOWN",
    (0, -1): "LEFT",
    (0, 1): "RIGHT",
}

# Valid maintenance corridor patterns:
# 4 strict one-way lanes OR 2 strict two-way straight axes (never perpendicular turns)
MAINTENANCE_CORRIDOR_MODES = [
    [(-1, 0)],           # Northbound only (South -> North)
    [(1, 0)],            # Southbound only (North -> South)
    [(0, -1)],           # Westbound only  (East -> West)
    [(0, 1)],            # Eastbound only  (West -> East)
    [(0, -1), (0, 1)],   # Two-way East <-> West only
    [(-1, 0), (1, 0)],   # Two-way North <-> South only
]


class RoadClosureManager:
    """
    Third Twist: Predictive Dynamic Road Closures and Directional Maintenance.
    Generates exact NUM_CLOSURE_EVENTS seeded closure/directional events biased toward
    the planned delivery corridor (0.50 on_route, 0.30 near_route, 0.20 random).
    """

    def __init__(self, grid, planned_route=None, seed=None):
        self.grid = grid
        self.seed = config.SEED if seed is None else seed
        self.rng = np.random.default_rng(self.seed)

        if planned_route:
            self.planned_route = list(planned_route)
        else:
            self.planned_route = self._compute_baseline_corridor()

        self.events = []
        if "predictive_dynamic_road_closures" in config.TWISTS:
            self.events = self._generate_events()

    def _compute_baseline_corridor(self):
        """
        Compute a baseline shortest-step corridor visiting START -> DELIVERY_STOPS
        when no explicit route is supplied.
        """
        waypoints = [config.START] + list(config.DELIVERY_STOPS)
        corridor = []
        rows, cols = self.grid.shape

        for i in range(len(waypoints) - 1):
            src, dst = waypoints[i], waypoints[i + 1]
            q = deque([src])
            parent = {src: None}
            while q:
                curr = q.popleft()
                if curr == dst:
                    break
                for dr, dc in DIRECTIONS:
                    nxt = (curr[0] + dr, curr[1] + dc)
                    if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols and self.grid[nxt] >= 0 and nxt not in parent:
                        parent[nxt] = curr
                        q.append(nxt)

            if dst in parent:
                seg = []
                cur = dst
                while cur is not None:
                    seg.append(cur)
                    cur = parent[cur]
                seg.reverse()
                corridor.extend(seg)

        return corridor

    def bind_route(self, planned_route):
        """Re-seed and regenerate closure events around an updated planned route."""
        self.planned_route = list(planned_route)
        self.rng = np.random.default_rng(self.seed)
        if "predictive_dynamic_road_closures" in config.TWISTS:
            self.events = self._generate_events()

    def _categorize_cells(self):
        """Partition eligible traversable cells into on_route, near_route, and random."""
        protected = {config.START, *config.DELIVERY_STOPS}
        route_set = set(self.planned_route)

        on_route = []
        near_route = []
        other_free = []

        candidates = np.argwhere(self.grid >= 0)

        for r, c in candidates:
            cell = (int(r), int(c))
            if cell in protected:
                continue

            if route_set:
                min_dist = min(abs(cell[0] - rr) + abs(cell[1] - rc) for rr, rc in route_set)
                if min_dist == 0:
                    on_route.append(cell)
                elif min_dist <= config.NEAR_ROUTE_RADIUS:
                    near_route.append(cell)
                else:
                    other_free.append(cell)
            else:
                other_free.append(cell)

        return on_route, near_route, other_free

    def _generate_events(self):
        """Generate exactly NUM_CLOSURE_EVENTS unique closure/directional events."""
        on_route, near_route, other_free = self._categorize_cells()
        used_cells = set()
        events = []

        zones = ["on_route", "near_route", "random"]
        probs = [
            config.CLOSURE_LOCATION_PROBS["on_route"],
            config.CLOSURE_LOCATION_PROBS["near_route"],
            config.CLOSURE_LOCATION_PROBS["random"],
        ]

        attempts = 0
        max_attempts = config.NUM_CLOSURE_EVENTS * 25

        while len(events) < config.NUM_CLOSURE_EVENTS and attempts < max_attempts:
            attempts += 1
            zone = str(self.rng.choice(zones, p=probs))
            pool = {
                "on_route": [c for c in on_route if c not in used_cells],
                "near_route": [c for c in near_route if c not in used_cells],
                "random": [c for c in other_free if c not in used_cells],
            }[zone]

            if not pool:
                fallback = [c for c in (on_route + near_route + other_free) if c not in used_cells]
                if not fallback:
                    break
                pool = fallback

            cell = pool[int(self.rng.integers(0, len(pool)))]
            used_cells.add(cell)

            active_start = int(self.rng.integers(4, 35))
            warn_start = max(0, active_start - config.CLOSURE_WARNING_LEAD)
            active_end = active_start + config.CLOSURE_DURATION

            if self.rng.random() < 0.5:
                mode = "closed"
                allowed_dirs = []
            else:
                mode = "directional"
                pattern_idx = int(self.rng.integers(0, len(MAINTENANCE_CORRIDOR_MODES)))
                allowed_dirs = list(MAINTENANCE_CORRIDOR_MODES[pattern_idx])

            events.append({
                "event_id": len(events) + 1,
                "cell": cell,
                "zone": zone,
                "mode": mode,
                "allowed_dirs": allowed_dirs,
                "warn_start": warn_start,
                "active_start": active_start,
                "active_end": active_end,
            })

        return sorted(events, key=lambda e: e["active_start"])

    def get_active_event(self, cell, t):
        """Return the active restriction dict for cell at step t, or None."""
        for ev in self.events:
            if ev["cell"] == cell and ev["active_start"] <= t < ev["active_end"]:
                return ev
        return None

    def get_closed_cells(self, t):
        """Return list of coordinates that are fully closed ('closed') at step t."""
        if "predictive_dynamic_road_closures" not in config.TWISTS:
            return []
        return [
            ev["cell"]
            for ev in self.events
            if ev["mode"] == "closed" and ev["active_start"] <= t < ev["active_end"]
        ]

    def get_upcoming_warnings(self, t):
        """Return events currently in their advance warning window or active at step t."""
        visible = []
        for ev in self.events:
            if ev["warn_start"] <= t < ev["active_end"]:
                status = "ACTIVE" if t >= ev["active_start"] else "WARNING"
                visible.append({
                    "event_id": ev["event_id"],
                    "cell": ev["cell"],
                    "zone": ev["zone"],
                    "mode": ev["mode"],
                    "allowed_directions": [DIR_NAMES[d] for d in ev["allowed_dirs"]],
                    "status": status,
                    "activates_at": ev["active_start"],
                    "ends_at": ev["active_end"],
                })
        return visible

    def is_move_allowed(self, curr_pos, next_pos, t):
        """
        Enforce strict dynamic closure & directional maintenance rules:
        1. Cannot enter a 'closed' cell under any circumstances.
        2. When entering a 'directional' maintenance cell (next_pos), move_vec MUST be in its allowed_dirs.
        3. When exiting a 'directional' maintenance cell (curr_pos), move_vec MUST ALSO be in its allowed_dirs
           (prevents entering from West and turning South inside a single-axis maintenance lane).
        4. Safe-exit rule: if curr_pos is fully 'closed' (because it closed while the agent stood on it),
           the agent may exit to any valid non-closed neighbor.
        """
        if "predictive_dynamic_road_closures" not in config.TWISTS:
            return True

        if curr_pos == next_pos:
            # Waiting on a fully closed cell is disallowed; must exit if possible
            curr_ev = self.get_active_event(curr_pos, t)
            if curr_ev is not None and curr_ev["mode"] == "closed":
                return False
            return True

        move_vec = (next_pos[0] - curr_pos[0], next_pos[1] - curr_pos[1])

        # Check exit constraint from curr_pos if curr_pos is under directional maintenance
        curr_event = self.get_active_event(curr_pos, t)
        if curr_event is not None and curr_event["mode"] == "directional":
            if move_vec not in curr_event["allowed_dirs"]:
                return False

        # Check entry constraint into next_pos
        next_event = self.get_active_event(next_pos, t)
        if next_event is None:
            return True

        if next_event["mode"] == "closed":
            return False

        if next_event["mode"] == "directional":
            return move_vec in next_event["allowed_dirs"]

        return True