import os
import sys
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


class RoadClosureManager:
    """
    Third Twist: Predictive Dynamic Road Closures and Directional Maintenance.
    Generates seeded closure/directional events with advance warnings and safe-exit rules.
    """

    def __init__(self, grid, planned_route=None, seed=None):
        self.grid = grid
        self.seed = config.SEED if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.planned_route = list(planned_route) if planned_route else []
        self.events = []

        if "predictive_dynamic_road_closures" in config.TWISTS:
            self.events = self._generate_events()

    def _categorize_cells(self):
        """Partition eligible road/traversable cells into on_route, near_route, and random."""
        protected = {config.START, *config.DELIVERY_STOPS}
        route_set = set(self.planned_route)

        on_route = []
        near_route = []
        other_free = []

        # Prefer road cells (0) first, fall back to any traversable cell (>= 0)
        candidates = np.argwhere(self.grid == 0)
        if len(candidates) < config.NUM_CLOSURE_EVENTS * 2:
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
        """Generate seeded road closure and directional restriction events."""
        on_route, near_route, other_free = self._categorize_cells()
        used_cells = set()
        events = []

        zones = ["on_route", "near_route", "random"]
        probs = [
            config.CLOSURE_LOCATION_PROBS["on_route"],
            config.CLOSURE_LOCATION_PROBS["near_route"],
            config.CLOSURE_LOCATION_PROBS["random"],
        ]

        for event_id in range(1, config.NUM_CLOSURE_EVENTS + 1):
            zone = str(self.rng.choice(zones, p=probs))
            pool = {
                "on_route": on_route,
                "near_route": near_route,
                "random": other_free,
            }[zone]

            # Fallback if chosen pool is empty
            if not pool:
                pool = on_route or near_route or other_free
            if not pool:
                break

            idx = int(self.rng.integers(0, len(pool)))
            cell = pool[idx]
            if cell in used_cells:
                continue
            used_cells.add(cell)

            active_start = int(self.rng.integers(4, 35))
            warn_start = max(0, active_start - config.CLOSURE_WARNING_LEAD)
            active_end = active_start + config.CLOSURE_DURATION

            # 50% full closure, 50% directional maintenance (1 or 2 allowed entry vectors)
            if self.rng.random() < 0.5:
                mode = "closed"
                allowed_dirs = []
            else:
                mode = "directional"
                num_dirs = int(self.rng.choice([1, 2]))
                chosen_idx = self.rng.choice(len(DIRECTIONS), size=num_dirs, replace=False)
                allowed_dirs = [DIRECTIONS[int(i)] for i in chosen_idx]

            events.append({
                "event_id": event_id,
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
        Check if moving from curr_pos to next_pos at step t is valid:
        - Safe-exit rule: if curr_pos is closed, agent may exit to any unrestricted neighbor.
        - Cannot enter a 'closed' cell.
        - Can enter a 'directional' cell only if move vector (dr, dc) is in allowed_dirs.
        """
        if "predictive_dynamic_road_closures" not in config.TWISTS:
            return True

        # Safe exit from curr_pos is always permitted as long as destination is valid
        event = self.get_active_event(next_pos, t)
        if event is None:
            return True

        if event["mode"] == "closed":
            return False

        if event["mode"] == "directional":
            move_vec = (next_pos[0] - curr_pos[0], next_pos[1] - curr_pos[1])
            return move_vec in event["allowed_dirs"]

        return True