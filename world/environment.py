import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from world.cost_map import get_step_cost, is_rush_hour
from world.road_closures import RoadClosureManager, DIRECTIONS


class GridWorldEnv:
    """
    Core grid city simulation environment integrating:
    - Terrain costs & rush-hour time-varying multipliers
    - Stochastic movement slip (slip_probability = 0.26)
    - Chaser rival that steals parcels
    - Predictive dynamic road closures & directional maintenance
    """

    def __init__(self, grid=None, slip_prob=None, planned_route=None, seed=None):
        self.grid = config.load_grid() if grid is None else grid
        self.rows, self.cols = self.grid.shape
        self.slip_prob = config.SLIP_PROBABILITY if slip_prob is None else slip_prob
        self.seed = config.SEED if seed is None else seed
        self.rng = np.random.default_rng(self.seed)

        self.start = config.START
        self.stops = list(config.DELIVERY_STOPS)
        self.rival_start = self._find_valid_rival_start()
        self.closure_mgr = RoadClosureManager(self.grid, planned_route=planned_route, seed=self.seed)

        self.reset()

    def _find_valid_rival_start(self):
        """Ensure rival spawns on a traversable cell distinct from start and stops."""
        occupied = {self.start, *self.stops}
        r0, c0 = config.RIVAL_START
        if 0 <= r0 < self.rows and 0 <= c0 < self.cols and self.grid[r0, c0] >= 0 and (r0, c0) not in occupied:
            return (r0, c0)

        free_cells = np.argwhere(self.grid >= 0)
        center = np.array([self.rows // 2, self.cols // 2])
        order = np.argsort(np.linalg.norm(free_cells - center, axis=1))
        for idx in order:
            cand = (int(free_cells[idx][0]), int(free_cells[idx][1]))
            if cand not in occupied:
                return cand
        return (int(free_cells[0][0]), int(free_cells[0][1]))

    def reset(self):
        """Reset episode state to t = 0."""
        self.agent_pos = self.start
        self.rival_pos = self.rival_start
        self.delivered = set()
        self.stolen = set()
        self.t = 0
        self.total_cost = 0.0
        self.slip_count = 0
        self.rival_cooldown = 0
        return self.get_state()

    def get_state(self):
        """Return a snapshot dictionary of the current world state."""
        return {
            "agent_pos": self.agent_pos,
            "rival_pos": self.rival_pos,
            "delivered": sorted(list(self.delivered)),
            "stolen": sorted(list(self.stolen)),
            "remaining_stops": [s for s in self.stops if s not in self.delivered and s not in self.stolen],
            "t": self.t,
            "rush_hour": is_rush_hour(self.t),
            "total_cost": round(self.total_cost, 2),
            "slip_count": self.slip_count,
            "warnings": self.closure_mgr.get_upcoming_warnings(self.t),
        }

    def in_bounds_and_free(self, pos):
        """Check grid boundaries and static obstacles (-1)."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] >= 0

    def get_valid_neighbors(self, pos, t):
        """Return valid (next_pos, step_cost) transitions from pos at step t."""
        neighbors = []
        # Allow waiting in place (useful when waiting out a brief road closure or rush hour)
        wait_cost = get_step_cost(self.grid, pos, t) * 0.5
        neighbors.append((pos, wait_cost))

        for dr, dc in DIRECTIONS:
            nxt = (pos[0] + dr, pos[1] + dc)
            if self.in_bounds_and_free(nxt) and self.closure_mgr.is_move_allowed(pos, nxt, t):
                neighbors.append((nxt, get_step_cost(self.grid, nxt, t)))
        return neighbors

    def _apply_slip(self, pos, intended_move):
        """Apply perpendicular slip with probability self.slip_prob."""
        if intended_move == (0, 0) or self.rng.random() >= self.slip_prob:
            return intended_move, False

        dr, dc = intended_move
        # Perpendicular directions to (dr, dc)
        perp_options = [(dc, dr), (-dc, -dr)]
        chosen_perp = perp_options[int(self.rng.integers(0, 2))]
        slipped_pos = (pos[0] + chosen_perp[0], pos[1] + chosen_perp[1])

        if self.in_bounds_and_free(slipped_pos) and self.closure_mgr.is_move_allowed(pos, slipped_pos, self.t):
            return chosen_perp, True

        # If perpendicular cell is blocked, agent slips and stays in current cell
        return (0, 0), True

    def _default_chaser_step(self):
        """Greedy chaser fallback step when not driven by Part C Minimax/Alpha-Beta."""
        if self.rival_cooldown > 0:
            self.rival_cooldown -= 1
            return self.rival_pos

        best_pos = self.rival_pos
        best_dist = abs(self.rival_pos[0] - self.agent_pos[0]) + abs(self.rival_pos[1] - self.agent_pos[1])
        for dr, dc in DIRECTIONS:
            cand = (self.rival_pos[0] + dr, self.rival_pos[1] + dc)
            if self.in_bounds_and_free(cand):
                d = abs(cand[0] - self.agent_pos[0]) + abs(cand[1] - self.agent_pos[1])
                if d < best_dist:
                    best_dist = d
                    best_pos = cand
        return best_pos

    def step(self, action, rival_next_pos=None):
        """
        Advance simulation by one time step.
        action: (dr, dc) movement tuple from DIRECTIONS or (0, 0) to wait.
        rival_next_pos: optional coordinate chosen by Part C adversarial search.
        """
        actual_move, slipped = self._apply_slip(self.agent_pos, action)
        if slipped:
            self.slip_count += 1

        candidate_pos = (self.agent_pos[0] + actual_move[0], self.agent_pos[1] + actual_move[1])
        if self.in_bounds_and_free(candidate_pos) and self.closure_mgr.is_move_allowed(
            self.agent_pos, candidate_pos, self.t
        ):
            next_pos = candidate_pos
        else:
            next_pos = self.agent_pos

        step_cost = get_step_cost(self.grid, next_pos, self.t)
        if next_pos == self.agent_pos:
            step_cost *= 0.5

        self.agent_pos = next_pos
        self.total_cost += step_cost

        # Check parcel delivery
        if self.agent_pos in self.stops and self.agent_pos not in self.delivered and self.agent_pos not in self.stolen:
            self.delivered.add(self.agent_pos)

        # Move rival
        if rival_next_pos is not None and self.in_bounds_and_free(rival_next_pos):
            self.rival_pos = rival_next_pos
        else:
            self.rival_pos = self._default_chaser_step()

        # Check rival parcel theft (collision with agent or reaching an active stop first)
        stole_parcel = False
        if "rival_steals_parcels" in config.TWISTS and self.rival_cooldown == 0:
            active_stops = [s for s in self.stops if s not in self.delivered and s not in self.stolen]
            if self.rival_pos in active_stops:
                self.stolen.add(self.rival_pos)
                self.total_cost += config.RIVAL_STEAL_PENALTY
                self.rival_cooldown = 3
                stole_parcel = True
            elif self.rival_pos == self.agent_pos and active_stops:
                # Chaser intercepts and steals the nearest undelivered parcel
                nearest = min(active_stops, key=lambda s: abs(s[0] - self.agent_pos[0]) + abs(s[1] - self.agent_pos[1]))
                self.stolen.add(nearest)
                self.total_cost += config.RIVAL_STEAL_PENALTY
                self.rival_cooldown = 3
                stole_parcel = True

        self.t += 1
        done = len(self.delivered) + len(self.stolen) == len(self.stops)

        info = {
            "slipped": slipped,
            "actual_move": actual_move,
            "step_cost": round(step_cost, 2),
            "stole_parcel": stole_parcel,
            "done": done,
        }
        return self.get_state(), info