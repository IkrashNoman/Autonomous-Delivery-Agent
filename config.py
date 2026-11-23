import os
import copy
import numpy as np
from oel_instance import make_instance, stamp

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Group Members & Mandatory Stamp
MEMBERS = [
    ("Akrash Noman", "00000453808"),
    ("Shahzaib Ali", "00000486980"),
    ("Saqib Khushal", "00000481674"),
    ("Capt. Hafiz Abdul Basit", "00000486992"),
]
CMS_IDS = [cms for _, cms in MEMBERS]
STAMP = stamp(MEMBERS)

# Instance: every value comes from the instructor's generator, so a change in
# oel_instance.py (for example the stop count) flows through automatically.
_INSTANCE, _GRID = make_instance(CMS_IDS)

INSTANCE_ID = _INSTANCE["instance_id"]
GRID_SIZE = _INSTANCE["grid_size"]
OBSTACLE_DENSITY = _INSTANCE["obstacle_density"]
SEED = int(INSTANCE_ID, 16)
GRID_FILE = os.path.join(ROOT_DIR, f"grid_{INSTANCE_ID}.npy")

TERRAIN_COST = dict(_INSTANCE["terrain_cost"])  # road, grass, mud, ford
COST_LOOKUP = np.array([TERRAIN_COST[t] for t in ("road", "grass", "mud", "ford")], dtype=float)  # 0=road,1=grass,2=mud,3=ford
MIN_TERRAIN_COST = min(TERRAIN_COST.values())

START = tuple(_INSTANCE["start"])
DELIVERY_STOPS = [tuple(s) for s in _INSTANCE["delivery_stops"]]

ADVERSARY = _INSTANCE["adversary"]
SLIP_PROBABILITY = _INSTANCE["slip_probability"]
ADVERSARY_SEARCH_DEPTH = _INSTANCE["adversary_search_depth"]

# Twists: two assigned by the generator, plus the group's approved third twist
RIVAL_TWIST = "rival_steals_parcels"
RUSH_TWIST = "rush_hour_time_varying_costs"
THIRD_TWIST = "road_closures_and_maintenance"
TWISTS = list(_INSTANCE["twists"]) + [THIRD_TWIST]

# Rival
RIVAL_START = (14, 14)
RIVAL_STEAL_PENALTY = 50

# Rush hour
RUSH_HOUR_CYCLE = 20
RUSH_HOUR_WINDOW = (6, 13)
RUSH_HOUR_MULTIPLIER = np.array([2.0, 1.0, 1.25, 1.0])

# Closures and maintenance (schedule is known to the planner)
CLOSURE_WARNING_LEAD = 4
CLOSURE_DURATION = 6
NUM_CLOSURE_EVENTS = 10
NEAR_ROUTE_RADIUS = 2
CLOSURE_LOCATION_PROBS = {
    "on_route": 0.50,
    "near_route": 0.30,
    "random": 0.20,
}

# Part B: search & heuristic defaults
IDS_MEMORY_LIMIT_STATES = 5000
INADMISSIBLE_HEURISTIC_WEIGHT = 1.8
WEIGHTED_ASTAR_EPSILON = INADMISSIBLE_HEURISTIC_WEIGHT

# Part C: rival game defaults
LOCAL_GAME_WINDOW_RADIUS = 6
NUM_RIVAL_GAMES = 30


def load_grid():
    """
    Return the instance grid. It is regenerated deterministically from CMS_IDS, so it always
    matches the generator. A copy is saved next to the project for reference only.
    """
    if not os.path.exists(GRID_FILE):
        try:
            np.save(GRID_FILE, _GRID)
        except OSError:
            pass
    return _GRID.copy()