import os
import numpy as np
from oel_instance import make_instance, stamp

# Group Members & Mandatory Stamp
MEMBERS = [
    ("Akrash Noman", "00000453808"),
    ("Shahzaib Ali", "00000486980"),
    ("Saqib Khushal", "00000481674"),
    ("Capt. Hafiz Abdul Basit", "00000486992"),
]
CMS_IDS = [cms for _, cms in MEMBERS]
STAMP = stamp(MEMBERS)

# Instance Parameters (ID: 51f66530)
INSTANCE_ID = "51f66530"
GRID_FILE = f"grid_{INSTANCE_ID}.npy"
GRID_SIZE = 28
OBSTACLE_DENSITY = 0.218
SEED = int(INSTANCE_ID, 16)

TERRAIN_COST = {
    "road": 3,
    "grass": 4,
    "mud": 8,
    "ford": 9,
}
COST_LOOKUP = np.array([3, 4, 8, 9], dtype=float)  # 0=road, 1=grass, 2=mud, 3=ford
MIN_TERRAIN_COST = min(TERRAIN_COST.values())     # 3


START = (2, 21)
DELIVERY_STOPS = [(24, 10), (2, 9), (23, 14)]

ADVERSARY = "chaser"
SLIP_PROBABILITY = 0.26
ADVERSARY_SEARCH_DEPTH = 5

# Assigned Twists + Group-Proposed 3rd Twist
TWISTS = [
    "rival_steals_parcels",
    "rush_hour_time_varying_costs",
    "predictive_dynamic_road_closures",
]

# Twist 1: Rush-Hour Time-Varying Costs (active steps 6..13 in every 20-step cycle)
RUSH_HOUR_CYCLE = 20
RUSH_HOUR_WINDOW = (6, 13)
RUSH_HOUR_MULTIPLIER = np.array([2.0, 1.0, 1.25, 1.0])  # road x2, mud x1.25

# Twist 2: Rival Steals Parcels
RIVAL_START = (14, 14)
RIVAL_STEAL_PENALTY = 50

# Twist 3: Predictive Dynamic Road Closures & Directional Maintenance
CLOSURE_WARNING_LEAD = 4       # Steps of advance warning before restriction activates
CLOSURE_DURATION = 6           # Steps each restriction stays active
NUM_CLOSURE_EVENTS = 10        # Total seeded closure/maintenance events per episode
NEAR_ROUTE_RADIUS = 2          # Manhattan distance defining "near the route"
CLOSURE_LOCATION_PROBS = {
    "on_route": 0.50,          # Directly on planned route (high impact)
    "near_route": 0.30,        # Within 1-2 cells of route (moderate/detour impact)
    "random": 0.20,            # Anywhere on traversable road/grid cells (low/background)
}

# Part B: Search & Heuristic Defaults
IDS_MEMORY_LIMIT_STATES = 5000
INADMISSIBLE_HEURISTIC_WEIGHT = 1.8

# Part C: Rival Game Search Defaults
LOCAL_GAME_WINDOW_RADIUS = 6
NUM_RIVAL_GAMES = 30

def load_grid():
    """Load grid_51f66530.npy if available, otherwise regenerate from CMS_IDS."""
    try:
        if os.path.exists(GRID_FILE):
            return np.load(GRID_FILE)
    except Exception as e:
        print(f"Warning: could not load {GRID_FILE} ({e}), regenerating...")

    _, grid = make_instance(CMS_IDS)
    np.save(GRID_FILE, grid)
    return grid