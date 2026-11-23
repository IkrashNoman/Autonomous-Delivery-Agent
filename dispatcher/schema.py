import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

ALLOWED_ACTIONS = {
    "prioritize_stop",
    "avoid_zone",
    "avoid_rush_hour",
    "replan_now",
    "query_status",
}


class DispatchValidationError(Exception):
    """Raised when a parsed LLM command violates schema or environment safety constraints."""


def validate_dispatch_command(raw_payload, env):
    """
    Validate and normalize an LLM-generated dispatch command against hard environment rules:
    1. Must be valid JSON / dict with an allowed 'action'.
    2. Coordinates must be within [0, GRID_SIZE) and not on static obstacles (-1).
    3. 'avoid_zone' cannot block START or any active DELIVERY_STOPS.
    4. 'prioritize_stop' must reference an actual stop in env.stops.
    Returns a clean, normalized command dictionary or raises DispatchValidationError.
    """
    if isinstance(raw_payload, str):
        try:
            cmd = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise DispatchValidationError(f"Malformed JSON output from LLM: {exc}") from exc
    elif isinstance(raw_payload, dict):
        cmd = dict(raw_payload)
    else:
        raise DispatchValidationError("Command payload must be a JSON string or dictionary.")

    action = cmd.get("action")
    if action not in ALLOWED_ACTIONS:
        raise DispatchValidationError(f"Unsupported action '{action}'. Allowed: {sorted(ALLOWED_ACTIONS)}")

    reason = str(cmd.get("reason", "Operator directive")).strip()
    target = cmd.get("target")
    radius = int(cmd.get("radius", 0))

    if action in {"prioritize_stop", "avoid_zone"}:
        if (
            not isinstance(target, (list, tuple))
            or len(target) != 2
            or not all(isinstance(v, int) for v in target)
        ):
            raise DispatchValidationError(f"Action '{action}' requires integer [row, col] target, got {target}.")

        r, c = int(target[0]), int(target[1])
        if not (0 <= r < env.rows and 0 <= c < env.cols):
            raise DispatchValidationError(f"Target {(r, c)} is out of bounds for {env.rows}x{env.cols} grid.")

        if env.grid[r, c] < 0:
            raise DispatchValidationError(f"Target {(r, c)} is a static obstacle (-1).")

        coord = (r, c)

        if action == "prioritize_stop":
            if coord not in env.stops:
                raise DispatchValidationError(
                    f"Target {coord} is not a valid delivery stop. Valid stops: {env.stops}"
                )
            if coord in env.delivered or coord in env.stolen:
                raise DispatchValidationError(f"Stop {coord} is already completed or stolen.")

        if action == "avoid_zone":
            radius = max(0, min(radius, 3))
            protected = {env.start, *env.stops}
            blocked_cells = []
            for rr in range(max(0, r - radius), min(env.rows, r + radius + 1)):
                for cc in range(max(0, c - radius), min(env.cols, c + radius + 1)):
                    if abs(rr - r) + abs(cc - c) <= radius and env.grid[rr, cc] >= 0:
                        if (rr, cc) in protected:
                            raise DispatchValidationError(
                                f"Safety violation: avoid_zone around {coord} (r={radius}) blocks protected cell {(rr, cc)}."
                            )
                        blocked_cells.append((rr, cc))

            return {
                "action": action,
                "target": coord,
                "radius": radius,
                "blocked_cells": blocked_cells,
                "reason": reason,
            }

        return {
            "action": action,
            "target": coord,
            "radius": 0,
            "blocked_cells": [],
            "reason": reason,
        }

    return {
        "action": action,
        "target": None,
        "radius": 0,
        "blocked_cells": [],
        "reason": reason,
    }

def validate_simulation_args(seed, slip_prob, max_steps):
    """Check LLM-supplied simulate() arguments against safe ranges. Returns the normalised (seed, slip_prob, max_steps)."""
    try:
        seed = int(seed)
        slip_prob = float(slip_prob)
        max_steps = int(max_steps)
    except (TypeError, ValueError) as exc:
        raise DispatchValidationError(f"simulate() arguments must be numeric: {exc}") from exc
    if not (0.0 <= slip_prob <= 0.6):
        raise DispatchValidationError(f"slip_prob {slip_prob} is outside the allowed range [0.0, 0.6].")
    if not (1 <= max_steps <= 120):
        raise DispatchValidationError(f"max_steps {max_steps} is outside the allowed range [1, 120].")
    return seed, slip_prob, max_steps


def validate_inspect_args(t, max_t=120):
    """Check the LLM-supplied time for inspect_closures(t). None means 'current time'."""
    if t is None:
        return None
    try:
        t = int(t)
    except (TypeError, ValueError) as exc:
        raise DispatchValidationError(f"inspect_closures t must be an integer: {exc}") from exc
    if not (0 <= t <= max_t):
        raise DispatchValidationError(f"inspect_closures t={t} is outside the allowed range [0, {max_t}].")
    return t