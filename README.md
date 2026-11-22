# Autonomous Delivery Agent

> Autonomous grid-city delivery agent — from-scratch A\*/UCS/IDS/Greedy search, Minimax + Alpha-Beta adversarial evasion, stochastic slip simulation, and an LLM dispatcher with a validation guardrail layer.

**Academic project** — CS-370 Artificial Intelligence, Open Ended Lab (OEL), Fall 2026. Built from scratch as a course requirement. Not for production use.

**Group Members:**
Akrash Noman (00000453808) | Shahzaib Ali (00000486980) | Saqib Khushal (00000481674) | Capt. Hafiz Abdul Basit (00000486992)

---

## What this project is

An autonomous delivery agent operating in a **28×28 grid city** under three kinds of uncertainty:

1. **Terrain uncertainty** — road, grass, mud, and ford cells carry different traversal costs, and rush-hour traffic doubles road and ford costs cyclically.
2. **Actuation uncertainty** — the agent's moves slip with probability `p = 0.26`, so intended transitions are not the same as actual transitions.
3. **Adversarial competition** — a rival *chaser* actively pursues the agent, steals undelivered parcels if it reaches them first, and must be evaded in real time.

An **LLM dispatcher** accepts natural-language requests ("deliver stop 1 first, avoid mud") and translates them into grounded tool calls (`plan_route`, `simulate`, `check_rush_hour`) with a validation layer that intercepts hallucinated arguments before they reach the planner.

All search and game-tree algorithms are implemented **from scratch** — no `networkx` shortest paths, no `simpleai`, no AIMA module, no ready-made minimax packages.

---

## Unique instance

This project targets a **single generated instance**, produced by the instructor's `oel_instance.py` from the CMS IDs above. All reported numbers are reproducible by regenerating the instance from those IDs.

| Field | Value |
|---|---|
| Instance ID | `51f66530` |
| Grid size | 28 × 28 |
| Obstacle density | 0.218 |
| Start depot | (2, 21) |
| Delivery stops | (24, 10), (2, 9), (23, 14) |
| Terrain costs | road=3, grass=4, mud=8, ford=9 |
| Rival type | `chaser`, search depth 5 |
| Slip probability | 0.26 |
| Assigned twists | `rival_steals_parcels`, `rush_hour_time_varying_costs` |
| Proposed third twist | `cargo_weight_inertia` (each undelivered parcel adds +1 to traversal cost) |

---

## OEL compliance

This repository is an **academic submission** for CS-370 AI Open Ended Lab. The following OEL rules are enforced throughout the codebase.

### Allowed

- Open internet, documentation, papers, Stack Overflow, AI assistants.
- Libraries: `numpy`, `scipy`, `pandas`, `matplotlib`, `streamlit` / `pygame` / `tkinter`, LangChain or any LLM wrapper, any hosted or local LLM.
- Visualisation helpers such as `networkx` drawing functions.

### Not allowed

- **Library implementations of the core algorithms**: `networkx` shortest-path functions, `simpleai`, the AIMA search module, `gym` solvers, ready-made minimax packages.
- **BFS, UCS, IDS, Greedy, A\*, Minimax, Alpha-Beta must be written by the group** — no wrapper, no library call, no copy-paste from a repository.
- Copying a complete repository or another group's work (treated as plagiarism under rubric R3).

### AI assistant policy

AI assistants may be used for reference and debugging, but **every member must be able to explain every line of the submission in the viva**. Where an AI assistant was consulted for a reference or a debugging hint, the code is still written, verified, and understood by the group — no single-prompt generated files are submitted.

### Mandatory stamp

> **Every output — plot, table, console log, UI screen, report figure — prints the full name and CMS ID of every group member.**
> An output without this stamp is not marked.

The `config.stamp()` helper returns the formatted line:
Akrash Noman (00000453808) | Shahzaib Ali (00000486980) | Saqib Khushal (00000481674) | Capt. Hafiz Abdul Basit (00000486992)


This line is printed in every section banner in `run_all.py`, every figure title, every UI screen, and every table header.

### Viva and authenticity

- Each member answers questions on **any part**, not only the part they wrote.
- The instructor may **change one instance parameter live** (slip probability, stop location, or twist flag) and the system must adapt and rerun within ~5 minutes.
- The instructor regenerates the instance from the CMS IDs and checks names, IDs, and key numbers against the report.

### Third twist

The third twist is proposed by the group and approved by the instructor before implementation. No two groups may share the same twist. The proposal:

> **`cargo_weight_inertia`** — movement cost increases by +1 per undelivered parcel carried, reflecting vehicle payload inertia. This changes the cost function inside the search problem and the simulator, not the UI.

---

## Algorithms implemented from scratch

No third-party graph-search, game-tree, or minimax libraries are used. Each of the following is implemented directly in Python inside this repository:

    Uninformed search (BFS, UCS, DLS, IDS)
    Informed search (Greedy Best-First, A\*)
    Game tree search (Minimax, Alpha-Beta with domain-specific move ordering)
    Heuristic (Custom Minimum Spanning Tree (MST) over remaining stops, scaled by minimum terrain cost)

---

## Quick start

### Prerequisites
- Python 3.10+
- `numpy`, `matplotlib` for the core pipeline
- An LLM provider for Part D (see `requirements.txt` for the chosen wrapper)

### Install

git clone https://github.com/IkrashNoman/autonomous-delivery-agent.git
cd autonomous-delivery-agent
pip install -r requirements.txt


# Project Directory
