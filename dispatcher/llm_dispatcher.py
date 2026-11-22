import os
import re
import sys
import json
import time
import urllib.request
import urllib.error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from world.environment import GridWorldEnv
from dispatcher.tools import DispatcherToolkit
from dispatcher.schema import DispatchValidationError


def _load_dotenv():
    """Load key=value pairs from .env or .env.txt in project root or CWD."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(root_dir, ".env"),
        os.path.join(root_dir, ".env.txt"),
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.getcwd(), ".env.txt"),
    ]
    for env_path in candidates:
        if not os.path.exists(env_path):
            continue
        with open(env_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and not os.environ.get(k):
                    os.environ[k] = v


_load_dotenv()


REACT_SYSTEM_PROMPT = """You are a Sense-Reason-Act Autonomous Delivery Dispatcher on a 28x28 grid.
Available tools:
1. inspect_closures(t=None) -> returns active/upcoming road closures, rush_hour, rival_dist, remaining_stops
2. add_avoid_zone(row, col, radius=0) -> blocks hazard cells (pass row=null, col=null if user gave no coordinates)
3. plan_route(priority_stop=None, avoid_rush_hour=False) -> runs constrained A* route planner
4. simulate(seed=42, slip_prob=0.26, max_steps=120) -> simulates full stochastic adversarial episode

CRITICAL RULES:
- Complete ALL actions requested by the user BEFORE returning "tool": "FINISH".
- Example 1: "Check closures and prioritize stop (2, 9)" -> Step 1: inspect_closures({}), Step 2: plan_route({"priority_stop": [2, 9]}), Step 3: FINISH.
- Example 2: "Prioritize stop (23, 14) and simulate" -> Step 1: plan_route({"priority_stop": [23, 14]}), Step 2: simulate({"seed": 42}), Step 3: FINISH.
- Example 3: "Avoid zone around (10, 14) radius 1" -> Step 1: add_avoid_zone({"row": 10, "col": 14, "radius": 1}), Step 2: plan_route({}), Step 3: FINISH.
- Never invent coordinates if the user gave none; pass null or call an invalid tool name so the guardrail catches it.
- Output STRICT JSON ONLY (no markdown):
{"thought": "...", "tool": "<inspect_closures|add_avoid_zone|plan_route|simulate|FINISH>", "args": {...}}"""


class LLMDispatcher:
    """
    Tool-calling ReAct Dispatcher Agent implementing Sense -> Reason -> Act loops
    with Groq LLM (`openai/gpt-oss-20b`), Thought/Action/Observation traces, and schema guardrails.
    """

    def __init__(self, env=None, use_live_api=True, enforce_guardrails=True, request_delay_sec=2.1):
        self.env = GridWorldEnv() if env is None else env
        self.enforce_guardrails = enforce_guardrails
        self.toolkit = DispatcherToolkit(self.env, enforce_guardrails=enforce_guardrails)
        self.use_live_api = use_live_api
        self.request_delay_sec = request_delay_sec
        self.api_errors = []
        self.guardrail_blocks = []
        self.api_calls_made = 0

    def _call_groq_step(self, instruction, trace):
        """Call Groq API (openai/gpt-oss-20b) with rate-limit pacing and retry on HTTP 429."""
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not self.use_live_api or not api_key:
            return None

        url = "https://api.groq.com/openai/v1/chat/completions"
        user_msg = (
            f"Grid Info -> Start: {self.env.start}, Stops: {self.env.stops}, Agent: {self.env.agent_pos}, t={self.env.t}\n"
            f"Operator Request: {instruction}\n"
            f"Completed Steps So Far: {json.dumps(trace)}\n"
            f"If any requested action in Operator Request has NOT been called yet, call that tool now. Otherwise return FINISH.\n"
            f"Return next JSON step:"
        )
        payload = {
            "model": "openai/gpt-oss-20b",
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        }

        for attempt in range(3):
            if self.request_delay_sec > 0:
                time.sleep(self.request_delay_sec)
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "Mozilla/5.0",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=14) as resp:
                    self.api_calls_made += 1
                    data = json.loads(resp.read().decode("utf-8"))
                    raw = data["choices"][0]["message"]["content"]
                    match = re.search(r"\{.*\}", raw, re.DOTALL)
                    if not match:
                        raise ValueError(f"Non-JSON response from LLM: {raw[:120]}")
                    return json.loads(match.group(0))
            except urllib.error.HTTPError as http_err:
                if http_err.code == 429 and attempt < 2:
                    time.sleep(4.0 * (attempt + 1))
                    continue
                self.api_errors.append({"instruction": instruction, "type": "GROQ_HTTP_ERROR", "detail": str(http_err)})
                return None
            except Exception as exc:
                self.api_errors.append({"instruction": instruction, "type": "GROQ_API_ERROR", "detail": str(exc)})
                return None
        return None

    def _plan_react_steps_offline(self, instruction):
        """Deterministic ReAct policy for offline mode or API fallback."""
        text = instruction.lower().strip()
        coord_match = re.search(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", text)
        target = [int(coord_match.group(1)), int(coord_match.group(2))] if coord_match else None
        radius_match = re.search(r"radius\s*(\d+)", text)
        radius = int(radius_match.group(1)) if radius_match else 0

        if any(p in text for p in ("ignore previous", "system prompt", "override safety", "delete grid")):
            return [{
                "thought": "SENSE: Policy-override keywords detected; testing tool guardrail.",
                "tool": "delete_all_obstacles",
                "args": {},
            }]

        steps = []
        if any(w in text for w in ("closure", "status", "warning", "rival", "chaser", "traffic", "rush", "simulate", "test", "nearest")):
            steps.append({
                "thought": "SENSE: Inspecting dynamic road closures, rush-hour phase, and rival distance.",
                "tool": "inspect_closures",
                "args": {},
            })

        if any(w in text for w in ("avoid", "block", "hazard", "danger", "detour", "close")) and "rush" not in text:
            if target is None:
                steps.append({
                    "thought": "REASON: Missing coordinates for avoid_zone; passing None to guardrail.",
                    "tool": "add_avoid_zone",
                    "args": {"row": None, "col": None, "radius": radius},
                })
            else:
                steps.append({
                    "thought": f"REASON: Validating avoidance zone around {target} (radius={radius}).",
                    "tool": "add_avoid_zone",
                    "args": {"row": target[0], "col": target[1], "radius": radius},
                })
            steps.append({
                "thought": "ACT: Recomputing constrained A* route.",
                "tool": "plan_route",
                "args": {},
            })
            return steps

        if any(w in text for w in ("prioritize", "priority", "first", "urgent", "deliver to", "visit", "head to")):
            if target is None and any(w in text for w in ("nearest", "closest")):
                rem = [s for s in self.env.stops if s not in self.env.delivered and s not in self.env.stolen]
                nearest = min(rem, key=lambda s: abs(s[0] - self.env.agent_pos[0]) + abs(s[1] - self.env.agent_pos[1]))
                target = [nearest[0], nearest[1]]
            if target is None:
                target = [99, 99]
            steps.append({
                "thought": f"REASON: Prioritizing stop {target} and computing A* route.",
                "tool": "plan_route",
                "args": {"priority_stop": target},
            })
            if any(w in text for w in ("simulate", "rollout", "verify", "test")):
                steps.append({
                    "thought": "ACT: Simulating stochastic rollout under slip risk.",
                    "tool": "simulate",
                    "args": {"seed": config.SEED, "slip_prob": config.SLIP_PROBABILITY},
                })
            return steps

        if any(w in text for w in ("rush", "congestion", "peak")):
            steps.append({
                "thought": "REASON: Applying rush-hour avoidance penalty and replanning.",
                "tool": "plan_route",
                "args": {"avoid_rush_hour": True},
            })
            return steps

        if any(w in text for w in ("simulate", "rollout", "win rate", "slip")):
            steps.append({
                "thought": "ACT: Executing stochastic adversarial simulation.",
                "tool": "simulate",
                "args": {"seed": config.SEED, "slip_prob": config.SLIP_PROBABILITY},
            })
            return steps

        if steps:
            return steps

        return [{
            "thought": "SENSE: Request is underspecified with no valid action.",
            "tool": "ambiguous_noop",
            "args": {},
        }]

    def execute_tool(self, tool_name, args):
        """Execute a tool call through the schema guardrail layer (when enforce_guardrails=True)."""
        valid_tools = {
            "inspect_closures": self.toolkit.inspect_closures,
            "add_avoid_zone": self.toolkit.add_avoid_zone,
            "plan_route": self.toolkit.plan_route,
            "simulate": self.toolkit.simulate,
        }
        if tool_name not in valid_tools:
            raise DispatchValidationError(f"Unknown/disallowed tool '{tool_name}'. Allowed: {sorted(valid_tools)}")
        return valid_tools[tool_name](**args)

    def dispatch(self, instruction, reset_before=True):
        """
        Execute a Sense -> Reason -> Act ReAct session using Groq LLM (`openai/gpt-oss-20b`)
        with automatic fallback and strict schema guardrail validation.
        """
        if reset_before:
            self.toolkit.reset_constraints()

        t0 = time.perf_counter()
        baseline = self.toolkit.plan_route()
        cost_before = baseline["cost"]

        tao_trace = []
        tool_calls = 0
        final_cost = cost_before
        used_live = False

        if self.use_live_api and os.environ.get("GROQ_API_KEY", "").strip():
            for idx in range(1, 5):
                llm_step = self._call_groq_step(instruction, tao_trace)
                if llm_step is None:
                    break
                used_live = True
                thought = str(llm_step.get("thought", "ReAct step"))
                tool_name = str(llm_step.get("tool", "FINISH"))
                args = llm_step.get("args") or {}

                if tool_name == "FINISH":
                    tao_trace.append({
                        "step": idx,
                        "source": "GROQ_GPT_OSS_20B",
                        "sense_reason_act": "FINISH",
                        "thought": thought,
                        "action": "FINISH()",
                        "observation": "TASK_COMPLETE",
                    })
                    break

                tool_calls += 1
                try:
                    obs = self.execute_tool(tool_name, args)
                    tao_trace.append({
                        "step": idx,
                        "source": "GROQ_GPT_OSS_20B",
                        "sense_reason_act": "SENSE" if tool_name == "inspect_closures" else ("REASON" if tool_name == "add_avoid_zone" else "ACT"),
                        "thought": thought,
                        "action": f"{tool_name}({args})",
                        "observation": obs,
                    })
                    if tool_name == "plan_route":
                        final_cost = obs["cost"]
                except (DispatchValidationError, TypeError, ValueError) as err:
                    err_msg = str(err)
                    self.guardrail_blocks.append({
                        "instruction": instruction,
                        "source": "GROQ_GPT_OSS_20B",
                        "tool": tool_name,
                        "args": args,
                        "caught_error": err_msg,
                    })
                    tao_trace.append({
                        "step": idx,
                        "source": "GROQ_GPT_OSS_20B",
                        "sense_reason_act": "GUARDRAIL_BLOCK",
                        "thought": thought,
                        "action": f"{tool_name}({args})",
                        "observation": f"REJECTED_BY_GUARDRAIL: {err_msg}",
                    })
                    return {
                        "instruction": instruction,
                        "source": "GROQ_GPT_OSS_20B",
                        "status": "REJECTED",
                        "error": err_msg,
                        "tool_calls": tool_calls,
                        "tools_used": [s["action"].split("(")[0] for s in tao_trace],
                        "cost_before": cost_before,
                        "cost_after": cost_before,
                        "trace": tao_trace,
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                    }

            if used_live and tool_calls > 0:
                return {
                    "instruction": instruction,
                    "source": "GROQ_GPT_OSS_20B",
                    "status": "EXECUTED",
                    "error": None,
                    "tool_calls": tool_calls,
                    "tools_used": [s["action"].split("(")[0] for s in tao_trace],
                    "cost_before": cost_before,
                    "cost_after": final_cost,
                    "trace": tao_trace,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                }

        # Deterministic offline fallback
        planned_steps = self._plan_react_steps_offline(instruction)
        for idx, step in enumerate(planned_steps, start=1):
            thought = step["thought"]
            tool_name = step["tool"]
            args = step["args"]
            tool_calls += 1
            try:
                obs = self.execute_tool(tool_name, args)
                tao_trace.append({
                    "step": idx,
                    "source": "OFFLINE_FALLBACK",
                    "sense_reason_act": "SENSE" if tool_name == "inspect_closures" else ("REASON" if tool_name == "add_avoid_zone" else "ACT"),
                    "thought": thought,
                    "action": f"{tool_name}({args})",
                    "observation": obs,
                })
                if tool_name == "plan_route":
                    final_cost = obs["cost"]
            except (DispatchValidationError, TypeError, ValueError) as err:
                err_msg = str(err)
                self.guardrail_blocks.append({
                    "instruction": instruction,
                    "source": "OFFLINE_FALLBACK",
                    "tool": tool_name,
                    "args": args,
                    "caught_error": err_msg,
                })
                tao_trace.append({
                    "step": idx,
                    "source": "OFFLINE_FALLBACK",
                    "sense_reason_act": "GUARDRAIL_BLOCK",
                    "thought": thought,
                    "action": f"{tool_name}({args})",
                    "observation": f"REJECTED_BY_GUARDRAIL: {err_msg}",
                })
                return {
                    "instruction": instruction,
                    "source": "OFFLINE_FALLBACK",
                    "status": "REJECTED",
                    "error": err_msg,
                    "tool_calls": tool_calls,
                    "tools_used": [s["action"].split("(")[0] for s in tao_trace],
                    "cost_before": cost_before,
                    "cost_after": cost_before,
                    "trace": tao_trace,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                }

        return {
            "instruction": instruction,
            "source": "OFFLINE_FALLBACK",
            "status": "EXECUTED",
            "error": None,
            "tool_calls": tool_calls,
            "tools_used": [s["action"].split("(")[0] for s in tao_trace],
            "cost_before": cost_before,
            "cost_after": final_cost,
            "trace": tao_trace,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }