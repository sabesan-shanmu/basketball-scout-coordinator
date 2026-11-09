"""LangGraph orchestration of the scouting swarm.

Default layout ("parallel"):

              +-> shot_chart_analyst ----+
              +-> star_coverage ---------+
 input_guard -+-> defensive_coordinator -+-> validator --(all pass / out of retries)--> head_coach -> END
              +-> offensive_architect ---+       |
                      ^                          | (failed reports, retries left)
                      +--------------------------+

The specialists write into the shared state (short-term memory). The validator
checks every report against the tool output that agent actually received.
Failing agents are sent back with their list of problems, and only passing
reports reach the Head Coach.

Every design choice the experiments compare is a field of RunConfig, so a
variant is just a different config passed to run().
"""
from __future__ import annotations

import operator
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Callable, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from . import memory, prompts
from .guardrails import check_matchup, ungrounded_numbers, validate_report
from .tools import ALL_TOOLS, to_json

DEFAULT_MODEL = "qwen3"
MAX_TOOL_ROUNDS = 8
SPECIALISTS = list(prompts.ROLES)
SINGLE_AGENT_ROLE = "solo_scout"
ZONE_LABEL_FIELDS = ("read", "label")


@dataclass(frozen=True)
class RunConfig:
    """Switches for every design choice. The defaults are the full system."""

    model: str = field(default_factory=lambda: os.getenv("SCOUT_MODEL", DEFAULT_MODEL))
    # prompting
    grounding_rules: bool = True       # shared "only use tool numbers" rules in every prompt
    report_format: bool = True         # fixed FINDINGS / RECOMMENDATIONS / DATA GAPS format
    zone_labels: bool = True           # plain-language "read"/"label" fields in tool output
    # orchestration
    mode: Literal["parallel", "sequential", "single_agent"] = "parallel"
    enforce_validation: bool = True    # False: validator only records problems; nothing is blocked
    max_retries: int = 1
    # memory
    use_memory: bool = True
    memory_file: Path = memory.MEMORY_FILE
    # tool use
    removed_tools: frozenset[str] = frozenset()
    # safeguards
    tool_allowlist: bool = True        # False: every agent may call every tool
    injected_text: str | None = None   # appended to every tool result (prompt-injection test)

    @property
    def roles(self) -> list[str]:
        return [SINGLE_AGENT_ROLE] if self.mode == "single_agent" else SPECIALISTS


def role_spec(role: str) -> dict:
    return prompts.SINGLE_AGENT if role == SINGLE_AGENT_ROLE else prompts.ROLES[role]


def merge_dicts(old: dict, new: dict) -> dict:
    """Reducer so parallel nodes can each add their own keys to the same dict."""
    return {**old, **new}


class ScoutState(TypedDict, total=False):
    my_team: str
    opponent: str
    focus_player: str | None
    reports: Annotated[dict, merge_dicts]          # role -> report text
    evidence: Annotated[dict, merge_dicts]         # role -> tool outputs (JSON strings)
    problems: Annotated[dict, merge_dicts]         # role -> validation problems
    attempts: Annotated[dict, merge_dicts]         # role -> number of runs
    trace: Annotated[list, operator.add]           # log of every tool call
    validation_rounds: Annotated[list, operator.add]  # problems found in each validator pass
    final_plan: str
    plan_problems: list[str]


def get_llm(config: RunConfig) -> ChatOllama:
    return ChatOllama(model=config.model, temperature=0.1, num_ctx=16384, reasoning=False)


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


# ---------------------------------------------------------------- specialists
def _system_prompt(role: str, state: ScoutState, config: RunConfig) -> str:
    focus = state.get("focus_player")
    task = role_spec(role)["task"].format(
        my_team=state["my_team"],
        opponent=state["opponent"],
        focus_clause=f" (always include {focus})" if focus else "",
    )
    parts = [prompts.GROUNDING_RULES] if config.grounding_rules else []
    parts.append(task)
    if config.report_format:
        parts.append(prompts.REPORT_FORMAT)
    return "\n\n".join(parts)


def _specialist_messages(role: str, state: ScoutState, config: RunConfig) -> list[BaseMessage]:
    messages: list[BaseMessage] = [
        SystemMessage(_system_prompt(role, state, config)),
        HumanMessage(f"Scout {state['opponent']} for {state['my_team']}. Start by calling your tools."),
    ]
    previous_problems = state.get("problems", {}).get(role)
    if previous_problems:
        problem_list = "\n".join(f"- {p}" for p in previous_problems)
        messages.append(HumanMessage(prompts.RETRY_NOTE.format(problems=problem_list)))
    return messages


def _tools_for(role: str, config: RunConfig) -> tuple[dict, dict]:
    """(tools the model is shown, tools it is allowed to run)."""
    available = {name: t for name, t in ALL_TOOLS.items() if name not in config.removed_tools}
    shown = {name: available[name] for name in role_spec(role)["tools"] if name in available}
    allowed = shown if config.tool_allowlist else available
    return shown, allowed


def _call_tool(name: str, args: dict, allowed_tools: dict, role: str, config: RunConfig) -> dict:
    """Run one tool call and apply the experiment's output tweaks."""
    if name not in allowed_tools:
        return {"error": f"tool '{name}' is not available to the {role_spec(role)['title']}"}
    try:
        result = allowed_tools[name].invoke(args)
    except Exception as exc:  # bad arguments from the model: report back instead of crashing
        return {"error": f"{type(exc).__name__}: {exc}"}

    if not config.zone_labels:
        for zone in result.get("zones", []):
            for label_field in ZONE_LABEL_FIELDS:
                zone.pop(label_field, None)
    if config.injected_text:
        result["note"] = config.injected_text
    return result


def make_specialist(role: str, config: RunConfig) -> Callable[[ScoutState], dict]:
    """Build the graph node for one specialist: a tool-calling loop followed by a report."""
    shown_tools, allowed_tools = _tools_for(role, config)

    def specialist(state: ScoutState) -> dict:
        messages = _specialist_messages(role, state, config)
        llm = get_llm(config)
        llm_with_tools = llm.bind_tools(list(shown_tools.values()))
        evidence: list[str] = []
        trace: list[dict] = []

        for _ in range(MAX_TOOL_ROUNDS):
            reply = llm_with_tools.invoke(messages)
            messages.append(reply)
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                result = _call_tool(call["name"], call["args"], allowed_tools, role, config)
                result_json = to_json(result)
                if "error" not in result:
                    evidence.append(result_json)
                trace.append({"role": role, "tool": call["name"], "args": call["args"],
                              "found": result.get("found"), "error": result.get("error")})
                messages.append(ToolMessage(content=result_json, tool_call_id=call["id"]))
        else:
            # Still calling tools after MAX_TOOL_ROUNDS: ask for the report without tools.
            messages.append(HumanMessage("Stop calling tools and write your final report now."))
            reply = llm.invoke(messages)

        return {
            "reports": {role: strip_thinking(reply.content)},
            "evidence": {role: evidence},
            "attempts": {role: state.get("attempts", {}).get(role, 0) + 1},
            "trace": trace,
        }

    specialist.__name__ = role
    return specialist


# ---------------------------------------------------------------- other nodes
def input_guard(state: ScoutState) -> dict:
    my_team, opponent = check_matchup(state["my_team"], state["opponent"])
    return {"my_team": my_team, "opponent": opponent}


def make_validator(config: RunConfig) -> Callable[[ScoutState], dict]:
    def validator(state: ScoutState) -> dict:
        problems, ungrounded = {}, {}
        for role in config.roles:
            report = state["reports"].get(role, "")
            evidence = state["evidence"].get(role, [])
            problems[role] = validate_report(report, evidence)
            ungrounded[role] = len(ungrounded_numbers(report, evidence))
        return {"problems": problems, "validation_rounds": [{"problems": problems, "ungrounded": ungrounded}]}

    return validator


def make_router(config: RunConfig) -> Callable[[ScoutState], list[str] | str]:
    def route_after_validation(state: ScoutState) -> list[str] | str:
        """Send failing specialists back while they have retries left; otherwise go to the Head Coach."""
        if not config.enforce_validation:
            return "head_coach"
        to_retry = [
            role for role in config.roles
            if state["problems"][role] and state["attempts"].get(role, 0) <= config.max_retries
        ]
        return to_retry or "head_coach"

    return route_after_validation


def _head_coach_prompt(state: ScoutState, passed: list[str], failed: list[str], prior: dict | None) -> str:
    sections = [f"### {role_spec(role)['title']}\n{state['reports'][role]}" for role in passed]
    if failed:
        excluded = ", ".join(role_spec(role)["title"] for role in failed)
        sections.append(f"### Excluded (failed validation)\n{excluded}")

    memory_clause = ""
    if prior:
        memory_clause = (
            f"\nFor continuity, your previous validated plan for this matchup ({prior['time']}) was:\n"
            f"{prior['plan']}\nKeep what the new reports still support; change what they don't.\n"
        )
    return prompts.HEAD_COACH.format(
        my_team=state["my_team"],
        opponent=state["opponent"],
        reports="\n\n".join(sections),
        memory_clause=memory_clause,
    )


def make_head_coach(config: RunConfig) -> Callable[[ScoutState], dict]:
    def head_coach(state: ScoutState) -> dict:
        if config.enforce_validation:
            passed = [role for role in config.roles if not state["problems"][role]]
        else:
            passed = list(config.roles)
        failed = [role for role in config.roles if role not in passed]

        prior = memory.last_plan(state["my_team"], state["opponent"], config.memory_file) if config.use_memory else None
        prompt = _head_coach_prompt(state, passed, failed, prior)

        # The coach may only use numbers that reached it through the reports (or its last plan).
        evidence = [state["reports"][role] for role in passed]
        if prior:
            evidence.append(prior["plan"])

        llm = get_llm(config)
        plan = strip_thinking(llm.invoke([HumanMessage(prompt)]).content)
        problems = validate_report(plan, evidence) if passed else ["No specialist report passed validation."]

        if problems and passed and config.enforce_validation:  # one corrective pass
            retry_note = prompts.RETRY_NOTE.format(problems="\n".join(problems))
            plan = strip_thinking(llm.invoke([HumanMessage(prompt), AIMessage(plan), HumanMessage(retry_note)]).content)
            problems = validate_report(plan, evidence)

        if config.use_memory:
            memory.save_plan(state["my_team"], state["opponent"], plan, validated=not problems,
                             path=config.memory_file)
        return {"final_plan": plan, "plan_problems": problems}

    return head_coach


# ---------------------------------------------------------------- graph
def _add_specialist_edges(graph: StateGraph, config: RunConfig) -> None:
    roles = config.roles
    if config.mode == "sequential":
        # One after another; on a retry the specialist goes straight back to the validator.
        graph.add_edge("input_guard", roles[0])
        for role, next_role in zip(roles, roles[1:] + ["validator"]):
            def next_step(state: ScoutState, next_role: str = next_role) -> str:
                return next_role if next_role not in state.get("reports", {}) else "validator"
            graph.add_conditional_edges(role, next_step, list({next_role, "validator"}))
    else:
        for role in roles:
            graph.add_edge("input_guard", role)   # fan out: run in parallel
            graph.add_edge(role, "validator")     # fan in


def build_graph(config: RunConfig = RunConfig()):
    graph = StateGraph(ScoutState)
    graph.add_node("input_guard", input_guard)
    for role in config.roles:
        graph.add_node(role, make_specialist(role, config))
    graph.add_node("validator", make_validator(config))
    graph.add_node("head_coach", make_head_coach(config))

    graph.add_edge(START, "input_guard")
    _add_specialist_edges(graph, config)
    graph.add_conditional_edges("validator", make_router(config), [*config.roles, "head_coach"])
    graph.add_edge("head_coach", END)
    return graph.compile()


def run(my_team: str, opponent: str, focus_player: str | None = None,
        config: RunConfig = RunConfig()) -> ScoutState:
    """Scout `opponent` for `my_team` and return the final state (plan, reports, trace...)."""
    initial_state: ScoutState = {
        "my_team": my_team,
        "opponent": opponent,
        "focus_player": focus_player,
        "reports": {},
        "evidence": {},
        "problems": {},
        "attempts": {},
        "trace": [],
        "validation_rounds": [],
    }
    return build_graph(config).invoke(initial_state)
