"""Generate notebooks/progress_demo.ipynb.

Usage:
    python scripts/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/progress_demo.ipynb
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat

NOTEBOOK_FILE = Path(__file__).resolve().parent.parent / "notebooks" / "progress_demo.ipynb"


def markdown(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(dedent(text).strip())


def code(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(dedent(text).strip())


CELLS = [
    markdown("""
        # Basketball Scouting & Game-Plan Coordinator - progress demo
        A LangGraph swarm of role-based agents (local `qwen3` 8B through Ollama) that reads
        **offline January 2026 NBA data** and produces a 3-key game plan for the head coach.
    """),
    code("""
        import json
        import sys
        import time
        from pathlib import Path

        sys.path.insert(0, str(Path.cwd().parent))

        from scout import memory
        from scout.data import load
        from scout.graph import SPECIALISTS, build_graph, run
        from scout.guardrails import check_matchup, validate_report
        from scout.tools import ALL_TOOLS
    """),

    markdown("## 1. Offline data (long-term knowledge) and load-time bounds checks"),
    code("""
        data = load()
        print("teams:", len(data.team_codes))
        print("games:", data.team_logs.GAME_ID.nunique())
        print("player-games:", len(data.player_logs))
        print("shots:", len(data.shots))
        print("games with rotation data:", data.rotations.GAME_ID.nunique())
        print("rows dropped by bounds checks:", data.dropped_rows)
    """),

    markdown("## 2. Tools - deterministic, return `found: False` instead of a guess"),
    code("""
        result = ALL_TOOLS["weakest_defenders"].invoke({"team": "DEN"})
        print(json.dumps(result, indent=1)[:900])
    """),
    code("""
        ALL_TOOLS["player_vs_opponent"].invoke({"player": "Shai Gilgeous-Alexander", "opponent": "DEN"})
    """),

    markdown("""
        ## 3. Guardrails
        Input validation, and the grounding check that rejects numbers not present in tool output.
    """),
    code("""
        try:
            check_matchup("OKC", "XYZ")
        except ValueError as error:
            print("Rejected:", str(error)[:90], "...")
    """),
    code("""
        evidence = [json.dumps(ALL_TOOLS["team_summary"].invoke({"team": "DEN"}))]
        made_up_report = (
            "FINDINGS:\\n- DEN score 109.8 points per game but Jokic averages 31.4 points "
            "in 52 minutes, shooting 104% at the rim."
        )
        for problem in validate_report(made_up_report, evidence):
            print("-", problem)
    """),

    markdown("""
        ## 4. The graph (orchestration)
        ![Project architecture](../docs/architecture.svg)
    """),
    code("print(build_graph().get_graph().draw_mermaid())"),

    markdown("## 5. End-to-end run: OKC scouting DEN"),
    code("""
        start = time.time()
        state = run("OKC", "DEN")
        print(f"finished in {time.time() - start:.0f}s")
    """),
    code("""
        print("Tool calls made by the agents:")
        for call in state["trace"]:
            print(f"  {call['role']:<22} {call['tool']:<20} {call['args']}  found={call['found']}")

        print()
        for role in SPECIALISTS:
            problems = state["problems"][role] or "none"
            print(f"{role:<22} attempts={state['attempts'][role]}  problems={problems}")
    """),
    code("""
        for role in SPECIALISTS:
            print(f"=== {role} ===")
            print(state["reports"][role], "\\n")
    """),
    code("""
        print(state["final_plan"])
        print("\\nPlan validation problems:", state["plan_problems"] or "none")
    """),

    markdown("## 6. Memory - plans persist and are fed back to the Head Coach next time"),
    code("""
        print(json.dumps(memory.last_plan("OKC", "DEN"), indent=1)[:600])
    """),
]


def main() -> None:
    notebook = nbformat.v4.new_notebook(
        cells=CELLS,
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    )
    NOTEBOOK_FILE.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, NOTEBOOK_FILE)
    print(f"wrote {NOTEBOOK_FILE}")


if __name__ == "__main__":
    main()
