"""
agent6.py — THE AGENT. The agent is this Python orchestration loop, not the LLM.
The loop moves all data between the four cognitive roles, attaches all context,
tracks goal state, and decides termination. LLMs are stateless: context is
rebuilt from scratch every iteration (memory re-read, history + artifact refs
re-fed).

Loop shape:
  1. run_id, empty history, empty prior_goals.
  2. memory.remember(query)         — durable contract: a fact in the query persists.
  3. open MCP stdio session; load + format tools for Decision.
  4. for it in 1..MAX_ITERATIONS:
       hits  = memory.read(query, history)                 # read memory at TOP
       obs   = perception.observe(query, hits, history, prior_goals, run_id)
       prior_goals = obs.goals;  break if obs.all_done()
       goal  = obs.next_unfinished();  load attached bytes if a valid handle
       out   = decision.next_step(goal, hits, attached, history, tools)
       if out.is_answer: append 'answer' event (Perception decides done-ness next)
       else: result, art_id = await action.execute(session, out.tool_call)
             memory.record_outcome(...); append 'action' event
  5. return the final answer assembled from history.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE importing the agent modules: gateway.py / decision.py read
# AGENT_PROVIDER and MAX_ATTACH_CHARS at import time, so the file must be loaded
# first or those settings would be missed.
ROOT = Path(__file__).parent
STATE_DIR = ROOT / "state"
load_dotenv(ROOT / ".env")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from action import Action  # noqa: E402
from artifacts import ArtifactStore  # noqa: E402
from decision import Decision  # noqa: E402
from memory import Memory  # noqa: E402
from perception import Perception  # noqa: E402

DEFAULT_MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "12"))


def _rule(char: str = "-") -> str:
    return char * 72


async def run_agent(query: str, max_iterations: int = DEFAULT_MAX_ITERATIONS) -> str:
    run_id = uuid.uuid4().hex[:8]
    history: list[dict] = []
    prior_goals: list = []

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    memory = Memory(STATE_DIR)
    artifacts = ArtifactStore(STATE_DIR)
    perception = Perception(artifacts)
    decision = Decision()
    action = Action(artifacts)

    print(_rule("="))
    print(f"RUN {run_id}  |  max_iterations={max_iterations}")
    print(f"QUERY: {query}")
    print(_rule("="))

    # Durable contract: persist any fact stated inside the query.
    remembered = memory.remember(query, source="user_query", run_id=run_id)
    print(f"[memory] remembered query as kind={remembered.kind} "
          f"keywords={remembered.keywords}")

    server = StdioServerParameters(
        command="uv",
        args=["run", "python", "mcp_server.py"],
        cwd=str(ROOT),
        env=os.environ.copy(),
    )

    final_answer = ""
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_resp = await session.list_tools()
            tools = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema or {},
                }
                for t in tools_resp.tools
            ]
            print(f"[mcp] tools: {', '.join(t['name'] for t in tools)}")

            for it in range(1, max_iterations + 1):
                print(f"\n{_rule()}\nITERATION {it}")

                hits = memory.read(query, history)
                print(f"[memory.read] {len(hits)} hit(s): "
                      f"{[h.descriptor[:40] for h in hits]}")

                obs = perception.observe(query, hits, history, prior_goals, run_id)
                prior_goals = obs.goals
                for i, g in enumerate(obs.goals):
                    flag = "x" if g.done else " "
                    att = f"  attach={g.attach_artifact_id}" if g.attach_artifact_id else ""
                    print(f"[perception] goal {i} [{flag}] {g.text}{att}")

                if obs.all_done():
                    print("[perception] all goals done -> terminating")
                    break

                goal = obs.next_unfinished()
                if goal is None:
                    break

                attached = []
                if goal.attach_artifact_id and artifacts.exists(goal.attach_artifact_id):
                    meta = artifacts.get_meta(goal.attach_artifact_id)
                    text = artifacts.get_bytes(goal.attach_artifact_id).decode(
                        "utf-8", errors="replace"
                    )
                    attached = [(meta, text)]
                    print(f"[loop] attached {meta.id} ({meta.size_bytes} bytes) to goal")

                out = decision.next_step(goal, hits, attached, history, tools, query=query)

                if out.is_answer:
                    print(f"[decision] ANSWER: {out.answer[:200]}")
                    history.append({"type": "answer", "goal_id": goal.id, "text": out.answer})
                    if out.answer.strip():
                        final_answer = out.answer
                    continue

                tc = out.tool_call
                print(f"[decision] TOOL_CALL: {tc.name}({tc.arguments})")
                result, art_id = await action.execute(session, tc)
                print(f"[action] -> {result[:160]}"
                      + (f"  (artifact {art_id})" if art_id else ""))
                memory.record_outcome(tc, result, art_id, run_id, goal_id=goal.id)
                history.append({
                    "type": "action",
                    "tool": tc.name,
                    "arguments": tc.arguments,
                    "descriptor": result,
                    "artifact_id": art_id,
                })
            else:
                print("\n[loop] hit MAX_ITERATIONS without all goals done")

    # Assemble the final answer from history (last substantive answer event).
    if not final_answer:
        answers = [e["text"] for e in history if e.get("type") == "answer" and e["text"].strip()]
        final_answer = answers[-1] if answers else "(no answer produced)"

    print(f"\n{_rule('=')}\nFINAL ANSWER:\n{final_answer}\n{_rule('=')}")
    return final_answer


def main() -> None:
    ap = argparse.ArgumentParser(description="EAG V3 Session 6 agent loop")
    ap.add_argument("query", help="the user query (quote it)")
    ap.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERATIONS)
    args = ap.parse_args()
    asyncio.run(run_agent(args.query, max_iterations=args.max_iters))


if __name__ == "__main__":
    main()
