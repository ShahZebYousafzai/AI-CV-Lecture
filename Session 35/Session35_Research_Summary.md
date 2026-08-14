# Session 35 — Research Summary
### AI Agents and LangGraph Fundamentals

*Prepared 12 August 2026. All API claims below were verified by installing and running the current
packages, not taken from tutorials.*

**Verified environment (fresh install, today):**

| Package | Version |
|---|---|
| `langgraph` | 1.2.11 |
| `langgraph-checkpoint` | 4.2.0 |
| `langgraph-prebuilt` | 1.1.0 |
| `langchain-core` | 1.5.4 |

---

## 1. Key concepts that should be taught

### Part 1 — AI Agents (theory)

**What separates an agent from an LLM app.** A normal LLM application has a path decided by the
programmer: prompt in, text out, maybe a fixed chain of two or three calls. An agent decides its own
path at runtime. The control flow is data, not code. LangChain's own docs make exactly this split:
*"Workflows have predetermined code paths and are designed to operate in a certain order. Agents are
dynamic and define their own processes and tool usage."* That single sentence is the cleanest framing
available and I'd build the whole first section around it.

Core components worth naming: a **model** (the reasoning engine), **tools** (its hands), **state /
memory** (what it has seen so far), and a **loop** (the thing that keeps calling the model until it's
done). Perception → reasoning → action → feedback maps onto: read state → model call → tool call →
observation written back to state.

**ReAct.** From Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*
(arXiv:2210.03629, ICLR 2023). The paper's own framing is the teachable one: reasoning traces help
the model *induce, track and update action plans and handle exceptions*, while actions let it
*interface with external sources to gather additional information*. Two failure modes motivated it —
chain-of-thought alone hallucinates because it never checks anything against the world, and
act-only agents flail because they never reason about what they just saw. Interleaving fixes both.
Evaluated on HotPotQA, FEVER, ALFWorld and WebShop.

Teach the loop as **Thought → Action → Observation → Thought → …** and be explicit that in 2026 this
is no longer a prompt format you write by hand — it's native tool calling. The name survived; the
mechanism changed. Worth saying out loud, because most tutorials still show the text-parsing version.

**Tools.** A tool is a function plus a description the model can read. The model doesn't execute
anything — it emits a structured request (`{"name": "multiply", "args": {"a": 3, "b": 4}}`) and your
code runs it and hands the result back as an observation. That separation is the single most
misunderstood point for beginners and is worth 90 seconds on its own. Examples: search, calculator,
SQL, HTTP APIs, code execution, file I/O.

**Planning.** Single-step (decide the next action only, ReAct-style) vs. multi-step
(plan-and-execute: write the whole plan first, then run it). Trade-off: plan-first is cheaper and
more auditable but brittle when the world doesn't match the plan; step-by-step adapts but drifts and
costs more. Re-planning is the middle ground. Planning and tool use are coupled — a plan is only as
good as the tools available to execute it.

**Agent loops and their failure modes.** Observe → Reason → Act → Observe. It stops when the model
stops asking for tools. Failure modes to name: infinite loops, wrong tool selection, hallucinated
tool arguments, and the fact that a crash halfway through loses everything. **This last one is the
bridge into LangGraph** — the honest motivation for the framework is that a `while` loop gives you no
persistence, no visibility, no resume, and no place for a human to intervene.

### Part 2 — LangGraph

**What it is.** LangGraph models agent workflows as a graph. The docs' own one-liner is worth quoting
verbatim on a slide: ***"nodes do the work, edges tell what to do next."*** Three primitives:

- **State** — a shared data structure, a snapshot of the application right now. Usually a `TypedDict`.
- **Nodes** — plain Python functions: take state, return a partial state update. May or may not contain an LLM call.
- **Edges** — determine which node runs next. Static or conditional.

Underneath it's a message-passing / Pregel-style engine that runs in discrete **super-steps**. Worth
one sentence only, but it explains why checkpoints appear where they do.

**Why graphs.** Loops and branches become first-class and inspectable instead of buried in control
flow. And because the runtime owns the loop, it can checkpoint every step, stream, and pause.

**State updates and reducers.** By default a returned key **overwrites** the existing value. Annotate
a key with a reducer (`Annotated[list[str], add]`) and it **accumulates** instead. Beginners trip on
this constantly — budget time for it. It is also the mechanism behind `add_messages`.

**Checkpointing.** A checkpointer saves a snapshot of state at each super-step, grouped into
**threads**. Terminology to get right: *checkpointer* (the saver object), *checkpoint* (one snapshot,
a `StateSnapshot`), *thread* (`thread_id`, the conversation/run identity), *super-step* (one tick of
the graph). Enables: conversational memory, human-in-the-loop, time travel / replay, and
fault-tolerance via pending writes.

---

## 2. Current APIs to use

```python
from typing import Annotated, TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    question: str
    steps: Annotated[list[str], add]      # reducer: appends
    answer: str

builder = StateGraph(State)
builder.add_node("think", think)          # node name inferred from fn if omitted
builder.add_edge(START, "think")
builder.add_conditional_edges("act", route, {"retry": "think", "done": END})

graph = builder.compile(checkpointer=InMemorySaver())
graph.invoke(inputs, {"configurable": {"thread_id": "1"}})
graph.get_state(config)                   # -> StateSnapshot
graph.get_state_history(config)           # -> newest-first list
```

Also current and safe to mention: `MessagesState`, `add_messages`, `ToolNode`,
`graph.get_graph().draw_mermaid_png()`, `graph.stream(...)`, `SqliteSaver` / `PostgresSaver`,
`interrupt()` + `Command(resume=...)` for human-in-the-loop.

I ran the full state → nodes → conditional-loop → checkpointer example on 1.2.11. It works, produces
6 checkpoints for a 2-iteration loop, and `get_state().next == ()` on completion.

## 3. Outdated APIs to avoid

| Avoid | Use instead | Why |
|---|---|---|
| `from langgraph.prebuilt import create_react_agent` | `from langchain.agents import create_agent` | Deprecated in LangGraph v1 in favour of LangChain's `create_agent` (which runs on LangGraph and adds middleware). Still importable — so tutorials still show it — but don't teach it. |
| `MemorySaver` | `InMemorySaver` | `MemorySaver` is the legacy alias; docs use `InMemorySaver` throughout. |
| `builder.set_entry_point("x")` / `set_finish_point` | `add_edge(START, "x")` / `add_edge("x", END)` | Both still work; `START`/`END` is what current docs teach and it reads better on a slide. |
| Hand-written `Thought:/Action:/Observation:` prompt parsing | native tool calling / `bind_tools` | The ReAct *pattern* is current; the ReAct *prompt format* is not how anyone builds this now. |
| `langgraph.checkpoint.sqlite` assumed bundled | `pip install langgraph-checkpoint-sqlite` | Ships separately. |
| `pip install langchain` implied by `langgraph` | separate install | Confirmed: a bare `langgraph` install does **not** provide `langchain.agents`. Matters for the notebook's install cell. |

One more trap: `graph.get_graph().draw_ascii()` needs `grandalf`, and `draw_mermaid_png()` calls a
remote renderer. For a live session, `draw_mermaid()` (prints text, no network) is the safe fallback
if the room's wifi is bad.

## 4. Recommended practical example

**A two-node "research assistant" that grades its own answer and retries.**

State: `question`, `notes` (reducer, accumulates), `answer`, `attempts`, `verdict`.
Nodes: `research` (looks something up) → `answer` (drafts) → `grade` (checks) → conditional edge back
to `research` or to `END`.

Why this one over the obvious alternatives:

- It produces a **genuine loop**, so the conditional edge does real work rather than being decorative.
- It mirrors the agent loop taught in Part 1 — Observe → Reason → Act → Observe — so the code is the
  diagram made concrete.
- The retry counter gives a natural hook for "how does an agent know when to stop" and for the
  infinite-loop failure mode.
- Checkpointing has something interesting to show: `get_state_history()` on a graph that looped twice
  is far more instructive than on a straight line.
- **Critically: the core of it needs no API key.** Nodes are plain Python. That means the notebook
  runs offline, in any room, at any wifi quality — and it makes the point that *a node is just a
  function* far better than an LLM call would. I'd add the LLM as a clearly-marked optional cell at
  the end rather than as a dependency.

I'd avoid a tool-calling agent as the main example. It's the more impressive demo, but it puts
message formats, tool schemas and API keys between the audience and the four concepts you actually
want them to leave with.

## 5. Recommended 45-minute structure

| # | Segment | Min | Running |
|---|---|---|---|
| 1 | Intro & where this sits in the course | 2 | 2 |
| 2 | What is an agent (vs. LLM app) | 4 | 6 |
| 3 | ReAct | 4 | 10 |
| 4 | Tool use | 3 | 13 |
| 5 | Planning | 3 | 16 |
| 6 | Agent loops & failure modes | 3 | 19 |
| 7 | **Why we need orchestration** (the bridge) | 2 | 21 |
| 8 | LangGraph: what & why, State / Nodes / Edges / StateGraph / checkpointing | 9 | 30 |
| 9 | **Notebook, live** | 12 | 42 |
| 10 | Recap + Q&A | 3 | 45 |

Theory total 21 min, LangGraph theory 9, practical 12. This is slightly tighter on agent theory than
your draft allocation (15–18 → 17 including the bridge) because the practical needs 12 real minutes
and live demos always overrun.

**Cut or defer:**

- *Multi-agent / supervisor patterns* → Session 36. Mentioned in one line, not taught.
- *Human-in-the-loop with `interrupt()`* → shown as a **named consequence** of checkpointing on a
  slide, not demoed. It needs its own 15 minutes to do properly.
- *Streaming, `Send` / map-reduce, subgraphs, `Command`* → out of scope entirely; one "here's what
  else exists" slide at the end.
- *Plan-and-execute as code* → concept only. No implementation.
- *Postgres/SQLite checkpointers* → one sentence ("swap one line for production"), no demo.

**Fragile bits to rehearse:** the notebook's install cell (do it before the session), and
`draw_mermaid_png()` (needs network — have `draw_mermaid()` ready as backup).

---

## Sources

- [LangGraph Graph API overview — docs.langchain.com](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [LangGraph Persistence — docs.langchain.com](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph Checkpointers — docs.langchain.com](https://docs.langchain.com/oss/python/langgraph/checkpointers)
- [Workflows and agents — docs.langchain.com](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [`StateGraph` API reference — reference.langchain.com](https://reference.langchain.com/python/langgraph/graph/state/StateGraph)
- [`create_react_agent` reference (deprecation notice)](https://reference.langchain.com/python/langgraph.prebuilt/chat_agent_executor/create_react_agent)
- [LangGraph v1 migration guide](https://docs.langchain.com/oss/python/migrate/langgraph-v1)
- [LangChain & LangGraph reach v1.0 — LangChain blog](https://www.langchain.com/blog/langchain-langgraph-1dot0)
- [Yao et al. (2023), *ReAct: Synergizing Reasoning and Acting in Language Models*, arXiv:2210.03629](https://arxiv.org/pdf/2210.03629)
- [ReAct — OpenReview (ICLR 2023)](https://openreview.net/pdf?id=WE_vluYUL-X)
