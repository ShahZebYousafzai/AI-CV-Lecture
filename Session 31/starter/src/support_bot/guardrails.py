from typing import Any

from langchain.agents.middleware import AgentState, before_agent
from langchain.messages import AIMessage
from langgraph.runtime import Runtime

from .schemas import SupportReply

BANNED_PATTERNS = ["ignore previous", "ignore your", "system prompt", "jailbreak"]

REFUSAL = SupportReply(
    answer=(
        "I can only help with Acme orders, shipping, refunds and billing. "
        "Could you rephrase your question?"
    ),
    category="other",
    sentiment="neutral",
    needs_human=False,
)


@before_agent(can_jump_to=["end"])
def topic_guard(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Deterministic input guardrail: stop prompt-injection and off-scope requests."""
    messages = state["messages"]
    if not messages:
        return None

    last = messages[-1]
    if last.type != "human":
        return None

    text = last.text.lower()
    if any(pattern in text for pattern in BANNED_PATTERNS):
        return {
            "messages": [AIMessage(REFUSAL.answer)],
            "structured_response": REFUSAL,
            "jump_to": "end",
        }
    return None