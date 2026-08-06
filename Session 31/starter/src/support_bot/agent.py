from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from .schemas import SupportReply
from langchain.agents.middleware import PIIMiddleware

from .guardrails import topic_guard

from .llm import get_model
from .prompts import SUPPORT_SYSTEM_PROMPT
from .tools import lookup_order, check_refund_eligibility, create_ticket

middleware=[
    topic_guard,
    PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
    PIIMiddleware("email", strategy="redact", apply_to_input=True),
]

def build_agent():
    return create_agent(
        model=get_model(),
        tools=[lookup_order, check_refund_eligibility, create_ticket],
        system_prompt=SUPPORT_SYSTEM_PROMPT,
        response_format=SupportReply,
        checkpointer=InMemorySaver(),
    )