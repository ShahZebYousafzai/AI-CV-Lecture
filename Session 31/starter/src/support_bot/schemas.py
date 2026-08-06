from typing import Literal, Sequence

from pydantic import BaseModel, Field

class SupportReply(BaseModel):
    """A structured reply from the Acme support assistant."""

    answer: str = Field(description="The message shown to the customer, plain language.")
    category: Literal[
        "order_status", "refund", "shipping", "billing", "technical", "other"
    ] = Field(description="What the customer is asking about.")
    sentiment: Literal["happy", "neutral", "frustrated"] = Field(
        description="How the customer appears to feel."
    )
    needs_human: bool = Field(
        description="True if a human agent should take over this conversation."
    )
    order_id: str | None = Field(
        default=None, description="Order ID mentioned by the customer, if any."
    )
