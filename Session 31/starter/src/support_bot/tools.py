import json

from langchain.tools import tool

from .config import DATA_DIR
from datetime import date

REFUND_WINDOW_DAYS = 30

def _orders() -> dict:
    print(DATA_DIR)
    return json.loads((DATA_DIR / "orders.json").read_text())


@tool
def lookup_order(order_id: str) -> dict:
    """Look up an Acme order by its ID (for example 'A-1043').

    Args:
        order_id: The customer's order ID.
    """
    order = _orders().get(order_id.strip().upper())
    if order is None:
        return {"found": False, "order_id": order_id}
    return {"found": True, "order_id": order_id.strip().upper(), **order}

@tool
def check_refund_eligibility(order_id: str) -> dict:
    """Check whether a delivered order is still inside Acme's 30-day refund window.

    Args:
        order_id: The customer's order ID.
    """
    order = _orders().get(order_id.strip().upper())
    if order is None:
        return {"eligible": False, "reason": "order_not_found"}
    if order["status"] != "delivered":
        return {"eligible": False, "reason": f"status_is_{order['status']}"}
    days = (date.today() - date.fromisoformat(order["delivered_on"])).days
    return {
        "eligible": days <= REFUND_WINDOW_DAYS,
        "days_since_delivery": days,
        "window_days": REFUND_WINDOW_DAYS,
    }


@tool
def create_ticket(order_id: str, issue: str, priority: str = "normal") -> dict:
    """Escalate to a human agent by opening a support ticket.

    Args:
        order_id: The order the issue relates to.
        issue: One-sentence description of the problem.
        priority: 'low', 'normal' or 'high'.
    """
    ticket_id = f"T-{abs(hash((order_id, issue))) % 90000 + 10000}"
    return {"ticket_id": ticket_id, "order_id": order_id,
            "issue": issue, "priority": priority, "status": "open"}
