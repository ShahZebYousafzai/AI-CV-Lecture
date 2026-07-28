# Session 31 — Customer Support Chatbot (starter)

Empty scaffold for Session 31. Every `.py` file under `src/support_bot/` is intentionally
blank — we fill them in together, step by step, during the session.

## One-time setup

```bash
uv sync            # creates .venv and installs everything from pyproject.toml
cp .env.example .env
# then open .env and paste your real OPENAI_API_KEY
```

## Run

```bash
uv run streamlit run app.py
```

## Layout

```
app.py                      Streamlit UI (the "front end")
data/orders.json            fake order database our tools read from
src/support_bot/
  config.py                 env loading + settings
  llm.py                    model factory
  prompts.py                system prompt
  chatbot.py                Lab 1 — prompt | model | parser chain
  tools.py                  Lab 2 — @tool functions
  schemas.py                Lab 2 — Pydantic response schema
  guardrails.py             Lab 2 — middleware guardrails
  agent.py                  Lab 2 — create_agent wiring
```

## ⚠️ Before each run of the session

`data/orders.json` is **date-sensitive**. The refund demo needs:

- `A-1043` delivered **~20 days ago** → still inside the 30-day refund window
- `A-1045` delivered **~60 days ago** → outside the window

Open `data/orders.json` and shift the `delivered_on` dates accordingly, otherwise both
orders come back "not eligible" and the demo loses its contrast.
