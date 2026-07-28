# Session 31 · Lab 1 — Build a Chatbot with LangChain + Streamlit

**Teaching guide (for the instructor).** Duration: ~45 minutes, live coding.
No slides. We build the app together, one file at a time.

**Stack verified on 27 July 2026:** `langchain 1.3.14` · `langchain-openai 1.4.1` ·
`streamlit 1.60.0` · `uv 0.11.19` · OpenAI model `gpt-5.6-luna`.

---

## How to use this guide

Each step follows the same seven-part rhythm. Say the **What** and **Why** *before* you
type anything — the client should always know what is about to appear on screen and why
it needs to exist. Then type the code, then explain it, then run it.

| Part | What you do |
|---|---|
| 🎯 What | One sentence: the thing we are about to build |
| 💡 Why | The problem it solves — never skip this |
| ⌨️ Code | Type it live. Only this step's code, nothing more |
| 🔍 How it works | Line by line / component by component |
| ⚙️ Behind the scenes | What actually happens at runtime |
| 🧩 Architecture fit | Where this piece sits in the whole picture |
| ❓ Likely questions | Prepared answers for what they will ask |
| ✅ Checkpoint | Pause. Ask. Do not move on until they answer |

> **Before the session:** run `uv sync` once in the starter folder so the packages are
> already cached. A 90-second silent download in the middle of a live lesson kills momentum.
> Also put a working `OPENAI_API_KEY` in `.env` and confirm the app runs.

---

## Step 0 — Draw the target (3 min)

Do this on a whiteboard or in a scratch file. **Do not open the editor yet.**

```
   ┌──────────────────────────────────────────────┐
   │  Browser  ──  Streamlit  (app.py)            │   ← the UI layer
   │     st.chat_input  →  st.chat_message        │
   │     st.session_state  =  conversation memory │
   └───────────────────┬──────────────────────────┘
                       │  a string goes down, a string comes back
   ┌───────────────────▼──────────────────────────┐
   │  Chain  (chatbot.py)                         │   ← the logic layer
   │     prompt  |  model  |  parser              │
   └───────────────────┬──────────────────────────┘
                       │  HTTPS
   ┌───────────────────▼──────────────────────────┐
   │  OpenAI API                                  │   ← the reasoning layer
   └──────────────────────────────────────────────┘
```

Say out loud:

> "Three layers. Streamlit owns the *screen* and the *memory*. LangChain owns the *shape
> of the request*. OpenAI owns the *thinking*. Every bug you will ever hit in a chatbot
> lives in exactly one of these three boxes, and the whole point of splitting the code
> into modules is so you can tell which box instantly."

**🧩 Architecture fit:** every step in this lab fills in one part of this diagram. Point
back at it each time.

**✅ Checkpoint 0:** *"If the bot answers but forgets my name, which box is broken?"*
→ The Streamlit box (memory). *"If it answers rudely, which box?"* → The chain box
(the prompt).

---

## Step 1 — Create the project with `uv` (6 min)

### 🎯 What
Set up an isolated, reproducible Python project using `uv`.

### 💡 Why
Three reasons, in order of how much they will matter to the client:

1. **Reproducibility.** `uv.lock` pins every transitive dependency to an exact version.
   Six months from now this project still installs identically. `pip install -r
   requirements.txt` does not give you that.
2. **Speed.** `uv` resolves and installs in seconds, not minutes, because it is written
   in Rust and uses a global cache with hardlinks.
3. **One tool.** It replaces `pyenv` + `venv` + `pip` + `pip-tools`. `uv` will even
   download the right Python version for you.

### ⌨️ Code

We are starting from the `starter/` folder, which already has the skeleton. Open a
terminal in it and run:

```bash
uv sync
```

Then look at `pyproject.toml` — this is the file that made that work:

```toml
[project]
name = "support-bot"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "langchain>=1.3.14",
    "langchain-openai>=1.4.1",
    "streamlit>=1.60.0",
    "python-dotenv>=1.0.1",
]

[build-system]
requires = ["uv_build>=0.9,<0.12"]
build-backend = "uv_build"

[tool.uv.build-backend]
module-name = "support_bot"
module-root = "src"
```

Add a package later with:

```bash
uv add <package-name>        # adds to pyproject.toml AND installs AND updates the lock
```

### 🔍 How it works

- **`[project]`** is the standard `pyproject.toml` metadata block (PEP 621). Nothing
  uv-specific about it — Poetry, Hatch and pip all read it.
- **`dependencies`** are *ranges*, not exact pins. You declare "at least 1.3.14"; uv
  figures out the exact set that satisfies everything and writes it to `uv.lock`.
- **`requires-python = ">=3.11"`** — we need 3.11+ because we use the `str | None` union
  syntax and `Self` typing in later steps.
- **`[build-system]`** tells Python *how to build this project into a package*. We need
  this because our code lives in `src/support_bot/`, and we want `import support_bot` to
  just work from anywhere in the project.
- **`module-root = "src"`** is the "src layout". Your importable code is in `src/`, and
  everything else (tests, data, notebooks) is outside it. This prevents the classic bug
  where `import support_bot` accidentally picks up the *folder* instead of the
  *installed package*.

### ⚙️ Behind the scenes

When you run `uv sync`, in order:

1. uv reads `.python-version` (`3.12`). If that interpreter isn't on the machine, **uv
   downloads a standalone build of it** — no system Python is touched.
2. It creates `.venv/` in the project folder.
3. It resolves the full dependency graph — `langchain` pulls in `langchain-core`,
   `langgraph`, `pydantic`, and about 40 more.
4. It writes the exact resolved versions to `uv.lock`.
5. It installs them into `.venv/`, hardlinking from `~/.cache/uv` where possible, so the
   second project on the same machine installs almost instantly.
6. Because we have a `[build-system]`, it also installs **our own project in editable
   mode**. That is the step that makes `from support_bot.chatbot import build_chain`
   resolve.

`uv run <cmd>` runs `<cmd>` inside `.venv` without you activating anything. That is why
every command in this lab starts with `uv run`.

### 🧩 Architecture fit

This is the foundation under all three boxes in the Step 0 diagram. It doesn't appear in
the diagram itself — it's the ground the diagram stands on.

### ❓ Likely questions

**"Do I still need to `activate` the venv?"**
No. `uv run streamlit run app.py` handles it. You *can* activate it if you want your IDE
to find the interpreter — point your editor at `.venv/bin/python` (or
`.venv\Scripts\python.exe` on Windows).

**"What is the difference between `uv.lock` and `requirements.txt`?"**
`requirements.txt` is usually hand-written and lists what *you* asked for.
`uv.lock` is machine-generated and lists what you *got* — every package, every
sub-dependency, with hashes. Commit `uv.lock`. Never edit it by hand.

**"Why `src/` instead of just a folder at the root?"**
Because with a root-level folder, Python finds your code via the current working
directory, which means it works when you run from the project root and mysteriously
breaks everywhere else. The `src` layout forces you to install the package properly, so
imports behave the same in your editor, your tests, and production.

**"What if the client has no `uv`?"**
`pip install uv`, or on macOS/Linux `curl -LsSf https://astral.sh/uv/install.sh | sh`.

### ✅ Checkpoint 1

Ask:
1. *"What would break if we deleted `uv.lock` and re-ran `uv sync` a year from now?"*
   → You'd get newer versions; the app might break on an API change. The lock file is
   what makes the install reproducible.
2. *"Where did our own code get installed?"*
   → Into `.venv`, in editable mode, because of `[build-system]`.

---

## Step 2 — `config.py`: one place for settings and secrets (5 min)

### 🎯 What
A single module that loads the API key and model name from the environment and fails
loudly if something is missing.

### 💡 Why
Two rules that every production LLM app follows:

1. **Secrets never live in source code.** An API key in `app.py` is one `git push` away
   from being scraped off GitHub and used to run up a bill.
2. **Fail fast, fail clearly.** If the key is missing, we want a one-line error at
   startup — not a confusing `AuthenticationError` from deep inside the OpenAI SDK
   twenty seconds after the user types their first message.

### ⌨️ Code

First, from the terminal:

```bash
cp .env.example .env
```

Open `.env` and paste the real key. Then write the module:

```python
# src/support_bot/config.py
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.6-luna")
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "none")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Copy .env.example to .env and paste your key."
    )
```

### 🔍 How it works

| Line | What it does |
|---|---|
| `load_dotenv()` | Reads the `.env` file and copies each `KEY=value` line into `os.environ`. It walks *up* from the current working directory to find `.env`, and it does **not** overwrite variables that are already set — so a real environment variable in production always wins over the local file. |
| `Path(__file__).resolve().parents[2]` | `__file__` is `.../src/support_bot/config.py`. `.parents[0]` = `support_bot/`, `.parents[1]` = `src/`, `.parents[2]` = the project root. We compute this instead of using `os.getcwd()` so file paths work no matter which directory the app is launched from. |
| `os.getenv("MODEL_NAME", "gpt-5.6-luna")` | Read with a default. The client can switch models by editing `.env` — no code change. |
| `if not OPENAI_API_KEY: raise` | The fail-fast guard. This runs at *import* time, so the app dies the moment the module is first imported. |

### ⚙️ Behind the scenes

Environment variables are a process-level key/value store the OS hands to every program
it launches. `load_dotenv()` is a small convenience: it does not create real OS
variables, it just mutates the `os.environ` dictionary of *this* Python process. Child
processes started from here inherit them; nothing outside is affected.

`.env` is in `.gitignore`. Show them that line. Say: *"This is the single most important
line in the repo."*

### 🧩 Architecture fit

Bottom of the logic box. Every other module imports from here and nothing here imports
from anywhere else in our package — it is a **leaf module**. That direction of
dependency is deliberate: settings must never depend on business logic.

### ❓ Likely questions

**"Why not just `st.secrets`?"**
Streamlit has its own secrets system (`.streamlit/secrets.toml`). It works, but it ties
your configuration to Streamlit. Our `config.py` works from a script, a notebook, a test
suite, or a FastAPI server. Keep the config framework-agnostic.

**"Is `reasoning_effort` an OpenAI thing or a LangChain thing?"**
OpenAI. GPT-5.x models are reasoning models — they can spend hidden tokens "thinking"
before answering. The levels are `none`, `low`, `medium`, `high`. We use `none` because
customer-support replies are simple and we want low latency and low cost. Bump it to
`medium` and they will *see* the reply get slower and better.

**"What if I commit `.env` by accident?"**
Rotate the key immediately at platform.openai.com. Deleting the commit is not enough —
it lives in the git history and in every clone.

### ✅ Checkpoint 2

1. *"If I set `OPENAI_API_KEY` as a real environment variable AND it's in `.env`, which one wins?"*
   → The real environment variable. `load_dotenv()` does not overwrite.
2. *"Why does this file import nothing from our own package?"*
   → So there are no circular imports and settings stay independent of logic.

---

## Step 3 — `llm.py`: the model factory (6 min)

### 🎯 What
One function that returns a configured chat model. Everything else in the app calls this
function instead of constructing a model itself.

### 💡 Why
This is the **swap point** of the entire application. Today it's OpenAI. Next month it
might be Anthropic, or a local Ollama model, or Azure OpenAI. If model construction is
scattered across five files, that swap is a refactor. If it lives in one function, it's
a two-line diff.

### ⌨️ Code

```python
# src/support_bot/llm.py
from functools import lru_cache

from langchain_openai import ChatOpenAI

from .config import MODEL_NAME, OPENAI_API_KEY, REASONING_EFFORT


@lru_cache(maxsize=1)
def get_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        reasoning_effort=REASONING_EFFORT,
        timeout=30,
        max_retries=2,
    )
```

Sanity-check it before moving on:

```bash
uv run python -c "from support_bot.llm import get_model; print(get_model().invoke('Say hi in 3 words').text)"
```

### 🔍 How it works

- **`ChatOpenAI`** is LangChain's adapter around the OpenAI SDK. It implements the
  `BaseChatModel` interface — which means it exposes `.invoke()`, `.stream()`,
  `.batch()`, `.bind_tools()` and `.with_structured_output()`. Every LangChain chat model
  exposes exactly the same interface. *That* is the abstraction we are buying.
- **`timeout=30`** — without this, a hung request hangs your UI forever.
- **`max_retries=2`** — the SDK automatically retries on 429 (rate limit) and 5xx, with
  exponential backoff. Retrying a 401 would be pointless, and it doesn't.
- **`@lru_cache(maxsize=1)`** — "least recently used cache", size 1. The first call
  builds the object; every later call returns the identical object. Since `get_model()`
  takes no arguments, the cache key is always the same.

### ⚙️ Behind the scenes

Two things to make explicit here:

1. **Constructing `ChatOpenAI` makes no network call.** It just builds a client object
   holding your key and settings. The HTTPS request happens on `.invoke()` / `.stream()`.
   Show them: the sanity check above returns instantly if you remove `.invoke(...)`.

2. **Why `lru_cache` matters so much in Streamlit.** Streamlit re-runs your *entire*
   script top to bottom on every user interaction. Without the cache, every keystroke
   submission would build a brand-new HTTP client and connection pool. With it, one
   client is reused for the life of the process.

3. **`.text` vs `.content`.** In LangChain v1, a model returns an `AIMessage` whose
   `.content` may be a *list of content blocks* (text, reasoning, tool calls, images) —
   not a plain string. The `.text` property flattens that to the plain string. **Teach
   `.text`.** `print(msg.content)` printing `[{'type': 'text', ...}]` is the single most
   common "huh?" moment for people coming from LangChain 0.x.

### 🧩 Architecture fit

`llm.py` is the boundary between our code and the OpenAI box in the Step 0 diagram.
Nothing above this line knows we use OpenAI.

### ❓ Likely questions

**"Could I use `init_chat_model` instead?"**
Yes — `init_chat_model("openai:gpt-5.6-luna")` is a provider-agnostic factory that
parses a `"provider:model"` string. It is great for demos and config-driven apps. We use
`ChatOpenAI` directly here because it makes the provider-specific parameters
(`reasoning_effort`, `verbosity`) discoverable in your editor's autocomplete.

**"Why isn't `temperature` set?"**
GPT-5.x reasoning models don't accept `temperature` the way GPT-4 did. Reasoning effort
is the knob now. If you switch to a non-reasoning model, add `temperature=0.3` and drop
`reasoning_effort`.

**"How do I switch to Anthropic?"**
`uv add langchain-anthropic`, then change two lines in this file to
`ChatAnthropic(model="claude-sonnet-4-6", ...)`. Nothing else in the project changes.
That is the payoff for having a factory.

### ✅ Checkpoint 3

1. *"How many HTTP requests has our app made so far?"* → Zero (until you call `.invoke`).
2. *"I want to try `gpt-5.6-terra` instead. What do I edit?"* → One line in `.env`.
3. *"Name two other files I'd have to change if I hadn't written this factory."*

---

## Step 4 — `prompts.py`: the system prompt (4 min)

### 🎯 What
The instructions that define the bot's personality, scope and rules — kept in their own
module.

### 💡 Why
The system prompt is **product logic**, not plumbing. Product people will want to edit
it. Putting it in its own file means it can be reviewed, version-controlled and diffed
without anyone scrolling past HTTP client configuration.

### ⌨️ Code

```python
# src/support_bot/prompts.py
SUPPORT_SYSTEM_PROMPT = """You are Acme Support, a friendly customer-support assistant.

Rules:
- Only help with Acme orders, shipping, refunds, billing and product issues.
- If you do not know something, say so. Never invent an order, date or policy.
- Keep replies under 120 words.
- Acme's refund window is 30 days from the delivery date.
"""
```

### 🔍 How it works

Read the four rules aloud and name what each one is doing:

| Rule | Technique | Purpose |
|---|---|---|
| "Only help with…" | **Scoping** | Reduces off-topic answers |
| "If you do not know, say so. Never invent…" | **Anti-hallucination** | Gives the model explicit permission to say "I don't know" — models hallucinate partly because they infer that not answering is failure |
| "under 120 words" | **Format constraint** | Controls output length and cost |
| "refund window is 30 days" | **Knowledge injection** | A business fact the model cannot possibly know |

This is the direct application of Session 30's prompt-engineering material. Call that
back explicitly.

### ⚙️ Behind the scenes

The system prompt is prepended to *every single request* as a message with
`role: "system"`. It is not stored anywhere on OpenAI's side — it is re-sent, and re-billed,
on every turn. A 500-token system prompt on a 20-turn conversation is 10,000 input
tokens spent purely on instructions.

Models are trained to weight system-role content more heavily than user-role content.
That is exactly why a **prompt-injection** attack — a user typing "ignore your
instructions" — is a real threat: it's the user trying to win an argument against the
system message. We defend against that with guardrails in Lab 2.

### 🧩 Architecture fit

Top of the chain. It is the first thing the model reads on every turn.

### ❓ Likely questions

**"Why not put this in the `.env` file or a database?"**
You can, and mature products do — it lets you A/B test prompts without a deploy. For a
lab, a Python constant is honest and readable. Mention that prompt versioning is a real
production concern.

**"Does a longer prompt make it better?"**
Up to a point, then it gets worse. Long prompts dilute attention and the model starts
ignoring the middle. Prefer few, sharp, testable rules.

### ✅ Checkpoint 4

1. *"Which rule is doing the most work to prevent hallucination?"*
2. *"If a customer asks about a competitor's product, which rule should stop it — and do you trust it to?"*
   → The scoping rule; and no, not on its own. That's the honest answer, and it sets up
   Lab 2's guardrails perfectly.

---

## Step 5 — `chatbot.py`: the chain (7 min)

### 🎯 What
Compose prompt → model → parser into a single reusable object using LCEL.

### 💡 Why
Right now we have three loose parts. A chain glues them into **one object with one
interface**. That means the Streamlit layer can call `chain.stream(...)` and stay
completely ignorant of what's inside — we could add retrieval, routing or a second model
later and `app.py` would not change.

### ⌨️ Code

```python
# src/support_bot/chatbot.py
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .llm import get_model
from .prompts import SUPPORT_SYSTEM_PROMPT


def build_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPPORT_SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    return prompt | get_model() | StrOutputParser()
```

Test it from the terminal *before* touching the UI:

```bash
uv run python -c "
from support_bot.chatbot import build_chain
c = build_chain()
print(c.invoke({'input': 'Where is my order A-1044?', 'history': []}))
"
```

### 🔍 How it works

**`ChatPromptTemplate.from_messages([...])`** takes a list of message slots and produces
a message list at call time.

- `("system", SUPPORT_SYSTEM_PROMPT)` — a fixed system message.
- `MessagesPlaceholder("history")` — a **slot for a whole list of messages**. At call
  time you pass `history=[HumanMessage(...), AIMessage(...), ...]` and they get spliced
  in, in order. This one line is what turns a stateless Q&A box into a conversation.
- `("human", "{input}")` — a templated human message. `{input}` is filled from the dict
  you pass to `.invoke()`.

**`|` — the pipe operator.** This is LCEL, LangChain Expression Language. It is Python's
`__or__` operator, overloaded. `a | b` builds a `RunnableSequence` that feeds `a`'s
output into `b`'s input.

**`StrOutputParser()`** takes the `AIMessage` the model returns and yields the plain
string. Without it, `chain.invoke(...)` would hand `app.py` a message object it would
have to unwrap itself.

### ⚙️ Behind the scenes

Trace one `.invoke({"input": "...", "history": [...]})` call out loud:

1. The dict hits `prompt`. It fills `{input}` and splices `history` into the placeholder.
   Output: a `ChatPromptValue` holding an ordered list of messages.
2. That list hits `get_model()`. `ChatOpenAI` converts each `SystemMessage` /
   `HumanMessage` / `AIMessage` into `{"role": ..., "content": ...}` JSON, POSTs it to
   the OpenAI API, waits, and wraps the response in an `AIMessage`.
3. That `AIMessage` hits `StrOutputParser`, which returns `message.text`.

**The critical realisation, and you should say it in these words:**

> "The model is a pure function. It has no memory whatsoever. Every request re-sends the
> entire conversation from scratch. `MessagesPlaceholder` is not *giving* the model
> memory — it is *re-reading the transcript to the model out loud, every single turn*.
> That's why long conversations get slower and more expensive: the input keeps growing."

Because every piece here is a `Runnable`, the composed chain is *also* a `Runnable`. It
has `.invoke()` (one input → one output), `.stream()` (yields pieces as they arrive) and
`.batch()` (many inputs, concurrently). You get streaming for free — we use that in Step 8.

### 🧩 Architecture fit

This is the entire middle box in the Step 0 diagram. Note what is **not** here: the
conversation history itself. The chain declares a *slot* for history; it does not own
it. Ownership belongs to Streamlit, which is Step 7.

### ❓ Likely questions

**"Why is `build_chain` a function and not a module-level variable?"**
Two reasons. Import-time side effects (like reading env vars or building clients) make
modules hard to test and hard to import in a notebook. And a function can later take
arguments — `build_chain(model_name=...)` — without changing any caller's shape.

**"What happens if I forget to pass `history`?"**
`KeyError`. `MessagesPlaceholder("history")` is required. Make it optional with
`MessagesPlaceholder("history", optional=True)`. Demo the error — it's a two-second fix
and they'll remember it.

**"Can I put two models in one chain?"**
Yes. `prompt | model_a | StrOutputParser() | summarise_prompt | model_b |
StrOutputParser()` is a valid chain. That's how "draft then critique" pipelines are built.

### ✅ Checkpoint 5

1. *"Where is the conversation stored right now?"* → Nowhere. We pass an empty list.
2. *"What does the `|` operator actually build?"* → A `RunnableSequence`.
3. *"If I want to log every prompt before it's sent, where's the cleanest place to add that?"*
   → Inside the chain, as another Runnable step — not in `app.py`.

---

## Step 6 — `app.py`: a chat UI in twenty lines (8 min)

### 🎯 What
A working Streamlit chat interface — no memory yet, no streaming yet. One question in,
one answer out.

### 💡 Why
Get something on screen as fast as possible. Momentum matters in a live session, and a
visible bug is easier to explain than an invisible one. We will deliberately ship this
version **broken** (it forgets everything) and let the client discover the bug themselves.

### ⌨️ Code

```python
# app.py
import streamlit as st

from support_bot.chatbot import build_chain

st.set_page_config(page_title="Acme Support", page_icon="🎧")
st.title("🎧 Acme Support Assistant")


@st.cache_resource
def get_chain():
    return build_chain()


chain = get_chain()

if user_text := st.chat_input("How can I help?"):
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        reply = chain.invoke({"input": user_text, "history": []})
        st.markdown(reply)
```

Run it:

```bash
uv run streamlit run app.py
```

### 🔍 How it works

| Element | What it does |
|---|---|
| `st.set_page_config(...)` | Browser tab title and favicon. Must be the **first** Streamlit call in the script, or Streamlit raises an error. |
| `@st.cache_resource` | Streamlit's cache for *non-serialisable, long-lived objects* — clients, connections, models. The decorated function runs once per process; every re-run gets the same object back. |
| `st.chat_input(...)` | Renders the input box pinned to the bottom of the page. Returns `None` on a normal re-run, and returns the **string the user typed** on the run triggered by submission. |
| `:=` (walrus) | Assigns and tests in one expression. `if user_text := st.chat_input(...)` means "call it, store the result, and only enter the block if it's truthy." |
| `st.chat_message("user")` | A context manager that renders a chat bubble with the right avatar and alignment. Anything written inside the `with` block goes in the bubble. |
| `st.markdown(...)` | Renders text as markdown — so the model's `**bold**` and bullet lists display properly. |

### ⚙️ Behind the scenes

**This is the single most important concept in Streamlit, so slow down here.**

Streamlit has no callbacks and no component tree. Instead:

> Every time the user interacts with *any* widget, Streamlit re-runs `app.py` **from line 1
> to the last line**, top to bottom, in a fresh execution — and repaints the page with
> whatever that run produced.

Consequences, all of which they will trip over:

- Any plain Python variable you set is **destroyed** at the end of each run.
- Anything you don't re-draw **disappears** from the page.
- Expensive setup would repeat on every keystroke — which is exactly why we wrapped the
  chain in `@st.cache_resource`.

**Now demonstrate the bug.** Type:

> "Hi, my name is Marcus."

then:

> "What's my name?"

It won't know. Ask the room *why*. Two things are broken and you want them to name both:

1. We hard-coded `history=[]` — we never send the past to the model.
2. Even the *display* of the previous turn is gone, because the re-run only drew the
   newest message.

That's the perfect setup for Step 7.

### 🧩 Architecture fit

This fills the top box in the Step 0 diagram, minus memory.

### ❓ Likely questions

**"`st.cache_resource` vs `st.cache_data` — which do I use?"**
`cache_data` is for *values* (DataFrames, dicts, API results); it returns a copy each
time and pickles the result. `cache_resource` is for *things* (DB connections, model
clients); it returns the **same object**, unpickled, shared across sessions. A LangChain
chain is a thing → `cache_resource`.

**"Does every user get their own copy of the app?"**
Each browser session gets its own `st.session_state` and its own script runs, but they
all share the same Python process and the same `cache_resource` objects. That's the right
split: one shared model client, separate conversations.

**"Why does the whole page flicker?"**
It's re-running. Streamlit diffs the output and only repaints what changed, so on a
small app it's imperceptible. On a heavy app you'd reach for `st.fragment`.

### ✅ Checkpoint 6

1. *"How many times has `app.py` run since we started the server?"*
   → Once at startup, then once per message submitted.
2. *"If I put `counter = 0; counter += 1; st.write(counter)` at the top, what does it print after five messages?"*
   → Always `1`. That's the whole lesson.
3. *"Name the two separate reasons the bot forgot the name."*

---

## Step 7 — Memory with `st.session_state` (7 min)

### 🎯 What
Store the conversation in `st.session_state`, re-draw it on every run, and feed it back
into the chain.

### 💡 Why
This is the fix for the bug we just demonstrated — and it is the actual definition of a
chatbot. A chatbot is a stateless model plus a transcript you keep re-reading to it.

### ⌨️ Code

Replace the body of `app.py` below `get_chain()`:

```python
# app.py  (revised)
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from support_bot.chatbot import build_chain

st.set_page_config(page_title="Acme Support", page_icon="🎧")
st.title("🎧 Acme Support Assistant")


@st.cache_resource
def get_chain():
    return build_chain()


chain = get_chain()

# 1. Create the transcript once per browser session.
if "history" not in st.session_state:
    st.session_state.history = []

# 2. Re-draw the whole transcript on every run.
for message in st.session_state.history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.text)

# 3. Handle a new message.
if user_text := st.chat_input("How can I help?"):
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        reply = chain.invoke({
            "input": user_text,
            "history": st.session_state.history,
        })
        st.markdown(reply)

    # 4. Append AFTER the call, so the model never sees the current turn twice.
    st.session_state.history.append(HumanMessage(user_text))
    st.session_state.history.append(AIMessage(reply))
```

### 🔍 How it works

**Block 1 — initialise.** `st.session_state` is a dict-like object that Streamlit keeps
alive *across re-runs* for one browser session. The `if "history" not in ...` guard is
the standard idiom: without it, we'd reset the list to `[]` on every single run and be
right back where we started.

**Block 2 — re-draw.** Because the page is rebuilt from scratch each run, *we* are
responsible for painting the history. This loop is what makes old messages stay visible.

**Block 3 — the call.** Now `history` is the real transcript instead of `[]`. The chain
splices it into `MessagesPlaceholder("history")`.

**Block 4 — append after.** Order matters. If we appended the `HumanMessage` *before*
calling the chain, the model would receive the current question twice: once in the
history list and once in the `{input}` slot. It usually still works, and it usually
still confuses the model. Appending after is the clean invariant:

> `history` always contains **completed** turns only.

**Why `HumanMessage` / `AIMessage` instead of plain dicts?** Because
`MessagesPlaceholder` expects LangChain message objects, and because typed messages
carry more than text — IDs, tool calls, token usage. Using the real classes now means
nothing breaks when we add tools in Lab 2.

### ⚙️ Behind the scenes

`st.session_state` is a server-side dictionary keyed by a session ID that Streamlit
stores in a browser cookie. Facts worth stating explicitly:

- It lives **in the server's RAM**. Restart the app → every conversation is gone.
- It is **per browser tab**. Open a second tab and you get a second, independent
  conversation. Great for demoing multi-user isolation live.
- It is **not persistence**. For real persistence you write to Redis, Postgres, or — as
  we'll see in Lab 2 — a LangGraph checkpointer.

Now trace what the model actually receives on turn 3:

```
[ system  ] You are Acme Support...          ← always, every turn
[ human   ] Hi, my name is Marcus.           ← from history
[ ai      ] Hi Marcus! How can I help?       ← from history
[ human   ] Where is order A-1044?           ← from history
[ ai      ] Let me check...                  ← from history
[ human   ] What's my name?                  ← the {input} slot
```

Six messages sent to answer one three-word question. **That** is the cost model of
chatbots, and it is why every production system eventually needs trimming or
summarisation.

### 🧩 Architecture fit

Memory now lives in the top box, exactly where the diagram says. Note the deliberate
design decision: **the UI owns the state, the chain stays stateless.** Stateless chains
are trivially testable — you hand them a list, you assert on a string.

### ❓ Likely questions

**"What if the conversation gets too long for the context window?"**
Then you get an API error, or (worse) silently degraded answers. Three standard fixes:
**trim** (keep the last N messages), **summarise** (replace old turns with a summary), or
**retrieve** (embed old turns, pull back only what's relevant). LangChain ships
`SummarizationMiddleware` for the second — we'll see the middleware system in Lab 2.

**"Can I persist this to a database?"**
Yes. Serialise the messages (each has `.model_dump()`) and key them by a user ID.
Lab 2's `InMemorySaver` checkpointer is the same idea with a pluggable backend —
swap in `PostgresSaver` and conversations survive a restart.

**"Why does refreshing the browser wipe the chat?"**
A refresh starts a new session ID → a new empty `session_state`.

### ✅ Checkpoint 7

1. *"Why do we append to history AFTER calling the chain rather than before?"*
2. *"Open a second browser tab. Will it see this conversation?"* → No. Have them try it.
3. *"On turn 10, how many messages does the model receive?"* → 21 (1 system + 9 completed
   turns × 2 + the new question).

---

## Step 8 — Streaming (5 min)

### 🎯 What
Show the answer word-by-word as it's generated instead of after a three-second pause.

### 💡 Why
Pure UX, and it matters more than it sounds. Time-to-first-token is under a second;
time-to-full-answer can be five. Streaming makes the app *feel* five times faster
without changing the model, and it's the difference between "prototype" and "product."

### ⌨️ Code

Change exactly two lines inside the `with st.chat_message("assistant"):` block:

```python
    with st.chat_message("assistant"):
        reply = st.write_stream(
            chain.stream({
                "input": user_text,
                "history": st.session_state.history,
            })
        )
```

`st.markdown(reply)` is no longer needed — `write_stream` draws as it goes.

### 🔍 How it works

- **`chain.stream(...)`** returns a Python generator. Because every part of the chain is
  a `Runnable`, and `StrOutputParser` is streaming-aware, the generator yields **plain
  string fragments** — usually a token or two each.
- **`st.write_stream(generator)`** consumes the generator, appends each chunk to the page
  as it arrives, and — importantly — **returns the fully concatenated string** when the
  generator is exhausted. That returned value is what we store in `history`.

### ⚙️ Behind the scenes

Under the hood, `ChatOpenAI.stream()` sets `stream=True` on the API request. The response
is a **Server-Sent Events** stream: the connection stays open and OpenAI pushes
`data: {...}` frames as tokens are decoded. LangChain wraps each frame in an
`AIMessageChunk`; `StrOutputParser` pulls `.text` out of each one.

Streamlit's side of it is a WebSocket to the browser, so each chunk is pushed to the page
without a page re-run.

Costs are identical — same tokens, same price. This is a latency-*perception* win, not a
latency win.

### 🧩 Architecture fit

Same three boxes, but now the arrow between them is a stream rather than a single
request/response.

### ❓ Likely questions

**"Why do I need `write_stream`'s return value?"**
Because the generator is consumed exactly once. If you tried to loop over it again to
build the string for `history`, you'd get nothing. `write_stream` conveniently does both
jobs.

**"Can I stream a chain that has tool calls in it?"**
Yes, but the shape changes — you get tool-call chunks mixed with text chunks, and you
need `agent.stream(..., stream_mode=...)` to pick what you want. That's Lab 2.

**"What if the connection drops halfway?"**
You keep the partial text and get an exception. Production apps wrap this in try/except
and show a "something went wrong, retry?" affordance.

### ✅ Checkpoint 8

1. *"Is streaming faster, cheaper, or neither?"* → Neither. It's the same tokens and the
   same total time. It just *feels* faster.
2. *"What type does `chain.stream()` return?"* → A generator of strings.

---

## Step 9 — Polish: sidebar and reset (4 min)

### 🎯 What
A sidebar showing which model is running, the turn count, and a "New conversation" button.

### 💡 Why
It makes the invisible state visible. When the client can *watch* the turn counter climb
and then click reset and watch it drop to zero, `session_state` stops being an abstract idea.

### ⌨️ Code

Insert after `chain = get_chain()`:

```python
with st.sidebar:
    st.subheader("Session")
    st.caption(f"Model: `{MODEL_NAME}`")
    st.caption(f"Turns stored: {len(st.session_state.get('history', [])) // 2}")
    if st.button("🔄 New conversation", width="stretch"):
        st.session_state.history = []
        st.rerun()
```

And add the import at the top:

```python
from support_bot.config import MODEL_NAME
```

### 🔍 How it works

- **`with st.sidebar:`** — a container. Everything inside renders in the collapsible left
  panel.
- **`st.session_state.get('history', [])`** — defensive `.get()`, because on the very
  first run the sidebar renders before we've initialised `history` (unless you put the
  init block above it — which is the tidier fix; mention both).
- **`// 2`** — each turn is two messages (human + AI).
- **`st.button(...)`** returns `True` **only on the run triggered by the click**, then
  `False` on every subsequent run. Buttons are events, not toggles. This trips up
  everybody exactly once.
- **`st.rerun()`** immediately restarts the script from line 1. Without it, the page
  would finish the current run and still show the old messages, because the re-draw loop
  already ran further up the file.

### ⚙️ Behind the scenes

`st.rerun()` raises a special internal exception that Streamlit catches to abort the
current run and schedule a new one. That's why any code after it never executes — worth
saying, because it looks like a normal function call.

### 🧩 Architecture fit

Still the UI box. The reset button is the clearest possible demonstration that the
conversation lives in Streamlit's memory and nowhere else.

### ❓ Likely questions

**"Why doesn't the button stay 'pressed'?"**
It's an event, not state. If you need a toggle, use `st.toggle` or store a boolean in
`session_state`.

**"Can I make the sidebar show token usage / cost?"**
Yes — switch from `StrOutputParser` to reading the raw `AIMessage`, which carries
`.usage_metadata` with input and output token counts. Good homework.

### ✅ Checkpoint 9

1. *"What does `st.button` return two runs after I click it?"* → `False`.
2. *"Why is `st.rerun()` needed after clearing the history?"*

---

## Wrap-up (3 min)

### What we built

A working, modular customer-support chatbot:

```
app.py                    Streamlit UI · owns conversation state
src/support_bot/
  config.py               settings & secrets     (leaf — imports nothing of ours)
  llm.py                  model factory          (imports config)
  prompts.py              system prompt          (leaf)
  chatbot.py              LCEL chain             (imports llm + prompts)
```

Notice the dependency arrows all point **one way**, toward the leaves. That's what makes
this refactorable.

### The five ideas that matter

1. **The model is stateless.** Memory is something *you* implement by re-sending the
   transcript.
2. **Streamlit re-runs the whole script on every interaction.** `session_state` is the
   only thing that survives.
3. **LCEL composes `Runnables` with `|`,** and the composed object has the same
   interface as its parts — including free streaming.
4. **A model factory is the swap point.** One function to change providers.
5. **`.text`, not `.content`,** when you want a plain string out of a message in
   LangChain v1.

### What's still missing — the bridge to Lab 2

Ask the client to list the gaps before you tell them. They should get most of these:

- ❌ It can't **look anything up**. Ask it about order A-1044 and it will make something
  up or apologise. → **Tools.**
- ❌ The reply is an unstructured blob. We can't route, log or analyse it. → **Structured
  output.**
- ❌ Nothing stops a user typing "ignore your instructions", and nothing stops the bot
  promising a refund it can't honour. → **Guardrails.**

> "Right now we have a chatbot that can *talk*. In Lab 2 we make one that can *act*, and
> that we can *trust*."

---

## Instructor cheat-sheet

**Commands**

```bash
uv sync                              # install everything
uv add <pkg>                         # add a dependency
uv run streamlit run app.py          # launch
uv run python -c "..."               # quick REPL test inside the venv
```

**If something breaks mid-session**

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: support_bot` | project not installed editable | `uv sync` (needs `[build-system]` in `pyproject.toml`) |
| `RuntimeError: OPENAI_API_KEY is not set` | no `.env` | `cp .env.example .env`, paste key, restart |
| `AuthenticationError` | bad/expired key | new key at platform.openai.com |
| `KeyError: 'history'` | forgot to pass `history` to `.invoke()` | pass it, or `MessagesPlaceholder("history", optional=True)` |
| Output prints `[{'type': 'text'...}]` | used `.content` | use `.text` |
| `st.set_page_config` error | it isn't the first Streamlit call | move it to the top |
| UI shows nothing after a click | forgot `st.rerun()` after mutating state | add it |
| Bot forgets everything | `history=[]` still hard-coded | Step 7 |

**Timing**

| Step | Minutes | Cumulative |
|---|---|---|
| 0 · Draw the target | 3 | 3 |
| 1 · uv project | 6 | 9 |
| 2 · config.py | 5 | 14 |
| 3 · llm.py | 6 | 20 |
| 4 · prompts.py | 4 | 24 |
| 5 · chatbot.py | 7 | 31 |
| 6 · app.py (broken) | 8 | 39 |
| 7 · session_state memory | 7 | 46 |
| 8 · streaming | 5 | 51 |
| 9 · sidebar | 4 | 55 |
| Wrap-up | 3 | 58 |

**If you are running short:** cut Step 9 entirely and shorten Step 1 to `uv sync` plus a
30-second explanation. Never cut Step 7 — it is the point of the lab.
