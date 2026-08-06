# Session 33 · RAG Pipeline Lab I — Instructor Guide

**Phase 9 — RAG Systems · Curriculum session 69 · Fully practical, no slides**
Duration: 60–90 minutes, live coding in `Session33_RAG_Pipeline_Lab.ipynb`.

**Stack verified 29 July 2026:** `openai` · `gpt-5.6-luna` ($1 in / $6 out per 1M tokens) ·
`text-embedding-3-small` ($0.02 per 1M) · `faiss-cpu` · `pypdf 6.13` · `rank_bm25 0.2.2` ·
`tiktoken` · `sentence-transformers` with `cross-encoder/ms-marco-MiniLM-L-6-v2`.

---

## The one sentence for this session

> Session 32b showed that retrieval **works**. Session 33 shows what it takes to make it
> work on documents you did not write, and how to **prove** it works.

Everything in the notebook serves that second clause. If the client leaves knowing that
every RAG decision is a knob with a measurable effect — and that they now own a harness to
measure it — the session succeeded, even if you skip a step.

---

## Prep the day before — non-negotiable

| # | Do this | Why |
|---|---|---|
| 1 | `pip install openai numpy faiss-cpu tiktoken pypdf rank_bm25 sentence-transformers` | ~2 min of installs you do not want live |
| 2 | Run the whole notebook top to bottom | Step 6 downloads a ~90 MB reranker; `tiktoken` downloads its BPE file on first use |
| 3 | Confirm `OPENAI_API_KEY` is set and the Step 0 ping prints `1536` | The single most common live failure |
| 4 | Check arXiv is reachable from the machine you will demo on | Corporate networks sometimes block it — if so, pre-download the three PDFs into `rag_corpus/` and the cell will skip the fetch |
| 5 | Note your own Step 8 numbers | You will want to know whether the client's table matches yours |

Full-run cost: **5–8 cents**.

---

## Timing plan

| Block | Steps | Minutes | Cumulative |
|---|---|---|---|
| Get documents in | 0 · 1 · 2 | 30 | 0:30 |
| Find the right piece | 3 · 4 · 5 · 6 | 30 | 1:00 |
| Answer and prove it | 7 · 8 · 9 | 25 | 1:25 |
| Recap | 10 | 5 | 1:30 |

**If you are running short:**

- **Drop Step 5 (hybrid).** Step 8 still runs — the table just compares dense, dense+rerank.
  Cheapest cut, smallest loss.
- **Drop Drills 2 and 3 in Step 9.** Keep Drill 1; it takes 30 seconds.
- **Never drop Step 8.** It is the point of the session. If you are down to 15 minutes,
  jump from Step 7 straight to Step 8.

**If you are running long:** Step 8's exercise 1 (the chunk-size sweep) can be run live and
easily fills 15 minutes of genuinely useful discussion.

---

## Step-by-step: what to say, what to watch for

### Step 0 · Setup and the corpus — 5 min

**Land this:** the three papers are *about* RAG, so the client can check answers themselves.

**Watch for:** the "why not just paste it all in?" question. It will come, and it is the
best question of the session — `gpt-5.6-luna` has a million-token context, so "it doesn't
fit" is no longer true. The notebook has the full answer. The short version: cost per
question, latency, accuracy, and the fact that a real corpus is 50 million tokens, not 50
thousand.

---

### Step 1 · Ingestion — 10 min

**Land this:** ingestion damage is silent and it caps the quality of everything downstream.

**Do this:** print the raw page **before** you show `clean()`. Read the hyphen breaks aloud —
`down-\nstream`, `knowl-\nedge`, `Pre-\ntrained`. Ask "what does an embedding model do with
`knowl` and `edge` as separate words?" Then run `clean()` on the same slice so they see
the repair on identical text.

**Watch for:** clients who want to perfect the cleaner. Cut this off kindly — "we'll know
whether it's good enough in Step 8." That deferral *is* the lesson.

**Optional 30 seconds:** `print(reader.pages[1].extract_text()[:400])` — page 2 is Figure 1
and extracts as word soup. Good for showing that layout-heavy pages need a better loader.

---

### Step 2 · Chunking — 15 min · **the most important step**

**Land this:** the chunk is the unit of retrieval. It is the highest-leverage decision in the
pipeline and it is made before any clever component runs.

**Do this:** run the naive splitter and let the output do the work — chunk 0 ends
`...generating Jeopardy ques` and chunk 1 starts `tions conditioned on...`. Cut mid-word.
Ask: "which of these two chunks answers a question about Jeopardy generation?" (Neither.)
Then run the recursive splitter and show it ending on a full stop.

**Land the trade-off table** (too small vs too large) properly — draw it if you have a
whiteboard. Everything else in the session is downstream of it.

**Watch for:** "is 320 right?" Answer honestly: it is a defensible starting point backed by
2025–2026 benchmarks (256–512 tokens, 10–20% overlap), not a right answer. Point at Step 8.

**Do not** walk the recursion line by line unless asked. Explain the *idea* — try the most
natural boundary, fall back to cruder ones — and move on. It is the longest function in the
notebook and the least important to memorise.

---

### Step 3 · Indexing — 7 min

**Land this:** FAISS stores vectors, not text. Save the index and the chunk JSON together or
your citations silently point at the wrong paragraph.

**Do this:** say the sizing rule out loud — **N × dims × 4 bytes** — and do the million-chunk
arithmetic with them (6 GB). That is the number that decides whether they need a hosted
vector DB.

**Watch for:** this step is mostly Session 32b revision. Move briskly. The two new ideas are
batching-with-order-preservation and the `assert`.

---

### Step 4 · Retrieval — 6 min

**Land this:** three queries, three lessons. Paraphrase works beautifully; exact strings like
`9%-19%` do not.

**Do this:** make them predict which of the three queries will struggle *before* you run it.
The third one failing is what motivates Step 5, so let them feel the gap.

**Watch for:** disappointment at cosine scores of 0.3–0.5. Explain that OpenAI embeddings
live in a narrow band and only the *gap* between ranks matters, not the absolute number.

---

### Step 5 · Hybrid retrieval — 10 min

**Land this:** dense and sparse fail in **opposite** directions, so combining them is close to
free quality. It is the default in Elasticsearch, Weaviate, Qdrant and Azure AI Search.

**Do this:** the RRF formula is one line — write `1/(60 + rank)` on the board. Emphasise
*ranks, not scores*: BM25 returns unbounded numbers, cosine returns `[-1, 1]`, and RRF never
has to reconcile them.

Then run the comparison table and read it row by row. Dense wins some, BM25 wins others,
hybrid is rarely worst.

**Watch for:** "why 60?" — empirical, from the 2009 paper, near-universally adopted, tune it
once and forget. Do not let this eat five minutes.

---

### Step 6 · Reranking — 12 min

**Land this:** bi-encoders find, cross-encoders rank. The chunk's vector was computed before
the question existed; the cross-encoder actually reads the question.

**Do this:** draw the funnel — 150 → 25 → 4. Then do the arithmetic that explains why the
funnel exists: cross-encoding 150 chunks is ~1.5 s per query, and a million chunks is three
hours. Bi-encoders precompute; cross-encoders cannot. That asymmetry *is* the architecture.

The before/after rank table is the money shot. A chunk that jumps from rank 9 to rank 1 is a
chunk the LLM would never otherwise have seen.

**Watch for:** the model download if you skipped prep. Also flag `max_length=512` — pairs
longer than that get truncated, which is a quiet argument for the chunk size chosen in Step 2.

**Say once, clearly:** recall lost in stage one is lost forever. A reranker cannot promote a
chunk that was never retrieved.

---

### Step 7 · Prompt construction — 12 min

**Land this:** **retrieval finds; the prompt decides.**

**Do this:** run the YOLOv8 question. Retrieval still returns four chunks — it always does,
because nearest-neighbour search has no concept of "no match." The refusal comes entirely
from the system prompt. Ask which line produced it.

Then land the U-shaped curve: model accuracy against the *position* of the relevant passage
is U-shaped — evidence at the start or end is used, evidence in the middle is often ignored.
**More context is not more accuracy.** The paper saying so is in the corpus we just indexed,
which makes the point land harder than any slide would.

**Watch for:** trust in citations. Be straight — a model can attach `[S2]` to a claim from
`[S1]`. Mitigations: small chunks a human can check, and measuring citation faithfulness
(Session 37, RAGAS).

---

### Step 8 · Evaluation — 12 min · **the point of the session**

**Land this:** every knob today was an opinion until this cell ran.

**Do this, in order:**

1. Explain hit rate ("did we find it at all?") and MRR ("how near the top?").
2. Run the **label sanity check first** and say why: a question with zero matching chunks
   scores 0 for every configuration and silently drags all four averages down.
3. **Before running the ablation, ask them to predict the ordering.** This is the single
   highest-value 60 seconds in the session.
4. Run it. Read the table against their prediction.

**How to read the result:**

- Hybrid should match or beat both single methods on **hit rate** — it is a recall play.
- The reranker should move **MRR** most, with less change to hit rate. It reorders; it does
  not find new chunks. If hit rate is already 1.0, MRR is the only number left that can move.
- **If the reranker does not help, say so.** On ten fairly easy questions over 150 chunks the
  first stage is often good enough. Rerankers earn their keep on large, noisy corpora with
  many near-duplicates. Pretending otherwise teaches cargo-culting, which is the opposite of
  the session's goal.

**Watch for:** "ten questions isn't a benchmark." Agree immediately. Ten questions catches
*regressions*, which is most of the value. Aim for 50–200 real user questions, and the best
source is production logs — especially the refusals from Step 7.

**Calibration numbers, if they ask what "good" is:** recall@5 ≥ 0.80 on ordinary corpora,
MRR ≥ 0.6 feels right to users, NDCG@5 ≥ 0.8 on head queries with a reranker in place.

---

### Step 9 · Full pipeline + failure drills — 8 min

**Land this:** the whole session collapses to fifteen readable lines. The complexity was in
the *choices*.

The three drills, and what each is for:

| Drill | What they see | The lesson |
|---|---|---|
| 1 · Unanswerable question | A clean refusal | Correct behaviour, produced by the prompt |
| 2 · Grounding rule removed | A fluent, plausible, unsourced answer | You can no longer tell which answers came from your documents |
| 3 · Off-by-one chunk mapping | Right answer, **wrong citations**, no error | The most dangerous RAG bugs raise no exception |

Drill 2 is the one clients remember: the ungrounded model answers "Canberra" perfectly well,
and *that is the problem*. In a medical or legal deployment, "which of these came from our
documents?" is the entire product.

Drill 3 is the one engineers remember. Point back at the `assert` in Step 3 and note that it
is the only thing standing between them and that bug.

---

### Step 10 · Recap — 5 min

Walk the pipeline diagram once. Then the knobs table — the punchline is the line underneath
it: **none of these has a correct value; each has a measurable one.**

Set exercise 1 (the chunk-size sweep) as homework at minimum. It is the exercise that turns
the session's method into a habit.

---

## Prepared answers for the hard questions

> **"This is a lot of machinery. Isn't there a library that does all of it?"**
> Yes — LangChain, LlamaIndex, Haystack all ship this pipeline in about fifteen lines. You
> will use one of them in production. But every one of them exposes exactly the knobs we
> turned today, and picking values for those knobs is the job. A library gives you defaults;
> it cannot tell you whether the defaults suit your corpus.

> **"How big can this get before it breaks?"**
> `IndexFlatIP` is comfortable to roughly a million vectors on one machine (6 GB at 1536
> dims). Past that: IVF or HNSW (Session 32b), or a hosted vector DB. Chunking, hybrid
> retrieval, reranking, prompting and evaluation are all unchanged by scale — only the index
> is.

> **"What does this cost in production?"**
> Per question: one embedding call (~15 tokens, negligible) plus one chat call (~1,700 in,
> ~80 out) — about 0.2 cents on `gpt-5.6-luna`. Indexing is a one-off ~$0.001 per 50k tokens.
> The reranker is local and free. The real costs are re-indexing when documents change and
> the human time spent maintaining the golden set.

> **"Our documents are confidential."**
> Everything except the two OpenAI calls already runs locally. Swap `embed()` for a
> `sentence-transformers` bi-encoder and the chat model for a local one, and the pipeline is
> fully on-premises. The architecture does not change — which is a good argument for having
> built it from parts rather than from a framework.

> **"Do we still need RAG now that context windows are a million tokens?"**
> Yes, and the reasons have shifted rather than disappeared. Cost per question, latency, the
> U-shaped attention curve, and corpora that are orders of magnitude larger than any window.
> Long context and retrieval are complements: bigger windows mean you can afford to pass 10
> chunks instead of 4, not that you can skip retrieval.

---

## Common live failures and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `OpenAIError` on the Step 0 ping | Key missing from the environment | `export OPENAI_API_KEY=...` and **restart the kernel** |
| Step 0 hangs or 403s | arXiv blocked or rate-limiting | Pre-download into `rag_corpus/`; the cell skips existing files |
| `tiktoken` hangs on first use | It downloads the BPE file on first call | Needs network once; run it during prep |
| Step 6 stalls for a minute | Reranker model downloading | Prep step 2. If live, talk through the funnel diagram while it downloads |
| `AssertionError` in Step 3 | Chunks changed without re-embedding | Re-run Steps 2 → 3 in order |
| Step 8 shows a `LABEL PROBLEM` row | A golden needle no longer appears in any chunk (usually after changing chunk size) | Fix the needle, not the metric |
| Metrics all zero | Golden set mismatched with the corpus | Run the sanity-check cell first, always |

---

## Checkpoint questions, collected

Use these to keep the session a dialogue rather than a demo.

1. **Step 0** — "We have 2 MB of PDF and a question. What has to happen in between?"
2. **Step 1** — "We kept the page number on every record. Why bother?"
3. **Step 2** — "A number lives in a table on page 6. What does our chunker do to it?"
4. **Step 3** — "I save the index but forget the chunk JSON. What breaks, and when do I find out?"
5. **Step 4** — "Retrieval returned five chunks even for a question the corpus can't answer. Why?"
6. **Step 5** — "Which column of that table is best, and what does 'best' mean?"
7. **Step 6** — "That chunk moved from rank 9 to rank 1. What did the cross-encoder know that the embedding didn't?"
8. **Step 7** — "Which single line in `SYSTEM` produced the refusal?"
9. **Step 8** — "Change chunk size to 128 and this table changes. Which number would you look at?"
10. **Step 9** — "Which of the three failures takes longest to notice in production?"

---

## What the client should leave with

- A working RAG pipeline over real PDFs, on their own machine, ~15 lines end to end.
- A saved FAISS index plus chunk store they can reload without re-embedding.
- An evaluation harness and a golden set — the thing that makes the next change safe.
- Seven principles: the chunk is the unit of retrieval · ingestion damage is silent · dense
  and sparse fail oppositely · bi-encoders find and cross-encoders rank · more context is
  not more accuracy · retrieval finds and the prompt decides · every knob needs a number.

## Where this goes next

| Session | Topic | Builds on today |
|---|---|---|
| 34 | RAG Pipeline Lab II | Metadata filtering, query rewriting, multi-query, streaming |
| 35 | Advanced RAG | HyDE, parent-document retrieval, contextual compression |
| 36 | Multimodal RAG | Retrieval over figures and tables |
| 37 | RAG Evaluation (RAGAS) | Automated faithfulness and answer relevance — the generation half of Step 8 |
| 38 | Milestone — Full RAG System | The client's own documents, end to end |
