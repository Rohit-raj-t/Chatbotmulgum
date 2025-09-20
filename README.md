# Unified Bot API (emobot + careerbot)

A lightweight FastAPI service that exposes a single chat API for two personas:

* **emobot** — a compassionate mental-health companion tailored for Indian teens and college students.
* **careerbot** — a practical, culturally-aware career guidance assistant for Indian school and college students.

The service integrates `mem0` for user memory, an optional Chroma vector knowledge base, optional DuckDuckGo (DDGS) web search for freshness queries, and an LLM provider (default: Ollama). It aims to be easy to run locally for demos and research while handling sensitive content responsibly.

---

## Features

* Single `/chat` endpoint with optional path-based persona (`/chat/{bot_type}`).
* Per-request LLM override via `metadata.llm_model`.
* Shared mem0 memory (messages and assistant replies are stored, tagged by persona).
* Optional Chroma knowledge-base lookup for `careerbot` (school/college collections).
* DuckDuckGo (via `ddgs`) web search used when `careerbot` needs "freshness" (e.g., exam dates, rankings).
* Ollama-compatible generator by default (`OLLAMA_API_URL`).
* Helpful development endpoints to inspect and manage memories.

---

## Quickstart

### Prerequisites

* Python 3.10+
* `mem0ai` (required) — the code will raise if `mem0ai` is not importable.
* (Optional) `chromadb` if you want to use Chroma manually.
* (Optional) `ddgs` for DuckDuckGo web search.
* (Optional) `ollama` running locally or another LLM provider reachable via HTTP.

### Install

```bash
# create virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

> `requirements.txt` should contain:
>
> ```text
> fastapi
> uvicorn[standard]
> httpx
> python-dotenv
> pydantic
> mem0ai
> chromadb
> ddgs
> ollama
> ```

### Environment

Copy the provided `.env` file (example shown later) and edit as needed. Then run:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The default server is available at `http://localhost:8000`.

---

## Configuration (`.env`)

A list of environment variables supported by the app and their defaults (the project contains a sample `.env`):

* `LLM_PROVIDER` — default: `ollama`.

* `LLM_MODEL` — default: `gemma3:4b` (used if no persona-specific model is provided).

* `LLM_MODEL_EMO` — optional per-persona override for **emobot**.

* `LLM_MODEL_CAREER` — optional per-persona override for **careerbot**.

* `LLM_TEMPERATURE` — default: `0.1`.

* `LLM_MAX_TOKENS` — default: `2000`.

* `EMBEDDER_PROVIDER` — default: `ollama`.

* `EMBEDDER_MODEL` — default: `nomic-embed-text:latest`.

* `VECTOR_STORE_PROVIDER` — default: `chroma`.

* `VECTOR_STORE_PATH` — default: `db_chroma`.

* `VECTOR_STORE_COLLECTION` — default: `mental_health_memories`.

* `VECTOR_STORE_SCHOOL_COLLECTION` — default: `career_school_guidance`.

* `VECTOR_STORE_COLLEGE_COLLECTION` — default: `career_college_guidance`.

* `OLLAMA_API_URL` — default: `http://localhost:11434/api/generate`.

* `DUCKDUCKGO_MAX_RESULTS` — default: `5`.

* `WEB_SEARCH_REGION` — default: `in-en`.

* `APP_HOST` — default: `0.0.0.0`.

* `APP_PORT` — default: `8000`.

* `DEBUG` — default: `false`.

**Example `.env` (already included in the repo):**

```ini
LLM_PROVIDER=ollama
LLM_MODEL=gemma3:4b
LLM_MODEL_EMO=
LLM_MODEL_CAREER=
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2000

EMBEDDER_PROVIDER=ollama
EMBEDDER_MODEL=nomic-embed-text:latest

VECTOR_STORE_PROVIDER=chroma
VECTOR_STORE_PATH=db_chroma
VECTOR_STORE_COLLECTION=mental_health_memories
VECTOR_STORE_SCHOOL_COLLECTION=career_school_guidance
VECTOR_STORE_COLLEGE_COLLECTION=career_college_guidance

OLLAMA_API_URL=http://localhost:11434/api/generate

DUCKDUCKGO_MAX_RESULTS=5
WEB_SEARCH_REGION=in-en

APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=false
```

---

## How the service decides behavior

* **Persona selection**: `bot` in the request body, path (`/chat/careerbot`) or `metadata.bot` determines persona. Default is `emobot`.
* **Model selection**: Per-request `metadata.llm_model` overrides; otherwise persona-specific env var (`LLM_MODEL_EMO` or `LLM_MODEL_CAREER`) is used if set, falling back to `LLM_MODEL`.
* **Memory**: All incoming messages are appended to mem0 under `user_id`. Assistant replies are also stored.
* **Knowledge base**: For `careerbot`, the app will attempt a Chroma query against `career_school_guidance` or `career_college_guidance` based on `metadata.stage` (`school`/`college`).
* **Freshness / web search**: If `needs_freshness()` detects freshness-related keywords in the user query OR `metadata.require_latest` is set, and `ddgs` is installed, a DuckDuckGo search is performed and included when building the prompt for `careerbot`.

---

## API

### `POST /chat`

Single unified chat endpoint. Body: `ChatRequest`.

### `POST /chat/{bot_type}`

Same as `/chat` but forces `bot_type` (e.g. `/chat/careerbot`).

### Memory endpoints

* `GET /memories/user/{user_id}` — returns all stored memories for `user_id`.
* `GET /memories/search?user_id=<>&query=<>` — search memory for a user.
* `DELETE /memories/user/{user_id}` — delete all memories for `user_id`.

### Misc

* `GET /` — basic service status
* `GET /health` — health check

---

## Request / Example

**emobot example**

```bash
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{
  "user_id": "user123",
  "bot": "emobot",
  "messages": [
    {"role": "user", "content": "I am feeling very stressed about exams."}
  ]
}'
```

**careerbot example (requests latest info)**

```bash
curl -X POST "http://localhost:8000/chat/careerbot" -H "Content-Type: application/json" -d '{
  "user_id": "user123",
  "bot": "careerbot",
  "messages": [
    {"role": "user", "content": "Which engineering colleges have good CS programs?"}
  ],
  "metadata": {"stage": "college", "require_latest": true}
}'
```

### Request model (JSON)

```json
{
  "user_id": "<string>",
  "messages": [{"role": "user|assistant|system", "content": "<text>"}],
  "metadata": { /* optional */ },
  "bot": "emobot|careerbot"
}
```

Response model:

```json
{ "reply": "<assistant text>" }
```

---

## Vector store (Chroma) and mem0 notes

* The app initializes `mem0` with an LLM, embedder and a vector store configuration. When `VECTOR_STORE_PROVIDER=chroma`, mem0 is expected to manage Chroma internally.
* If `chromadb` is installed and `VECTOR_STORE_PROVIDER` is not `chroma`, the code tries to create a manual `chroma.PersistentClient` using `VECTOR_STORE_PATH`.
* The repo includes separate collections for mental-health memories and two career guidance collections (`career_school_guidance`, `career_college_guidance`). Populate them with documents to make `careerbot` KB lookups useful.

---

## Freshness and web search

* `careerbot` will perform a DuckDuckGo search when either the query contains freshness keywords (`latest`, `rank`, `exam date`, `results`, `cutoff`, year tokens like `2024`, `2025`, etc.) or when `metadata.require_latest` is true.
* Web results (if `ddgs` available) are included in the prompt and the assistant is instructed to indicate when facts were fetched live and include source URLs.

---

## Safety & privacy

* **Sensitive domain**: `emobot` is intended for mental-health support, not diagnosis. The assistant prompt instructs to provide the Indian suicide prevention hotline (`9152987821`) only when a user expresses unbearable pain or self-harm intent.
* **Data retention**: Conversations are written into the configured mem0 vector store. Treat the store as containing user-sensitive data — secure backups, restrict access, and delete test data when done.

---

## Troubleshooting & tips

* **`mem0ai` import fails**: ensure `mem0ai` is installed and compatible with your Python version.
* **Ollama not reachable**: configure `OLLAMA_API_URL` to an accessible provider or run Ollama locally. If generation fails, the API returns a fallback apology message.
* **Chroma**: if KB queries return nothing, confirm your collections exist at `VECTOR_STORE_PATH` and contain documents.
* **ddgs**: web search is optional; if `ddgs` is not installed the app will still run but without web-search freshness.

---

## Development

* `DEBUG=true` will enable `uvicorn --reload` style behavior (the code reads `DEBUG` and you typically launch uvicorn with `--reload` during development).
* The code supports both async and sync `mem0` clients: it prefers `AsyncMemory` if available, otherwise it falls back to a synchronous `Memory` via a thread executor wrapper.

---

## Contributing

PRs welcome. Please:

1. Open an issue describing the feature/bug.
2. Send a small, focused PR with tests or manual test steps.

---

