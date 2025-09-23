# main.py
"""
Unified emobot + careerbot API
- Single /chat endpoint. Choose persona with "bot": "emobot" | "careerbot".
- Per-request LLM override via metadata.llm_model.
- Shared mem0 memory (tagged with persona in metadata).
- Chroma RAG only used when a Chroma client is available AND not already managed by mem0.
- DuckDuckGo (DDGS) web search for "require_latest" or freshness queries (careerbot).
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ---------------------------
# Optional imports and flags
# ---------------------------
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except Exception:
    DDGS_AVAILABLE = False

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except Exception:
    CHROMADB_AVAILABLE = False

try:
    from mem0 import AsyncMemory
    from mem0.configs.base import MemoryConfig
    MEM0_ASYNC = True
except Exception:
    try:
        from mem0 import Memory as SyncMemory
        MEM0_ASYNC = False
    except Exception as e:
        raise ImportError("mem0ai is required (pip install mem0ai). Import error: " + str(e))

# ---------------------------
# Load environment
# ---------------------------
load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b")
LLM_MODEL_EMO = os.getenv("LLM_MODEL_EMO", "")
LLM_MODEL_CAREER = os.getenv("LLM_MODEL_CAREER", "")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.1))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 2000))

EMBEDDER_PROVIDER = os.getenv("EMBEDDER_PROVIDER", "ollama")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "nomic-embed-text:latest")

VECTOR_STORE_PROVIDER = os.getenv("VECTOR_STORE_PROVIDER", "chroma")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "db_chroma")
VECTOR_STORE_COLLECTION = os.getenv("VECTOR_STORE_COLLECTION", "mental_health_memories")
VECTOR_STORE_SCHOOL_COLLECTION = os.getenv("VECTOR_STORE_SCHOOL_COLLECTION", "career_school_guidance")
VECTOR_STORE_COLLEGE_COLLECTION = os.getenv("VECTOR_STORE_COLLEGE_COLLECTION", "career_college_guidance")

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

WEB_SEARCH_MAX_RESULTS = int(os.getenv("DUCKDUCKGO_MAX_RESULTS", 5))
WEB_SEARCH_REGION = os.getenv("WEB_SEARCH_REGION", "in-en")

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")

logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger("unified-bot")

# ---------------------------
# Initialize mem0
# ---------------------------
if MEM0_ASYNC:
    memory_config = MemoryConfig(
        llm={
            "provider": LLM_PROVIDER,
            "config": {"model": LLM_MODEL, "temperature": LLM_TEMPERATURE, "max_tokens": LLM_MAX_TOKENS},
        },
        vector_store={
            "provider": VECTOR_STORE_PROVIDER,
            "config": {"path": VECTOR_STORE_PATH, "collection_name": VECTOR_STORE_COLLECTION},
        },
        embedder={"provider": EMBEDDER_PROVIDER, "config": {"model": EMBEDDER_MODEL}},
    )
    memory = AsyncMemory(config=memory_config)
else:
    sync_config = {
        "llm": {"provider": LLM_PROVIDER, "config": {"model": LLM_MODEL, "temperature": LLM_TEMPERATURE, "max_tokens": LLM_MAX_TOKENS}},
        "vector_store": {"provider": VECTOR_STORE_PROVIDER, "config": {"path": VECTOR_STORE_PATH, "collection_name": VECTOR_STORE_COLLECTION}},
        "embedder": {"provider": EMBEDDER_PROVIDER, "config": {"model": EMBEDDER_MODEL}},
    }
    memory = SyncMemory(config=sync_config)

logger.info("Mem0 memory initialized (async=%s) using vector provider '%s'", MEM0_ASYNC, VECTOR_STORE_PROVIDER)

# ---------------------------
# Chroma client
# ---------------------------
chroma_client: Optional[Any] = None
if CHROMADB_AVAILABLE and VECTOR_STORE_PROVIDER != "chroma":
    try:
        chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
        logger.info("Manual Chroma client created at %s", VECTOR_STORE_PATH)
    except Exception:
        logger.exception("Failed to create manual chromadb client; continuing without it")
else:
    if VECTOR_STORE_PROVIDER == "chroma":
        logger.info("Vector store provider is 'chroma' — mem0 manages Chroma internally")

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="Unified Bot API (emobot + careerbot)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ---------------------------
# Models
# ---------------------------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_id: str
    messages: List[Message]
    metadata: Optional[Dict[str, Any]] = None
    bot: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str

# ---------------------------
# Prompts
# ---------------------------
EMO_SYSTEM_PROMPT = (
    "You are a compassionate mental health companion for Indian teens and college students. "
    "Engage with them like a supportive therapist, always trying to understand their feelings, thoughts, and experiences. "
    "Respond with empathy, gently guiding them to navigate emotions, challenges, and stress in practical, culturally relevant ways. "
    "Only if the student expresses unbearable pain or thoughts of self-harm or suicide should you explicitly encourage them to seek professional help. "
    "In such cases, provide the suicide prevention hotline in India: 9152987821. "
    "Otherwise, focus on listening, validating their feelings, exploring coping strategies, and offering guidance appropriate to Indian societal norms and contexts."
    "Try to keep it under a single paragraph at a time"
)

CAREER_SYSTEM_PROMPT = (
    "You are a practical, culturally-aware career guidance assistant for Indian school and college students. "
    "When answering, be clear and actionable: list exams, college types, typical career outcomes, skill roadmaps, and what companies expect (for college students). "
    "Use the student's metadata (stage, OCEAN traits, interests) to personalize suggestions. When you mention facts like rankings or exam dates, indicate if they were fetched live and include source URLs. "
    "Ask clarifying follow-ups when needed. Keep language simple and encouraging."
    "Try to keep it under a single paragraph at a time and keep the answers short"
)
# ---------------------------
# Utils
# ---------------------------
async def _run_in_thread(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

async def memory_add(messages, user_id, metadata=None, **kw):
    try:
        if MEM0_ASYNC:
            return await memory.add(messages=messages, user_id=user_id, metadata=metadata or {}, **kw)
        else:
            return await _run_in_thread(memory.add, messages, user_id, metadata or {}, **kw)
    except Exception:
        logger.exception("memory.add failed")

async def memory_search(query: str, user_id: Optional[str] = None, limit: int = 3):
    try:
        if MEM0_ASYNC:
            return await memory.search(query=query, user_id=user_id, limit=limit)
        else:
            return await _run_in_thread(memory.search, query, user_id, limit)
    except Exception:
        logger.exception("memory.search failed")
        return {"results": []}

async def memory_get_all(user_id: str):
    try:
        if MEM0_ASYNC:
            return await memory.get_all(user_id=user_id)
        else:
            return await _run_in_thread(memory.get_all, user_id)
    except Exception:
        logger.exception("memory.get_all failed")
        return []

async def memory_delete_all(user_id: str):
    try:
        if MEM0_ASYNC:
            return await memory.delete_all(user_id=user_id)
        else:
            return await _run_in_thread(memory.delete_all, user_id)
    except Exception:
        logger.exception("memory.delete_all failed")

async def generate_reply_ollama(prompt: str, model: Optional[str] = None) -> str:
    payload = {"model": model or LLM_MODEL, "prompt": prompt, "stream": False}
    timeout = httpx.Timeout(120.0, connect=60.0)
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(OLLAMA_API_URL, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json() if r.text and r.text.strip() else {}
            for key in ("response", "text", "generated", "result", "output"):
                if isinstance(data, dict) and data.get(key):
                    return data.get(key)
            if r.text and r.text.strip():
                return r.text.strip()
    except Exception:
        logger.exception("Ollama API call failed")
    return "Sorry — I couldn't generate a reply right now."

async def web_search_ddgs(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS, region: str = WEB_SEARCH_REGION):
    if not DDGS_AVAILABLE:
        return []
    def _ddgs_search():
        with DDGS() as ddgs:
            return list(ddgs.text(query, region=region, safesearch="moderate", max_results=max_results))
    try:
        return await _run_in_thread(_ddgs_search)
    except Exception:
        logger.exception("DDGS search failed")
        return []

def _chroma_query(collection_name: str, query: str, limit: int = 4):
    if not chroma_client:
        return []
    try:
        collection = chroma_client.get_collection(name=collection_name)
        res = collection.query(query_texts=[query], n_results=limit, include=["documents", "metadatas"])
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return [{"document": docs[i], "metadata": metas[i] if i < len(metas) else {}} for i in range(len(docs))]
    except Exception:
        logger.exception("chroma collection query failed")
        return []

async def chroma_search(collection_name: str, query: str, limit: int = 4):
    return await _run_in_thread(_chroma_query, collection_name, query, limit)

FRESHNESS_KEYWORDS = {"latest", "new", "2025", "2024", "rank", "ranking", "placement", "average package", "package", "cutoff", "exam date", "results", "deadline"}

def needs_freshness(query: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    if not query:
        return False
    if metadata and metadata.get("require_latest"):
        return True
    q = query.lower()
    return any(k in q for k in FRESHNESS_KEYWORDS)

# ---------------------------
# Chat handler
# ---------------------------
async def _handle_chat(request: ChatRequest, bot_type: Optional[str]) -> ChatResponse:
    bot = (bot_type or request.bot or (request.metadata or {}).get("bot") or "emobot").lower()
    if bot not in ("emobot", "careerbot", "career"):
        bot = "emobot"

    if not request.user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    last_user_msg = next((m.content for m in reversed(request.messages) if m.role == "user" and m.content.strip()), None)
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="At least one non-empty user message is required")

    metadata = dict(request.metadata or {})
    metadata["bot"] = bot

    system_prompt = EMO_SYSTEM_PROMPT if bot == "emobot" else CAREER_SYSTEM_PROMPT
    model_override = metadata.get("llm_model") or (
        LLM_MODEL_EMO if bot == "emobot" and LLM_MODEL_EMO else (
            LLM_MODEL_CAREER if bot in ("careerbot", "career") and LLM_MODEL_CAREER else LLM_MODEL
        )
    )

    try:
        await memory_add([{"role": m.role, "content": m.content} for m in request.messages], user_id=request.user_id, metadata=metadata)
    except Exception:
        logger.exception("Failed to add messages to memory")

    try:
        mem_search = await memory_search(last_user_msg, user_id=request.user_id, limit=3)
        raw_results = mem_search.get("results", []) if isinstance(mem_search, dict) else (mem_search or [])
        memories_text = "\n".join(f"- { (r.get('memory') or r.get('content') or r.get('text') or str(r)) }" for r in raw_results)
    except Exception:
        logger.exception("Memory retrieval failed")
        memories_text = ""

    kb_text = ""
    web_text = ""
    if bot in ("careerbot", "career"):
        stage = (metadata.get("stage") or "").lower()
        kb_collection = VECTOR_STORE_SCHOOL_COLLECTION if stage == "school" else (VECTOR_STORE_COLLEGE_COLLECTION if stage == "college" else None)
        if kb_collection and chroma_client:
            try:
                kb_results = await chroma_search(kb_collection, last_user_msg, limit=4)
                if kb_results:
                    kb_text = "\n".join(f"- {r.get('document','')} (source: {r.get('metadata',{}).get('source','')})" for r in kb_results)
            except Exception:
                logger.exception("Chroma search failed")
        if needs_freshness(last_user_msg, metadata) and DDGS_AVAILABLE:
            try:
                web_hits = await web_search_ddgs(last_user_msg, max_results=WEB_SEARCH_MAX_RESULTS, region=WEB_SEARCH_REGION)
                web_text = "\n".join(f"- {h.get('title','')} ({h.get('href') or h.get('url','')})\n  {h.get('body','')}" for h in web_hits[:WEB_SEARCH_MAX_RESULTS])
            except Exception:
                logger.exception("Web search failed")

    if bot == "emobot":
        prompt = f"{system_prompt}\n\nUser Memories:\n{memories_text}\n\nUser: {last_user_msg}"
    else:
        stage = metadata.get("stage") or "unknown"
        ocean = metadata.get("ocean") or {}
        interests = metadata.get("interests") or []
        profile = f"Stage: {stage}\nOCEAN: {ocean}\nInterests: {', '.join(interests) if interests else 'not provided'}"
        parts = [system_prompt, "\nUser Profile:\n" + profile]
        if memories_text:
            parts.append("\nUser Memory:\n" + memories_text)
        if kb_text:
            parts.append("\nKnowledge Base:\n" + kb_text)
        if web_text:
            parts.append("\nWeb Results:\n" + web_text)
        parts.append(f"\nUser asked:\n{last_user_msg}")
        prompt = "\n\n".join(parts)

    reply = await generate_reply_ollama(prompt, model=model_override)
    try:
        await memory_add([{"role": "assistant", "content": reply}], user_id=request.user_id, metadata=metadata)
    except Exception:
        logger.exception("Failed to store reply in memory")

    return ChatResponse(reply=reply)

# ---------------------------
# API endpoints
# ---------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    return await _handle_chat(request, bot_type=None)

@app.post("/chat/{bot_type}", response_model=ChatResponse)
async def chat_with_path(bot_type: str, request: ChatRequest):
    return await _handle_chat(request, bot_type=bot_type)

@app.get("/memories/user/{user_id}")
async def get_user_memories(user_id: str):
    return {"user_id": user_id, "memories": await memory_get_all(user_id)}

@app.get("/memories/search")
async def search_memories(user_id: str = Query(...), query: str = Query(...), limit: int = 5):
    results = await memory_search(query=query, user_id=user_id, limit=limit)
    return {"user_id": user_id, "query": query, "results": results.get("results", []) if isinstance(results, dict) else results}

@app.delete("/memories/user/{user_id}")
async def delete_user_memories(user_id: str):
    await memory_delete_all(user_id)
    return {"status": "success", "message": f"All memories for {user_id} deleted."}

@app.get("/")
def read_root():
    return {"status": "ok", "service": "unified-bot"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=APP_HOST, port=APP_PORT, reload=DEBUG)
