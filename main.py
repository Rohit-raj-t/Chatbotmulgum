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
import re
import json
from collections import Counter
from urllib.parse import urlparse

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

class OceanTestRequest(BaseModel):
    user_id: str
    ocean_scores: Dict[str, float]  # {"openness": 7.5, "conscientiousness": 8, ...}
    stage: Optional[str] = None     # "school" or "college"
    interests: Optional[List[str]] = None

class OceanTestResponse(BaseModel):
    user_id: str
    top_jobs: List[Dict[str, Any]]

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
# Improved OCEAN Test endpoint
# ---------------------------
# ---------------------------
# Improved OCEAN Test endpoint (robust, LLM-enrichment for missing fields)
# ---------------------------



@app.post("/oceantest", response_model=OceanTestResponse)
async def ocean_test_recommendations(data: OceanTestRequest):
    """
    Take OCEAN test scores, stage, and interests, and return top 10 recommendations.
    - 10th and below -> streams with core_subjects
    - 12th and below -> courses with curriculum, colleges, eligibility
    - college -> careers/jobs with roles, companies, eligibility
    """
    logger.info("Received oceantest request for user=%s stage=%s", data.user_id, data.stage)

    # -------------------- Normalize OCEAN scores --------------------
    raw_ocean = {}
    for k, v in (data.ocean_scores or {}).items():
        try:
            raw_ocean[str(k).lower()] = float(v)
        except Exception:
            raw_ocean[str(k).lower()] = 0.0
    for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
        raw_ocean.setdefault(trait, 0.0)
    mapping = {"o": "openness", "c": "conscientiousness", "e": "extraversion", "a": "agreeableness", "n": "neuroticism"}
    for short, full in mapping.items():
        if short in raw_ocean and raw_ocean.get(full, 0.0) == 0.0:
            raw_ocean[full] = raw_ocean[short]
    for k in list(raw_ocean.keys()):
        try:
            raw_ocean[k] = max(0.0, min(10.0, float(raw_ocean[k])))
        except Exception:
            raw_ocean[k] = 0.0

    interests = [str(i).strip() for i in (data.interests or [])]
    interests_blob = " ".join(i.lower() for i in interests)

    stage_raw = (data.stage or "").strip().lower()

    # -------------------- Normalize stage --------------------
    def normalize_stage(s: str) -> str:
        if not s: return "12th"
        if any(tok in s for tok in ("10", "10th", "ssc", "secondary", "class 10", "class10")):
            return "10th"
        if any(tok in s for tok in ("11", "11th", "12", "12th", "hsc", "higher secondary", "class 11", "class 12", "12th and below")):
            return "12th"
        if any(tok in s for tok in ("college", "undergrad", "btech", "b.e", "ba", "bsc", "b.com")):
            return "college"
        if s.isdigit():
            return "10th" if int(s) <= 10 else "12th"
        return "12th"

    stage_key = normalize_stage(stage_raw)

    # -------------------- Predefined mappings --------------------
    stream_subjects = {
        "pcm-cs": ["Physics", "Chemistry", "Mathematics", "Computer Science", "English"],
        "pcm": ["Physics", "Chemistry", "Mathematics", "English", "Optional (CS/Economics)"],
        "bio-math": ["Physics", "Chemistry", "Biology", "Mathematics", "English"],
        "pcb": ["Physics", "Chemistry", "Biology", "English", "Optional (Mathematics)"],
        "commerce-math": ["Accountancy", "Business Studies", "Economics", "Mathematics", "English"],
        "commerce": ["Accountancy", "Business Studies", "Economics", "English", "Mathematics (optional)"],
        "humanities": ["History", "Political Science", "Geography", "Economics", "English"],
        "arts-design": ["Art & Craft", "Design Fundamentals", "English", "History/Cultural Studies", "Optional (Mathematics)"],
        "vocational-it": ["Basic IT", "Computer Applications", "English", "Workplace Skills", "Mathematics (applied)"],
        "hospitality": ["English", "Food & Nutrition Basics", "Tourism Studies", "Basic Maths", "Communication Skills"],
        "agriculture": ["Biology (Plant/Animal basics)", "Chemistry basics", "Mathematics (applied)", "Environmental Studies", "English"]
    }

    predefined_courses = {
        "b.tech cse": {
            "overview": "Bachelor of Technology in Computer Science Engineering focuses on computation, algorithms, systems and software engineering.",
            "key_skills": ["Programming", "Data Structures & Algorithms", "Systems", "Problem Solving"],
            "curriculum": {
                "1st Year": ["Mathematics", "Physics/Chemistry basics", "Programming fundamentals"],
                "2nd Year": ["Data Structures", "Discrete Maths", "DBMS"],
                "3rd Year": ["Algorithms", "Operating Systems", "Machine Learning basics"],
                "4th Year": ["Advanced electives", "Capstone project", "Internship"]
            },
            "top_colleges": ["IITs", "NITs", "IIITs", "BITS Pilani", "VIT"],
            "career_opportunities": ["Software Engineer", "Systems Engineer", "Data Scientist", "DevOps Engineer"],
            "eligibility": {
                "educational_requirements": "12th with Physics & Mathematics (Chemistry optional), competitive entrance scores",
                "entrance_exams": ["JEE Main", "JEE Advanced", "BITSAT"]
            }
        }
    }

    predefined_careers = {
        "software engineer": {
            "overview": "Designs and builds software products and systems.",
            "key_skills": ["Programming", "Data Structures", "System Design"],
            "roles": ["Frontend", "Backend", "Full-stack", "SRE"],
            "top_companies": ["TCS", "Infosys", "Wipro", "Google", "Microsoft"],
            "eligibility": "B.Tech/BSc in CS or equivalent experience"
        }
    }

    def extract_json_array(text: str):
        if not text: return None
        try:
            start, end = text.index("["), text.rindex("]")
            return json.loads(text[start:end + 1])
        except Exception:
            pass
        try:
            return json.loads(text)
        except Exception:
            return None

    async def llm_json_object_for_course(course_name: str):
        p = f"""For the course \"{course_name}\", return EXACTLY a JSON object with keys: \"overview\", \"key_skills\" (list), \"curriculum\" (dict), \"top_colleges\" (list), \"career_opportunities\" (list), \"eligibility\" (dict)."""
        s = await generate_reply_ollama(p, model=LLM_MODEL_CAREER or LLM_MODEL)
        m = re.search(r'(\{.*\})', s, re.DOTALL)
        return json.loads(m.group(1)) if m else {}

    async def llm_json_object_for_job(job_name: str):
        p = f"""For the job '{job_name}', return EXACTLY a JSON object with keys: \"overview\", \"key_skills\" (list), \"roles\" (list), \"top_companies\" (list), \"eligibility\"."""
        s = await generate_reply_ollama(p, model=LLM_MODEL_CAREER or LLM_MODEL)
        m = re.search(r'(\{.*\})', s, re.DOTALL)
        return json.loads(m.group(1)) if m else {}

    async def llm_core_subjects_for_stream(stream_name: str):
        p = f"""List the typical Indian 10th-grade stream subjects for '{stream_name}' as a JSON array of strings only."""
        s = await generate_reply_ollama(p, model=LLM_MODEL_CAREER or LLM_MODEL)
        return extract_json_array(s)

    if stage_key == "10th":
        prompt = "Return EXACTLY 10 JSON objects with keys: stream, reason, overview, key_skills, future_scope, core_subjects."
    elif stage_key == "12th":
        prompt = "Return EXACTLY 10 JSON objects with keys: course, overview, key_skills, curriculum, top_colleges, career_opportunities, eligibility."
    else:
        prompt = "Return EXACTLY 10 JSON objects with keys: job, overview, key_skills, roles, top_companies, eligibility."

    llm_prompt = f"""
    You are an Indian career counsellor. Stage={stage_key}.
    OCEAN={raw_ocean}. Interests={interests_blob or 'not provided'}.
    {prompt}
    """

    try:
        llm_reply = await generate_reply_ollama(llm_prompt, model=LLM_MODEL_CAREER or LLM_MODEL)
        parsed = extract_json_array(llm_reply)
    except Exception:
        parsed = None

    def deterministic_fallback(stage: str):
        if stage == "10th":
            return [{"stream": s, "reason": "Fallback", "core_subjects": stream_subjects.get(s.lower(), [])} for s in list(stream_subjects.keys())[:10]]
        if stage == "12th":
            return [{"course": c} for c in ["B.Tech CSE", "B.Com", "MBBS", "B.Sc Physics", "BBA", "B.Des", "B.Arch", "BA", "Polytechnic", "B.Sc Life Sciences"]]
        return [{"job": j} for j in ["Software Engineer", "Data Scientist", "Product Manager", "Doctor", "UX Designer", "Chartered Accountant", "Entrepreneur", "Civil Engineer", "Lawyer", "Investment Banker"]]

    if not isinstance(parsed, list):
        parsed = deterministic_fallback(stage_key)

    final = []

    async def process_10th(item):
        name = item.get("stream", "General Science")
        cs = item.get("core_subjects") or stream_subjects.get(name.lower())
        if not cs:
            cs = await llm_core_subjects_for_stream(name) or ["English", "Maths", "Science", "Social Science"]
        return {
            "stream": name,
            "reason": item.get("reason", "Suggested based on interests and traits."),
            "core_subjects": cs,
            "overview": item.get("overview", f"Overview for {name}."),
            "key_skills": item.get("key_skills", ["Analytical Thinking", "Problem Solving"]),
            "future_scope": item.get("future_scope", "Multiple pathways.")
        }

    async def process_12th(item):
        name = item.get("course", "Unknown Course")
        info = predefined_courses.get(name.lower(), {})
        enriched = await llm_json_object_for_course(name) if not info else {}
        # embed best colleges from web search
        best_colleges = []
        try:
            hits = await web_search_ddgs(f"top colleges in India for {name}", max_results=5)
            best_colleges = [f"{h.get('title')} ({h.get('href') or h.get('url')})" for h in hits]
        except Exception:
            pass
        return {
            "course": name,
            "overview": item.get("overview") or info.get("overview") or enriched.get("overview", f"Overview for {name}."),
            "key_skills": item.get("key_skills") or info.get("key_skills") or enriched.get("key_skills", []),
            "curriculum": item.get("curriculum") or info.get("curriculum") or enriched.get("curriculum", {}),
            "top_colleges": best_colleges or item.get("top_colleges") or info.get("top_colleges") or enriched.get("top_colleges", []),
            "career_opportunities": item.get("career_opportunities") or info.get("career_opportunities") or enriched.get("career_opportunities", []),
            "eligibility": item.get("eligibility") or info.get("eligibility") or enriched.get("eligibility", {"educational_requirements": "12th Grade"})
        }

    async def process_college(item):
        name = item.get("job", "Unknown Job")
        info = predefined_careers.get(name.lower(), {})
        enriched = await llm_json_object_for_job(name) if not info else {}
        return {
            "job": name,
            "overview": item.get("overview") or info.get("overview") or enriched.get("overview", f"Overview for {name}."),
            "key_skills": item.get("key_skills") or info.get("key_skills") or enriched.get("key_skills", []),
            "roles": item.get("roles") or info.get("roles") or enriched.get("roles", []),
            "top_companies": item.get("top_companies") or info.get("top_companies") or enriched.get("top_companies", []),
            "eligibility": item.get("eligibility") or info.get("eligibility") or enriched.get("eligibility", "Typical requirements")
        }

    for item in parsed[:10]:
        try:
            if stage_key == "10th":
                final.append(await process_10th(item))
            elif stage_key == "12th":
                final.append(await process_12th(item))
            else:
                final.append(await process_college(item))
        except Exception:
            continue

    return OceanTestResponse(user_id=data.user_id, top_jobs=final[:10])


# ---------------------------
# Main entry
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, log_level="debug" if DEBUG else "info")