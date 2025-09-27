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
# Improved OCEAN Test endpoint with caching
# ---------------------------

@app.post("/oceantest", response_model=OceanTestResponse)
async def ocean_test_recommendations(data: OceanTestRequest):
    """
    Take OCEAN test scores, stage, and interests, and return top 10 recommendations.
    - 10th and below -> canonical Indian streams (PCM-CS, PCM, PCMB, PCB, PCB-CS, COMMERCE-MATH, COMMERCE, ACCOUNTANCY, HUMANITIES, ARTS-DESIGN, ...)
      Deterministic / scored; NO LLM calls for 10th to keep consistent and fast.
    - 12th and below -> courses with curriculum, colleges (context only), eligibility (single LLM call for the list; enrichment cached)
    - college -> careers/jobs with roles, companies, eligibility (single LLM call for the list; enrichment cached)
    Includes per-request caching for enrichment so repeated names in same request don't re-call LLM.
    """
    import re, json, httpx
    from typing import Dict, Any, List

    logger.info("Received oceantest request for user=%s stage=%s", data.user_id, data.stage)

    # -------------------- Normalize OCEAN scores --------------------
    raw_ocean: Dict[str, float] = {}
    for k, v in (data.ocean_scores or {}).items():
        try:
            raw_ocean[str(k).lower()] = float(v)
        except Exception:
            raw_ocean[str(k).lower()] = 0.0

    for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
        raw_ocean.setdefault(trait, 0.0)
    # map shorthand O C E A N
    mapping = {"o": "openness", "c": "conscientiousness", "e": "extraversion", "a": "agreeableness", "n": "neuroticism"}
    for short, full in mapping.items():
        if short in raw_ocean and raw_ocean.get(full, 0.0) == 0.0:
            raw_ocean[full] = raw_ocean[short]
    # clamp to [0,10]
    for k in list(raw_ocean.keys()):
        try:
            raw_ocean[k] = max(0.0, min(10.0, float(raw_ocean[k])))
        except Exception:
            raw_ocean[k] = 0.0

    # Normalize interests to UPPERCASE so they match UPPERCASE keyword sets below
    interests = [str(i).strip().upper() for i in (data.interests or [])]
    interests_blob = " ".join(interests)

    stage_raw = (data.stage or "").strip().lower()

    # -------------------- Normalize stage --------------------
    def normalize_stage(s: str) -> str:
        if not s:
            return "12th"
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

    # -------------------- Canonical 10th streams & their core subjects --------------------
    # Keys are canonical stream IDs used in output (UPPERCASE)
    stream_subjects = {
        "PCM-CS": ["Physics", "Chemistry", "Mathematics", "Computer Science", "English"],   # PCM with CS
        "PCM": ["Physics", "Chemistry", "Mathematics", "English", "Optional (CS/Economics)"],
        "PCMB": ["Physics", "Chemistry", "Mathematics", "Biology", "English"],             # PCMB (both Maths & Bio)
        "PCB": ["Physics", "Chemistry", "Biology", "English", "Optional (Mathematics)"],
        "PCB-CS": ["Physics", "Chemistry", "Biology", "Computer Science", "English"],      # PCB + CS hybrid
        "COMMERCE-MATH": ["Accountancy", "Business Studies", "Economics", "Mathematics", "English"],
        "COMMERCE": ["Accountancy", "Business Studies", "Economics", "English", "Mathematics (optional)"],
        "ACCOUNTANCY": ["Accountancy", "Business Studies", "Economics", "English", "Mathematics (optional)"],
        "HUMANITIES": ["History", "Political Science", "Geography", "Economics", "English"],
        "ARTS-DESIGN": ["Art & Craft", "Design Fundamentals", "English", "History/Cultural Studies", "Optional (Mathematics)"],
        "VOCATIONAL-IT": ["Basic IT", "Computer Applications", "English", "Workplace Skills", "Mathematics (applied)"],
        "HOSPITALITY": ["English", "Food & Nutrition Basics", "Tourism Studies", "Basic Maths", "Communication Skills"],
        "AGRICULTURE": ["Biology (Plant/Animal basics)", "Chemistry basics", "Mathematics (applied)", "Environmental Studies", "English"]
    }

    # Keywords mapping to canonical streams (keys are UPPERCASE)
    stream_interest_keywords = {
        "PCM": {"MATH", "PHYSICS", "CHEMISTRY", "ENGINEERING", "PROBLEM", "ALGORITHMS"},
        "PCM-CS": {"PROGRAMMING", "COMPUTER", "CODING", "ALGORITHMS", "SOFTWARE", "MATH"},
        "PCMB": {"MATH", "BIOLOGY", "BOTH", "PCMB", "BOTH MATH BIOLOGY", "BIO+MATH"},
        "PCB": {"BIOLOGY", "MEDICINE", "BIO", "NEUROSCIENCE", "CHEMISTRY"},
        "PCB-CS": {"BIOLOGY", "COMPUTER", "BIOIT", "BIOINFORMATICS", "PROGRAMMING"},
        "COMMERCE": {"BUSINESS", "ACCOUNTANCY", "ECONOMICS", "MONEY", "FINANCE", "COMMERCE"},
        "COMMERCE-MATH": {"FINANCE", "MATH", "ACCOUNTS", "ECONOMICS", "COMMERCE"},
        "ACCOUNTANCY": {"ACCOUNT", "ACCOUNTS", "ACCOUNTANCY", "AUDITING"},
        "HUMANITIES": {"HISTORY", "POLITICAL", "GEOGRAPHY", "SOCIETY", "LAW", "HUMANITIES"},
        "ARTS-DESIGN": {"ART", "DESIGN", "CREATIVE", "DRAWING", "ANIMATION", "FASHION"},
        "VOCATIONAL-IT": {"IT", "COMPUTER", "OFFICE", "APPLICATIONS", "SUPPORT"},
        "HOSPITALITY": {"HOTEL", "TOURISM", "TRAVEL", "SERVICE", "COOKING", "HOSPITALITY"},
        "AGRICULTURE": {"AGRI", "FARMING", "AGRICULTURE", "PLANTS", "ANIMALS"}
    }

    # -------------------- Predefined fallback course/career templates --------------------
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
            "career_opportunities": ["Software Engineer", "Systems Engineer", "Data Scientist", "DevOps Engineer"],
            "eligibility": {"educational_requirements": "12th with Physics & Mathematics (Chemistry optional)", "entrance_exams": ["JEE Main", "JEE Advanced", "BITSAT"]}
        },
        "mbbs": {
            "overview": "Professional medical degree training to become a physician/doctor.",
            "key_skills": ["Clinical knowledge", "Empathy", "Patient management", "Medical ethics"],
            "curriculum": {"Pre-clinical": ["Anatomy", "Physiology"], "Para-clinical": ["Pharmacology", "Pathology"], "Clinical": ["Medicine", "Surgery"]},
            "career_opportunities": ["Clinician", "Surgeon", "Public Health Specialist"],
            "eligibility": {"educational_requirements": "12th with PCB", "entrance_exams": ["NEET-UG"]}
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

    # -------------------- Helpers --------------------
    def extract_json_array(text: str):
        if not text:
            return None
        try:
            start, end = text.index("["), text.rindex("]")
            return json.loads(text[start:end + 1])
        except Exception:
            pass
        try:
            return json.loads(text)
        except Exception:
            return None

    # per-request caches (avoid duplicate LLM calls within same request)
    course_enrich_cache: Dict[str, Dict[str, Any]] = {}
    job_enrich_cache: Dict[str, Dict[str, Any]] = {}

    async def llm_json_object_for_course_cached(course_name: str):
        key = course_name.lower().strip()
        if key in course_enrich_cache:
            return course_enrich_cache[key]
        p = f"""For the course "{course_name}", return EXACTLY a JSON object with keys:
"overview", "key_skills" (list), "curriculum" (dict), "career_opportunities" (list), "eligibility" (dict)."""
        s = await generate_reply_ollama(p, model=LLM_MODEL_CAREER or LLM_MODEL)
        m = re.search(r'(\{.*\})', s, re.DOTALL)
        try:
            obj = json.loads(m.group(1)) if m else {}
        except Exception:
            obj = {}
        course_enrich_cache[key] = obj
        return obj

    async def llm_json_object_for_job_cached(job_name: str):
        key = job_name.lower().strip()
        if key in job_enrich_cache:
            return job_enrich_cache[key]
        p = f"""For the job '{job_name}', return EXACTLY a JSON object with keys:
"overview", "key_skills" (list), "roles" (list), "top_companies" (list), "eligibility" (dict)."""
        s = await generate_reply_ollama(p, model=LLM_MODEL_CAREER or LLM_MODEL)
        m = re.search(r'(\{.*\})', s, re.DOTALL)
        try:
            obj = json.loads(m.group(1)) if m else {}
        except Exception:
            obj = {}
        job_enrich_cache[key] = obj
        return obj

    async def llm_core_subjects_for_stream(stream_name: str):
        p = f"""List the typical Indian 10th-grade stream subjects for '{stream_name}' as a JSON array of strings only."""
        s = await generate_reply_ollama(p, model=LLM_MODEL_CAREER or LLM_MODEL)
        return extract_json_array(s)

    # -------------------- NIRF / government-colleges as prompt context (single fetch) --------------------
    top_colleges_reference: List[str] = []
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get("https://www.nirfindia.org/Rankings/2025/Ranking.html")
            html = r.text
        matches = re.findall(r'<td[^>]*class[^>]*institution[^>]*>(.*?)</td>', html, re.DOTALL | re.IGNORECASE)
        if not matches:
            matches = re.findall(r'<td[^>]*>([^<]*IIT[^<]*|[^<]*NIT[^<]*|[^<]*University[^<]*)</td>', html, re.DOTALL | re.IGNORECASE)
        cleaned = []
        for m in matches:
            text = re.sub(r'<.*?>', '', m).strip()
            if text and len(text) > 3:
                cleaned.append(text)
        govt_candidates = [c for c in cleaned if any(x.lower() in c.lower() for x in ("iit", "nit", "iiit", "central university", "aiims", "iisc", "jipmer"))]
        if not govt_candidates:
            govt_candidates = cleaned[:15]
        top_colleges_reference = govt_candidates[:15]
    except Exception:
        top_colleges_reference = ["IIT Delhi", "IIT Bombay", "IIT Madras", "IIT Kanpur", "IIT Kharagpur", "IIT Roorkee", "IIT Guwahati", "IIT BHU", "NIT Trichy", "NIT Surathkal"]

    top_colleges_prompt_context = "Top government institutes (NIRF-preferred): " + ", ".join(top_colleges_reference[:10])

    # -------------------- Stage-specific behavior --------------------
    final: List[Dict[str, Any]] = []

    # ---------- 10th: deterministic canonical streams (no LLM) ----------
    if stage_key == "10th":
        # If interests are empty or 'GENERAL', return canonical recommended order (commonly used in India)
        if (not interests) or ("GENERAL" in interests):
            canonical_order = [
                "PCM-CS",      # PCM + Computer Science
                "PCM",         # PCM
                "PCMB",        # PCMB (both Maths & Bio)
                "PCB",         # PCB
                "PCB-CS",      # PCB + CS (hybrid)
                "COMMERCE-MATH",
                "COMMERCE",
                "ACCOUNTANCY",
                "HUMANITIES",
                "ARTS-DESIGN"
            ]
            for s in canonical_order[:10]:
                core = stream_subjects.get(s, ["English", "Mathematics", "Science"])
                result = {
                    "stream": s,
                    "reason": f"Recommended as a common Indian 10th stream option: {s}.",
                    "core_subjects": core,
                    "overview": f"Overview for {s}.",
                    "key_skills": ["Analytical Thinking", "Problem Solving"] if (s.startswith("PCM") or s in ("PCMB", "PCB")) else ["Communication", "Creativity"],
                    "future_scope": "Multiple pathways (higher education, vocational training, professional courses)."
                }
                final.append(result)
            return OceanTestResponse(user_id=data.user_id, top_jobs=final[:10])

        # Otherwise compute simple scoring based on interests + trait heuristics
        def score_stream(stream_key: str) -> float:
            skey = stream_key.upper()
            score = 0.0
            kws = stream_interest_keywords.get(skey, set())
            if interests:
                # 2 points per keyword match
                score += 2.0 * len(kws.intersection(set(interests)))
            # Add small OCEAN heuristics
            o = raw_ocean.get("openness", 0.0)
            c = raw_ocean.get("conscientiousness", 0.0)
            e = raw_ocean.get("extraversion", 0.0)
            a = raw_ocean.get("agreeableness", 0.0)
            n = raw_ocean.get("neuroticism", 0.0)
            # Heuristic boosts (use UPPERCASE keys)
            if skey == "ARTS-DESIGN" and o >= 6.0:
                score += 1.5
            if skey in ("PCM", "PCM-CS", "PCMB", "PCB") and c >= 5.0:
                score += 1.0
            if skey in ("HOSPITALITY", "VOCATIONAL-IT") and e >= 6.0:
                score += 1.0
            if skey in ("HUMANITIES", "COMMERCE", "ACCOUNTANCY") and a >= 5.0:
                score += 0.8
            # small negative bias if neuroticism is high for hardcore technical streams
            if n >= 7.0 and skey in ("PCM-CS", "PCM", "PCB"):
                score -= 0.5
            return score

        scored = []
        for key in stream_subjects.keys():
            scored.append((score_stream(key), key))
        scored.sort(reverse=True, key=lambda x: x[0])

        top_streams = [k for _, k in scored][:10]

        for s in top_streams:
            core = stream_subjects.get(s, ["English", "Mathematics", "Science"])
            result = {
                "stream": s,
                "reason": f"Suggested because your interest/trait profile aligns with the stream '{s}'.",
                "core_subjects": core,
                "overview": f"Overview for {s}.",
                "key_skills": ["Analytical Thinking", "Problem Solving"] if (s.startswith("PCM") or s in ("PCMB", "PCB")) else ["Communication", "Critical Thinking"],
                "future_scope": "Multiple pathways (higher education, vocational training, professional courses)."
            }
            final.append(result)

        return OceanTestResponse(user_id=data.user_id, top_jobs=final[:10])

    # ---------- 12th or college: single LLM call for 10 items (with top_colleges_prompt_context only used as prompt context) ----------
    if stage_key == "12th":
        llm_required_keys = ["course", "overview", "key_skills", "curriculum", "career_opportunities", "eligibility"]
        stage_instruction = "Return EXACTLY 10 JSON objects with keys: " + ", ".join(llm_required_keys) + "."
    else:
        llm_required_keys = ["job", "overview", "key_skills", "roles", "top_companies", "eligibility"]
        stage_instruction = "Return EXACTLY 10 JSON objects with keys: " + ", ".join(llm_required_keys) + "."

    llm_prompt = f"""
You are an Indian career counsellor. Stage={stage_key}.
OCEAN={raw_ocean}.
Interests={interests_blob or 'not provided'}.
Context (do not expose raw links in the API response): {top_colleges_prompt_context}

Instruction: {stage_instruction}
When listing colleges (if relevant), prioritize government institutes (IITs, NITs, IIITs, central universities) from the provided context.
Return pure JSON (an array of objects) and nothing else.
"""

    try:
        llm_reply = await generate_reply_ollama(llm_prompt, model=LLM_MODEL_CAREER or LLM_MODEL)
        parsed = extract_json_array(llm_reply)
    except Exception:
        parsed = None

    # fallback deterministic
    if not isinstance(parsed, list):
        if stage_key == "12th":
            parsed = [{"course": c} for c in ["B.Tech CSE", "B.Com", "MBBS", "B.Sc Physics", "BBA", "B.Des", "B.Arch", "BA", "Polytechnic", "B.Sc Life Sciences"]]
        else:
            parsed = [{"job": j} for j in ["Software Engineer", "Data Scientist", "Doctor", "Lawyer", "UX Designer", "Chartered Accountant", "Civil Engineer", "Teacher", "Product Manager", "Entrepreneur"]]

    # Process parsed items and enrich with cached LLM calls when necessary (no top_colleges list in output)
    for item in parsed[:10]:
        try:
            if stage_key == "12th":
                name = item.get("course") or item.get("title") or "Unknown Course"
                info = predefined_courses.get(name.lower(), {})
                # If the LLM already returned the fields, trust them (except top_colleges which we drop)
                if any(k in item for k in ("overview", "key_skills", "curriculum", "career_opportunities", "eligibility")):
                    overview = item.get("overview") or info.get("overview", f"Overview for {name}.")
                    key_skills = item.get("key_skills") or info.get("key_skills", [])
                    curriculum = item.get("curriculum") or info.get("curriculum", {})
                    career_ops = item.get("career_opportunities") or info.get("career_opportunities", [])
                    eligibility = item.get("eligibility") or info.get("eligibility", {"educational_requirements": "12th Grade"})
                else:
                    enriched = await llm_json_object_for_course_cached(name)
                    overview = enriched.get("overview") or info.get("overview", f"Overview for {name}.")
                    key_skills = enriched.get("key_skills") or info.get("key_skills", [])
                    curriculum = enriched.get("curriculum") or info.get("curriculum", {})
                    career_ops = enriched.get("career_opportunities") or info.get("career_opportunities", [])
                    eligibility = enriched.get("eligibility") or info.get("eligibility", {"educational_requirements": "12th Grade"})

                response_item = {
                    "course": name,
                    "overview": overview,
                    "key_skills": key_skills,
                    "curriculum": curriculum,
                    # intentionally DO NOT include top_colleges list/links in API output per request
                    "career_opportunities": career_ops,
                    "eligibility": eligibility
                }
                final.append(response_item)

            else:  # college stage: jobs/careers
                name = item.get("job") or item.get("title") or "Unknown Job"
                info = predefined_careers.get(name.lower(), {})
                if any(k in item for k in ("overview", "key_skills", "roles", "top_companies", "eligibility")):
                    overview = item.get("overview") or info.get("overview", f"Overview for {name}.")
                    key_skills = item.get("key_skills") or info.get("key_skills", [])
                    roles = item.get("roles") or info.get("roles", [])
                    top_companies = item.get("top_companies") or info.get("top_companies", [])
                    eligibility = item.get("eligibility") or info.get("eligibility", "Typical requirements")
                else:
                    enriched = await llm_json_object_for_job_cached(name)
                    overview = enriched.get("overview") or info.get("overview", f"Overview for {name}.")
                    key_skills = enriched.get("key_skills") or info.get("key_skills", [])
                    roles = enriched.get("roles") or info.get("roles", [])
                    top_companies = enriched.get("top_companies") or info.get("top_companies", [])
                    eligibility = enriched.get("eligibility") or info.get("eligibility", "Typical requirements")

                response_item = {
                    "job": name,
                    "overview": overview,
                    "key_skills": key_skills,
                    "roles": roles,
                    "top_companies": top_companies,
                    "eligibility": eligibility
                }
                final.append(response_item)

        except Exception:
            logger.exception("Failed to process item: %s", item)
            continue

    return OceanTestResponse(user_id=data.user_id, top_jobs=final[:10])

# ---------------------------
# Main entry
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, log_level="debug" if DEBUG else "info")