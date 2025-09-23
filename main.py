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
@app.post("/oceantest", response_model=OceanTestResponse)
async def ocean_test_recommendations(data: OceanTestRequest):
    """
    Improved OCEAN Test endpoint:
    - stage accepts: "10th" / "10th and below", "12th" / "12th and below" (covers 11th/12th), "college"
    - Returns a JSON array (>=10 objects) tailored to the stage:
      * 10th: best streams (PCM-CS, BIO-Maths, Commerce, Arts, Vocational...) with Overview, Key Skills, Career Paths, Future Scope
      * 12th: same + Curriculum (year-wise where applicable), Top Colleges in India, Career Opportunities, Eligibility
      * college: career-focused entries (job/role) with roles, skills, top companies and typical pathways
    """
    import json

    logger.info("Received oceantest request for user=%s stage=%s", data.user_id, data.stage)

    # --- Normalize + validate OCEAN scores ---
    raw_ocean = {k.lower(): float(v) for k, v in (data.ocean_scores or {}).items()}
    expected = {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}
    for trait in expected:
        raw_ocean.setdefault(trait, 0.0)
    for k in list(raw_ocean.keys()):
        try:
            val = float(raw_ocean[k])
            raw_ocean[k] = max(0.0, min(10.0, val))
        except Exception:
            raw_ocean[k] = 0.0

    interests = [i.lower() for i in (data.interests or [])]
    interests_text = ", ".join(data.interests or []) if data.interests else "not specified"

    # Stage classification
    stage_raw = (data.stage or "").strip().lower()
    if any(tok in stage_raw for tok in ("10th", "10", "10th and below", "secondary", "ssc")):
        stage_key = "10th"
    elif any(tok in stage_raw for tok in ("12th", "11th", "11", "12", "12th and below", "higher secondary", "hsc")):
        stage_key = "12th"
    elif "college" in stage_raw or "undergrad" in stage_raw or "btech" in stage_raw:
        stage_key = "college"
    else:
        # default to 12th-style guidance if unclear
        stage_key = "12th"

    # Helpers to pick suggestions based on interests + dominant traits
    top_trait = max(raw_ocean.items(), key=lambda x: x[1])[0]

    def interest_matches(*keys):
        blob = " ".join(interests)
        return any(k in blob for k in keys)

    # --- Generators for each stage ---
    def generate_for_10th():
        """
        Return 10 stream suggestions. Each item:
          { "stream": str, "overview": str, "key_skills": [...], "career_paths": [...], "future_scope": str }
        """
        streams = []

        # Prioritise suggestions from explicit interests
        if interest_matches("computer", "coding", "programming", "ai", "machine"):
            streams.append({
                "stream": "PCM + Computer Science (PCM-CS)",
                "overview": "Science stream with Mathematics and foundational Computer Science exposure — suited for students who enjoy problem solving & coding.",
                "key_skills": ["Logical reasoning", "Mathematics", "Basic programming concepts", "Analytical thinking"],
                "career_paths": ["B.Tech CSE / IT", "BSc (Computer Science)", "Data Science", "Software Developer"],
                "future_scope": "Strong demand in software, AI, startups, and research; flexible for interdisciplinary paths."
            })
        # Add other common streams
        streams.append({
            "stream": "PCM (Engineering-focused)",
            "overview": "Traditional science stream focused on Physics, Chemistry and Mathematics — a general foundation for engineering disciplines.",
            "key_skills": ["Mathematics", "Physics fundamentals", "Problem solving", "Analytical thinking"],
            "career_paths": ["Engineering (various branches)", "Civil/Mechanical/Electrical Engineer", "R&D roles"],
            "future_scope": "Broad engineering opportunities in manufacturing, infrastructure, automotive, and product design."
        })
        streams.append({
            "stream": "PCB + Maths (Bio + Maths hybrid)",
            "overview": "Biology-centric stream augmented with Mathematics for students interested in biotech, bioinformatics or computational biology.",
            "key_skills": ["Biology basics", "Mathematics", "Statistical thinking", "Laboratory skills"],
            "career_paths": ["Biotechnology", "Bioinformatics", "Research in life sciences", "Healthcare analytics"],
            "future_scope": "Growing intersection of biology and data science — roles in pharma, biotech startups, and research labs."
        })
        streams.append({
            "stream": "PCB (Pure Biology / Medicine track)",
            "overview": "Biology-heavy stream for students aiming at medical, allied health and biological sciences careers.",
            "key_skills": ["Anatomy & physiology basics", "Lab techniques", "Observation skills", "Biological reasoning"],
            "career_paths": ["MBBS / BDS / BSc Nursing / Allied Health", "Research in life sciences"],
            "future_scope": "Traditional medical and allied health careers; steady demand in hospitals, clinics and research institutions."
        })
        streams.append({
            "stream": "Commerce (Accountancy & Business)",
            "overview": "Commerce stream focusing on accounting, economics and business studies — suited for finance, business, and management interests.",
            "key_skills": ["Numeracy", "Basic accounting", "Business awareness", "Analytical thinking"],
            "career_paths": ["Chartered Accountant", "Financial Analyst", "Business Management", "Banking"],
            "future_scope": "Strong demand in finance, accounting, fintech and corporate roles; good for entrepreneurial routes."
        })
        streams.append({
            "stream": "Commerce + Maths (Economics / Finance heavy)",
            "overview": "Commerce with Mathematics to enable analytical and quantitative careers (economics, finance, data in commerce).",
            "key_skills": ["Statistics", "Mathematical reasoning", "Economics basics", "Data interpretation"],
            "career_paths": ["Economist (academia/industry)", "Quantitative Finance", "Data roles in finance"],
            "future_scope": "Good fit for analytics-heavy finance roles and competitive commerce streams."
        })
        streams.append({
            "stream": "Arts / Humanities (Social Sciences)",
            "overview": "Humanities-focused stream for interests in social sciences, languages, history, and creative fields.",
            "key_skills": ["Critical thinking", "Communication", "Research", "Cultural understanding"],
            "career_paths": ["Law", "Public Policy", "Journalism", "Academia", "Social Work"],
            "future_scope": "Flexible career paths across public sector, NGOs, media and academia."
        })
        streams.append({
            "stream": "Arts - Design & Creative (Fine arts, Design foundation)",
            "overview": "Focus on art, design fundamentals and creative skills — suitable for students inclined toward visual arts and design thinking.",
            "key_skills": ["Visual design", "Sketching", "Creativity", "Design thinking"],
            "career_paths": ["Graphic Designer", "Product Designer", "Animator", "Fashion / Textile Design"],
            "future_scope": "Growing opportunities in product design, UI/UX, animation, advertising and creative startups."
        })
        streams.append({
            "stream": "Vocational / IT (Skill-based diplomas)",
            "overview": "Skill and career-oriented stream that focuses on vocational training, IT diplomas and practical skills for earlier employment.",
            "key_skills": ["Basic IT skills", "Hands-on technical training", "Soft skills", "Domain-specific tools"],
            "career_paths": ["Diploma in IT / Polytechnic routes", "Technical support", "Web developer (entry-level)"],
            "future_scope": "Fast route to technical jobs, apprenticeships, and industry-specific roles; good for hands-on learners."
        })
        streams.append({
            "stream": "Hospitality / Agriculture / Allied Sciences",
            "overview": "Applied streams focusing on hospitality management, agriculture sciences, or allied vocational areas.",
            "key_skills": ["Operational skills", "Domain-specific techniques", "Customer-facing abilities"],
            "career_paths": ["Hospitality manager", "Agricultural scientist", "Food technologist"],
            "future_scope": "Steady regional and national demand; good for industry-specific careers and entrepreneurship."
        })

        # Tailor ordering slightly by top trait
        if top_trait == "openness":
            # emphasize creative/novel paths earlier
            streams = sorted(streams, key=lambda x: 0 if "Design" in x["stream"] or "openness" else 1)
        return streams[:10]

    def generate_for_12th():
        """
        Return 10 course suggestions. Each item:
          { "course": str, "overview": str, "curriculum": {...}, "top_colleges": [...], "career_opportunities": [...], "eligibility": {...} }
        """
        items = []

        # B.Tech / CSE
        items.append({
            "course": "B.Tech / B.E. — Computer Science & Engineering (CSE)",
            "overview": "Undergraduate engineering degree focusing on computation, algorithms, systems and software engineering.",
            "curriculum": {
                "1st Year": ["Mathematics", "Physics", "Programming fundamentals", "Engineering drawing / basics"],
                "2nd Year": ["Data Structures", "Discrete Maths", "Digital Logic", "Database Management"],
                "3rd Year": ["Algorithms", "Operating Systems", "Software Engineering", "Machine Learning basics"],
                "4th Year": ["Advanced CS electives", "Distributed Systems / Cloud", "Capstone Project", "Industry Internship"]
            },
            "top_colleges": ["IITs (various)", "NITs", "IIIT Hyderabad", "BITS Pilani", "VIT"],
            "career_opportunities": ["Software Engineer", "Systems Developer", "Data Scientist", "ML Engineer", "Research"],
            "eligibility": {
                "educational_requirements": "12th Grade with Physics, Mathematics and a second science (usually Chemistry/Computer Science).",
                "minimum_percent": "Varies by institute (competitive admissions typically require high scores/entrance rank).",
                "entrance_exams": ["JEE Main", "JEE Advanced (for IITs)", "Institute-specific tests (BITSAT, VITEEE)"],
                "prerequisites": "Strong maths foundation, basic programming aptitude, logical thinking"
            }
        })

        # B.Sc / Data Science
        items.append({
            "course": "B.Sc / B.Tech — Data Science / Analytics",
            "overview": "Interdisciplinary course combining statistics, programming, and machine learning to analyze real-world data.",
            "curriculum": {
                "1st Year": ["Mathematics", "Statistics", "Programming fundamentals", "Intro to Data"],
                "2nd Year": ["Database Management", "Data Structures", "Machine Learning foundations", "Data Visualization"],
                "3rd Year": ["Linear Algebra", "Statistical Modeling", "Big Data tools", "Applied ML"],
                "4th Year": ["Deep Learning", "Capstone Project", "Industry Internship", "Electives in domain analytics"]
            },
            "top_colleges": ["IITs (selected programs)", "IIITs", "BITS Pilani", "ISB / Universities offering specialised BSc programmes", "Top private universities"],
            "career_opportunities": ["Data Scientist", "Data Analyst", "ML Engineer", "Business Analyst"],
            "eligibility": {
                "educational_requirements": "12th Grade (Science/Maths recommended).",
                "required_subjects": ["Mathematics recommended; Computer Science helpful"],
                "entrance_exams": ["Institute-specific tests; some universities via merit/entrance"],
                "prerequisites": "Mathematics, basic programming, statistical aptitude"
            }
        })

        # B.Com
        items.append({
            "course": "B.Com (Honours) / Commerce",
            "overview": "Commerce undergraduate focusing on accounting, finance, economics and business law.",
            "curriculum": {
                "1st Year": ["Financial Accounting", "Business Economics", "Business Law basics"],
                "2nd Year": ["Cost Accounting", "Corporate Law", "Taxation basics"],
                "3rd Year": ["Auditing", "Financial Management", "Electives (Banking / Finance)"]
            },
            "top_colleges": ["SRCC (DU)", "St. Xavier's", "Loyola", "Top commerce colleges across state universities", "Private universities (varied)"],
            "career_opportunities": ["Accountant", "Financial Analyst", "Company Secretary (further exams)"],
            "eligibility": {
                "educational_requirements": "12th Grade (any stream, commerce preferred).",
                "required_subjects": ["Not strictly enforced; commerce subjects helpful"],
                "entrance_exams": ["College-specific entrance/merit lists"],
                "prerequisites": "Basic numeracy and interest in commerce/finance"
            }
        })

        # B.Des
        items.append({
            "course": "B.Des / Design (Industrial / Product / Graphic / Fashion)",
            "overview": "Undergraduate design degree teaching design thinking, visual language and applied creativity.",
            "curriculum": {
                "1st Year": ["Design fundamentals", "Drawing & Visualization", "History of Design"],
                "2nd Year": ["Materials & Processes", "Digital Tools", "User-centred design"],
                "3rd Year": ["Specialisation electives", "Workshops", "Industry projects"],
                "4th Year": ["Portfolio, Thesis project, Internship", "Electives"]
            },
            "top_colleges": ["NID", "NIFT", "IITs (Design departments)", "Symbiosis School of Design", "Srishti Institute"],
            "career_opportunities": ["Product Designer", "UX/UI Designer", "Graphic Designer", "Fashion Designer"],
            "eligibility": {
                "educational_requirements": "12th Grade (any stream).",
                "entrance_exams": ["NID / NIFT / institute-specific entrance tests", "Portfolio rounds"],
                "prerequisites": "Creative portfolio, design aptitude"
            }
        })

        # MBBS / Medical
        items.append({
            "course": "MBBS / Medicine (for students focused on clinical careers)",
            "overview": "Professional degree leading to clinical practice as a physician/doctor.",
            "curriculum": {
                "pre-clinical": ["Anatomy", "Physiology", "Biochemistry"],
                "para-clinical": ["Pathology", "Pharmacology", "Microbiology"],
                "clinical": ["Medicine, Surgery, Pediatrics, Obstetrics & Gynecology"],
                "internship": ["Compulsory rotatory internship in clinical departments"]
            },
            "top_colleges": ["AIIMS (various)", "Top government medical colleges (varies by state)", "Private medical colleges (varied)"],
            "career_opportunities": ["Clinician (MD/MS after MBBS)", "Public Health", "Research", "Healthcare management"],
            "eligibility": {
                "educational_requirements": "12th Grade with Biology, Chemistry, Physics.",
                "entrance_exams": ["NEET-UG (national-level)"],
                "prerequisites": "Strong biology foundation, commitment to clinical work"
            }
        })

        # B.Arch
        items.append({
            "course": "B.Arch (Architecture)",
            "overview": "Undergraduate professional degree in architecture, combining design, structure and planning.",
            "curriculum": {
                "1st Year": ["Design basics", "Visual representation", "Mathematics"],
                "2nd-4th Years": ["Architectural design", "History of architecture", "Construction technology"],
                "Final Year": ["Thesis project", "Professional practice", "Internship"]
            },
            "top_colleges": ["Architecture colleges under state universities", "IITs (architecture where available)", "CEPT University", "SPA / Top private institutes"],
            "career_opportunities": ["Architect", "Urban Planner", "Interior Designer", "Conservation Specialist"],
            "eligibility": {
                "educational_requirements": "12th Grade with Mathematics (usually required).",
                "entrance_exams": ["NATA / JEE Paper 2 (for some institutes)"],
                "prerequisites": "Spatial ability, drawing skills"
            }
        })

        # B.Sc — Life Sciences
        items.append({
            "course": "B.Sc — Life Sciences / Biotechnology",
            "overview": "Undergraduate focus on biological sciences, lab skills and research fundamentals.",
            "curriculum": {
                "1st Year": ["Cell Biology", "Chemistry", "Biostatistics basics"],
                "2nd Year": ["Genetics", "Microbiology", "Lab techniques"],
                "3rd Year": ["Biotechnology applications", "Research project / internship"]
            },
            "top_colleges": ["Top universities with life-science programs (IISC-associated programs, Delhi University colleges, private universities)"],
            "career_opportunities": ["Research Assistant", "Biotech industry roles", "Lab technician", "Further MSc/PhD"],
            "eligibility": {
                "educational_requirements": "12th Grade with Biology (preferred).",
                "prerequisites": "Interest in lab work and research methods"
            }
        })

        # BBA / Business
        items.append({
            "course": "BBA / Management (Business Administration)",
            "overview": "Undergraduate business degree teaching management fundamentals and soft skills for corporate careers.",
            "curriculum": {
                "1st Year": ["Management basics", "Economics", "Business communication"],
                "2nd Year": ["Marketing fundamentals", "Accounting", "HR basics"],
                "3rd Year": ["Strategic Management", "Projects", "Internship"]
            },
            "top_colleges": ["Top private management colleges, university BBA programs, institute-specific programs"],
            "career_opportunities": ["Business Analyst", "HR Executive", "Marketing Coordinator", "Entrepreneurship"],
            "eligibility": {
                "educational_requirements": "12th Grade (any stream); some institutes have entrance tests.",
                "prerequisites": "Communication skills, basic numeracy"
            }
        })

        # Polytechnic / Diploma applied courses
        items.append({
            "course": "Polytechnic / Diploma (Engineering or Applied Vocational Courses)",
            "overview": "Shorter, practical diploma programmes that prepare students for industry roles or lateral entry into degree courses.",
            "curriculum": {
                "year_1": ["Fundamentals & hands-on labs"],
                "year_2": ["Core technical subjects", "Workshops"],
                "year_3": ["Project work", "Industry training / internship"]
            },
            "top_colleges": ["State polytechnic institutes, reputed private polytechnics, industrial training centres"],
            "career_opportunities": ["Technician roles", "Diploma engineer roles", "Shop-floor engineering", "Lateral entry into degree programs"],
            "eligibility": {
                "educational_requirements": "10th/12th depending on the program.",
                "prerequisites": "Interest in hands-on technical skills"
            }
        })

        return items[:10]

    def generate_for_college():
        """
        Return >=10 career entries for college students. Each item:
          { "career": str, "overview": str, "roles": [...], "skills": [...], "top_companies": [...], "typical_pathway": str }
        """
        careers = []

        careers.append({
            "career": "Software Engineer",
            "overview": "Develop, maintain and scale software products across platforms.",
            "roles": ["Backend Developer", "Frontend Developer", "Full-stack Engineer", "SRE/Platform Engineer"],
            "skills": ["Programming (Python/Java/JS/etc.)", "Data Structures & Algorithms", "System design", "Version control"],
            "top_companies": ["Google", "Microsoft", "Amazon", "Infosys", "TCS", "Flipkart"],
            "typical_pathway": "B.Tech / BSc in CS → Internships → Entry-level SWE → Mid/Senior roles / specialization"
        })
        careers.append({
            "career": "Data Scientist / ML Engineer",
            "overview": "Extract insights from data and build predictive models for products and business decisions.",
            "roles": ["Data Scientist", "ML Engineer", "Research Scientist"],
            "skills": ["Statistics", "Machine Learning", "Python / R", "SQL", "Model deployment"],
            "top_companies": ["Amazon", "Google", "Microsoft", "Fractal Analytics", "Mu Sigma", "Accenture"],
            "typical_pathway": "BTech / BSc + internships → Data Analyst → Data Scientist / ML Engineer"
        })
        careers.append({
            "career": "Product Manager",
            "overview": "Define product vision, coordinate engineering and design, and measure impact.",
            "roles": ["Associate PM", "Product Manager", "Group PM", "Technical PM"],
            "skills": ["Product sense", "Metrics & analytics", "Stakeholder management", "Market research"],
            "top_companies": ["Google", "Amazon", "Flipkart", "Microsoft", "Swiggy"],
            "typical_pathway": "Engineering degree / MBA / cross-functional experience → PM roles via internships or lateral moves"
        })
        careers.append({
            "career": "UX / Product Designer",
            "overview": "Design user experiences, interfaces and product workflows grounded in user research.",
            "roles": ["UX Researcher", "UX Designer", "UI Designer", "Product Designer"],
            "skills": ["User research", "Wireframing & prototyping", "Interaction design", "Design tools & portfolios"],
            "top_companies": ["Google", "Adobe", "Microsoft", "Tata Consultancy Services (design teams)", "UX agencies"],
            "typical_pathway": "B.Des / relevant portfolio → internships → junior designer → senior/product designer"
        })
        careers.append({
            "career": "Management Consultant",
            "overview": "Advise organisations on strategy, operations and growth using data and structured problem solving.",
            "roles": ["Analyst", "Consultant", "Senior Consultant", "Engagement Manager"],
            "skills": ["Problem solving", "Quantitative analysis", "Communication", "Domain knowledge"],
            "top_companies": ["McKinsey", "BCG", "Bain", "Deloitte", "KPMG"],
            "typical_pathway": "Bachelors → internships / small projects → consulting analyst → consultant; MBA common later"
        })
        careers.append({
            "career": "Investment Analyst / Finance Professional",
            "overview": "Analyze companies and markets to support investment decisions, corporate finance and advisory.",
            "roles": ["Equity Research Analyst", "Investment Banking Analyst", "Financial Analyst"],
            "skills": ["Financial modelling", "Accounting", "Excel", "Economic analysis"],
            "top_companies": ["Goldman Sachs", "Morgan Stanley", "ICICI Securities", "HDFC", "JP Morgan"],
            "typical_pathway": "B.Com / B.Tech / Economics → internships → analyst roles; CFA/CA add value"
        })
        careers.append({
            "career": "Chartered Accountant / Accounting Specialist",
            "overview": "Professional accounting, auditing, taxation and financial compliance expertise.",
            "roles": ["CA (practice)", "Tax Consultant", "Internal Auditor", "Financial Controller"],
            "skills": ["Accounting standards", "Taxation", "Audit procedures", "Ethics & compliance"],
            "top_companies": ["Big 4 (Deloitte, PwC, EY, KPMG)", "Corporate finance teams", "Chartered practices"],
            "typical_pathway": "B.Com / Article ship + CA exams → CA qualification → professional roles"
        })
        careers.append({
            "career": "Clinical Doctor / Medical Specialist",
            "overview": "Clinical practice and healthcare delivery after MBBS and specialization.",
            "roles": ["General Physician", "Surgeon (specialized)", "Pediatrician", "Radiologist"],
            "skills": ["Clinical knowledge", "Patient management", "Decision making", "Procedural skills"],
            "top_companies": ["Apollo Hospitals", "Fortis Healthcare", "AIIMS (institutes)", "Multi-specialty hospitals"],
            "typical_pathway": "MBBS → Internship → MD/MS/DM specialisation → clinical practice / research"
        })
        careers.append({
            "career": "Civil / Mechanical / Electrical Engineer (Core Engineering)",
            "overview": "Design, develop and maintain physical infrastructure or mechanical/electrical systems.",
            "roles": ["Design Engineer", "Site Engineer", "R&D Engineer", "Maintenance Engineer"],
            "skills": ["Domain engineering fundamentals", "CAD / simulation", "Project management", "Problem solving"],
            "top_companies": ["L&T", "Tata Motors", "Mahindra", "Siemens", "ABB"],
            "typical_pathway": "B.Tech → internships → campus placements / industry roles → senior engineering roles"
        })
        careers.append({
            "career": "Entrepreneur / Startup Founder",
            "overview": "Identify problems, build products/services and scale a business venture.",
            "roles": ["Founder / Co-founder", "Product lead in startup", "Growth / Operations head"],
            "skills": ["Risk-taking", "Product-market fit", "Fundraising basics", "Leadership"],
            "top_companies": ["(Founders typically build their own companies) — common investors / accelerators include Sequoia India, Accel, Y Combinator alumni"],
            "typical_pathway": "Any degree + domain expertise / internships → startup roles → founding a company"
        })

        return careers[:10]

    # Generate the output according to stage
    if stage_key == "10th":
        results = generate_for_10th()
        # Standardize output objects for compatibility with existing client expectations
        final = [{"stream": r["stream"], "overview": r["overview"], "key_skills": r["key_skills"],
                  "career_paths": r["career_paths"], "future_scope": r["future_scope"]} for r in results]
    elif stage_key == "12th":
        results = generate_for_12th()
        final = []
        for r in results:
            final.append({
                "course": r.get("course"),
                "overview": r.get("overview"),
                "curriculum": r.get("curriculum"),
                "top_colleges": r.get("top_colleges"),
                "career_opportunities": r.get("career_opportunities"),
                "eligibility": r.get("eligibility")
            })
    else:  # college
        results = generate_for_college()
        final = []
        for r in results:
            final.append({
                "career": r.get("career"),
                "overview": r.get("overview"),
                "roles": r.get("roles"),
                "skills": r.get("skills"),
                "top_companies": r.get("top_companies"),
                "typical_pathway": r.get("typical_pathway")
            })

    # Ensure at least 10 items (trim/pad deterministically if needed)
    if len(final) < 10:
        # pad with simple deterministic suggestions derived from top_trait/interests
        pad_source = {
            "10th": {
                "stream": "General Science (flexible)",
                "overview": "Flexible science stream keeping options open for both engineering and life sciences.",
                "key_skills": ["Mathematics", "Science reasoning"],
                "career_paths": ["Multiple downstream choices"],
                "future_scope": "Keeps options open for competitive engineering/medical/analytics paths."
            },
            "12th": {
                "course": "Interdisciplinary / Foundation program",
                "overview": "Foundation coursework allowing exploration across STEM/Commerce/Arts.",
                "curriculum": {"Year1": ["Foundation modules", "Domain exploration"]},
                "top_colleges": ["Various universities offering foundation programs"],
                "career_opportunities": ["Flexible pathways into degree courses"],
                "eligibility": {"educational_requirements": "12th Grade"}
            },
            "college": {
                "career": "Generalist / Operations roles",
                "overview": "Roles that use broad management and operations skills.",
                "roles": ["Operations Executive", "Generalist"],
                "skills": ["Coordination", "Basic analytics", "Communication"],
                "top_companies": ["Multiple mid-size companies"],
                "typical_pathway": "Any undergraduate degree → internships → operations roles"
            }
        }
        needed = 10 - len(final)
        for _ in range(needed):
            final.append(pad_source[stage_key])

    # Save summarized OCEAN results to memory as before
    try:
        await memory_add(
            [{"role": "system", "content": f"OCEAN Test Results: {raw_ocean}, Stage: {stage_key}, Interests: {interests_text}"}],
            user_id=data.user_id,
            metadata={"bot": "careerbot", "stage": stage_key, "ocean": raw_ocean, "interests": interests}
        )
    except Exception:
        logger.exception("Failed to store ocean test results in memory")

    return OceanTestResponse(user_id=data.user_id, top_jobs=final)

# ---------------------------
# Main entry
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, log_level="debug" if DEBUG else "info")