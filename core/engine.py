"""
RAG + Memory + Guardrails engine.

Week 3 patterns: SemanticChunker, ChromaDB, similarity search
Week 4 patterns: ConversationSummaryBufferMemory, PII redaction,
                 keyword blocking, topic classification
Week 5 addition: RequestTrace wrapping each request so MLFlow shows
                 rag_retrieval → llm_inference in the trace tree
"""

import re
import json
import logging

from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

from core.llmops import OpsCallbackHandler, RequestTrace

logger = logging.getLogger("engine")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
LLM_MODEL   = "gemma3:1b"
EMBED_MODEL = "nomic-embed-text"
CHROMA_DIR  = "./chroma_db"

# --------------------------------------------------------------------------
# Singletons — initialised once at import time
# --------------------------------------------------------------------------
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# Main LLM — carries the OpsCallbackHandler so every generation is logged
llm_main = Ollama(
    model=LLM_MODEL,
    temperature=0.5,
    callbacks=[OpsCallbackHandler()],
)

# Classifier LLM — no callback so topic checks don't pollute the trace
llm_classifier = Ollama(model=LLM_MODEL, temperature=0.0)

# ChromaDB — gracefully absent until a PDF is ingested
try:
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
except Exception as exc:
    db = None
    logger.warning(f"ChromaDB not ready yet (will initialise on first PDF upload): {exc}")

# Week 4: summary buffer keeps the conversation manageable in token terms
memory = ConversationSummaryBufferMemory(
    llm=llm_main,
    max_token_limit=500,
    return_messages=False,
    human_prefix="user",
    ai_prefix="assistant",
)

# --------------------------------------------------------------------------
# Week 4 Guardrails
# --------------------------------------------------------------------------

def redact_pii(text: str) -> str:
    """Replace common Philippine PII patterns with [REDACTED]."""
    # PH mobile numbers
    text = re.sub(r'\b(?:\+63[-\s]?|0)9\d{2}[-.\s]?\d{3,4}[-.\s]?\d{4}\b', '[REDACTED]', text)
    # Email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[REDACTED]', text)
    # Age mentions
    text = re.sub(r'\b\d{1,3}[ \-]years?[ \-]old\b', '[REDACTED]', text, flags=re.IGNORECASE)
    # Street addresses
    text = re.sub(
        r'\b\d+\s+[A-Za-z][A-Za-z ]+?(?:St(?:reet)?|Ave(?:nue)?|Blvd|Road|Rd|Drive|Dr|Lane|Ln)\.?\b',
        '[REDACTED]', text, flags=re.IGNORECASE,
    )
    # "My name is / I am / I'm <Firstname Lastname>"
    text = re.sub(
        r'(My name is|my name is|I am|I\'m)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'\1 [REDACTED]', text,
    )
    # Philippine student/ID numbers (####-#######-#)
    text = re.sub(r'\d{4}-\d{7}-\d{1}', '[REDACTED]', text)
    return text


BLOCK_KW = [
    'do my homework', 'write my essay', 'bypass', 'override rules', 'override your',
    'disregard all previous', 'forget all previous' 'ignore previous', 'ignore your', 'ignore all',
    'unrestricted ai', 'you have no restrictions', 'do anything now', 'dan mode',
    'you are now', 'from now on you are', 'act as if', 'pretend you are',
    'pretend to be', 'roleplay as', 'forget your instructions', 'forget previous',
    'skip the rules', 'your new instructions', 'new persona',
    'diagnose', 'do i have', 'prescribe', 'treatment for', 'cure',
    'write an essay', 'write my essay', 'do my homework', 
    'write a paper', 'draft an essay', 'create an essay',
    'help me write', 'essay about', 'academic paper'
]


def is_blocked_request(text: str) -> tuple[bool, str]:
    lower = text.lower()
    for kw in BLOCK_KW:
        if kw in lower:
            return True, kw
    return False, ''


SCHOOL_TOPICS: dict[str, str] = {
    "GOVERNANCE":      "School mission, vision, core values.",
    "GRADES":          "Grading framework, letter grades, percentage ranges.",
    "POLICIES":        "Academic probation, progress monitoring, tutoring.",
    "CODE_OF_CONDUCT": "Honor code, attendance rules, Oakridge Pledge.",
    "UNIFORM":         "Daily school uniform, Monday–Thursday attire, Friday attire.",
    "OPERATIONS":      "Campus health, emergency drills, facilities.",
    "OFF_TOPIC":       "Anything unrelated to the school handbook CONTEXT/POLICIES — real-world events, entertainment, sports, recipes, generative requests etc.",
}

_TOPIC_SYSTEM = (
    "You are a topic classifier for a school handbook chatbot.\n\n"
    "Classify the user message into EXACTLY ONE topic:\n"
    + "\n".join(f"- {k}: {v}" for k, v in SCHOOL_TOPICS.items())
    + "\n\nRespond with ONLY a JSON object:\n"
      '{"topic": "TOPIC_NAME", "allowed": true, "confidence": 0.95}\n\n'
      "Rules:\n"
      "- allowed=true for GOVERNANCE, GRADES, POLICIES, CODE_OF_CONDUCT, UNIFORM, OPERATIONS\n"
      "- allowed=false for OFF_TOPIC\n"
      "- confidence is a float 0–1"
)

_TOPIC_FEW_SHOT = [
    {"role": "user",      "content": "What is the minimum grade to pass?"},
    {"role": "assistant", "content": '{"topic": "POLICIES", "allowed": true, "confidence": 0.95}'},
    {"role": "user",      "content": "What happens if a student cheats on an exam?"},
    {"role": "assistant", "content": '{"topic": "CODE_OF_CONDUCT", "allowed": true, "confidence": 0.96}'},
    {"role": "user",      "content": "What is the passing grade without failing?"},
    {"role": "assistant", "content": '{"topic": "GRADES", "allowed": true, "confidence": 0.94}'},
    {"role": "user",      "content": "What is the school mission statement?"},
    {"role": "assistant", "content": '{"topic": "GOVERNANCE", "allowed": true, "confidence": 0.94}'},
    {"role": "user",      "content": "What is the required uniform on Fridays?"},
    {"role": "assistant", "content": '{"topic": "OPERATIONS", "allowed": true, "confidence": 0.95}'},
    {"role": "user",      "content": "Who won the NBA finals?"},
    {"role": "assistant", "content": '{"topic": "OFF_TOPIC", "allowed": false, "confidence": 0.98}'},
    {"role": "user",      "content": "What is the best recipe for adobo?"},
    {"role": "assistant", "content": '{"topic": "OFF_TOPIC", "allowed": false, "confidence": 0.97}'},
    {"role": "user",      "content": "What is the current bitcoin price?"},
    {"role": "assistant", "content": '{"topic": "OFF_TOPIC", "allowed": false, "confidence": 0.99}'},
    {"role": "user",      "content": "Hi my name is [REDACTED]. What is the grading system?"},
    {"role": "assistant", "content": '{"topic": "GRADES", "allowed": true, "confidence": 0.95}'},
    {"role": "user",      "content": "Can you create an essay?"},
    {"role": "assistant", "content": '{"topic": "OFF_TOPIC", "allowed": false, "confidence": 0.98}'},
]


def is_on_topic(user_message: str) -> dict:
    messages = [
        {"role": "system", "content": _TOPIC_SYSTEM},
        *_TOPIC_FEW_SHOT,
        {"role": "user", "content": user_message},
    ]
    response = llm_classifier.invoke(messages)
    print(response)
    try:
        match = re.search(r'\{.*?\}', response, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in classifier response")
        result = json.loads(match.group())
        if result.get("topic") not in SCHOOL_TOPICS:
            raise ValueError(f"Unknown topic: {result.get('topic')}")
        return result
    except (json.JSONDecodeError, ValueError):
        # Fail open so classifier bugs don't brick the chatbot
        return {"topic": "UNKNOWN", "allowed": True, "confidence": 0.0, "fallback": True}


def input_guard(text: str) -> str:
    """
    Run all input-side guardrails.  Returns the cleaned text or raises
    ValueError with a human-readable reason.
    """
    blocked, kw = is_blocked_request(text)
    if blocked:
        raise ValueError(f"Request blocked: '{kw}' is not permitted.")

    clean = redact_pii(text)

    topic_result = is_on_topic(clean)
    if not topic_result.get("allowed", True):
        raise ValueError(
            f"Off-topic request blocked (topic: '{topic_result.get('topic')}')."
            " I can only answer questions about school policies."
        )

    return clean


def output_guard(response: str) -> str:
    """
    Run all output-side guardrails.  Returns a safe version of the response.
    """
    response = redact_pii(response)
    blocked, _ = is_blocked_request(response)
    if blocked:
        return "I cannot provide that information. Please refer to school administration directly."
    return response


# --------------------------------------------------------------------------
# RAG prompt & chain
# --------------------------------------------------------------------------
_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful, factual AI assistant that answers school policy-related questions.
Answer using only the retrieved context below. If the context does not contain
the answer, say so clearly — do not make up information.
Use logical reasoning to answer the question. Make sure your response sufficiently answers the question and makes sense.
Do not make up answers or provide alternatives.
When referring to the context, call it "the school handbook".
Answer concisely in fewer than three sentences.

---
RETRIEVED CONTEXT:
{context}

---
CONVERSATION HISTORY:
{history}

---
USER QUESTION:
{question}

ANSWER:
""")

_chain = _PROMPT | llm_main | StrOutputParser()


# --------------------------------------------------------------------------
# Public streaming generator (called by both app.py and api.py)
# --------------------------------------------------------------------------
def generate_chat_stream(query: str):
    """
    Generator that yields response text chunks.

    Wraps the full request in an MLFlow RequestTrace so the UI shows:
        chat_request
          └─ rag_retrieval   (RETRIEVER span)
          └─ llm_inference   (LLM span — opened by OpsCallbackHandler)

    Raises nothing — errors are yielded as emoji-prefixed strings so the
    UI can display them gracefully.
    """
    with RequestTrace("chat_request") as trace:
        try:
            # 1. Input guardrails
            clean_query = input_guard(query)

            # 2. RAG retrieval — logged as its own child span
            retrieved_context = "No handbook has been uploaded yet."
            if db is not None:
                with trace.child("rag_retrieval", "RETRIEVER") as rag_span:
                    results = db.similarity_search(clean_query, k=3)
                    retrieved_context = "\n\n".join(d.page_content for d in results)
                    try:
                        rag_span.set_attribute("query", clean_query)
                        rag_span.set_attribute("num_results", len(results))
                    except Exception:
                        pass

            # 3. Conversation memory
            history = memory.load_memory_variables({})["history"]

            # 4. Streaming LLM call (OpsCallbackHandler opens llm_inference span)
            full_response = ""
            for chunk in _chain.stream({
                "context":  retrieved_context,
                "history":  history,
                "question": clean_query,
            }):
                full_response += chunk
                yield chunk

            # 5. Output guardrail & memory update
            safe_response = output_guard(full_response)
            memory.save_context(
                {"input": clean_query},
                {"output": safe_response},
            )

        except ValueError as exc:
            yield f"🚨 {exc}"
        except Exception as exc:
            logger.exception("Unexpected error in generate_chat_stream")
            yield f"🚨 An unexpected error occurred: {exc}"


# --------------------------------------------------------------------------
# PDF ingestion
# --------------------------------------------------------------------------
def process_pdf(file_path: str) -> int:
    """
    Load a PDF, chunk it with SemanticChunker, and upsert into ChromaDB.
    Returns the number of chunks ingested.
    """
    global db

    # Week 3: PyPDFLoader + SemanticChunker
    loader = PyPDFLoader(file_path)
    docs   = loader.load()

    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=50,
        min_chunk_size=300,
    )
    chunks = text_splitter.split_documents(docs)

    if db is None:
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
        )
    else:
        db.add_documents(chunks)

    logger.info(f"Ingested {len(chunks)} chunks from {file_path}")
    return len(chunks)


# --------------------------------------------------------------------------
# Handbook initialisation — called once at app startup
# --------------------------------------------------------------------------
import os as _os

# Canonical locations to search for the handbook, in priority order.
# 1. Working directory (where `streamlit run app.py` / `uvicorn` is invoked)
# 2. Directory containing this engine.py file
_HANDBOOK_CANDIDATES = [
    _os.path.join(_os.getcwd(), "student_handbook.pdf"),
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "student_handbook.pdf"),
]


def _find_handbook() -> str | None:
    """Return the first existing handbook path, or None."""
    for candidate in _HANDBOOK_CANDIDATES:
        if _os.path.exists(candidate):
            return _os.path.abspath(candidate)
    return None


def initialize_handbook() -> dict:
    """
    Locate and ingest student_handbook.pdf into ChromaDB.

    Returns a status dict consumed by app.py:
        {"ok": True,  "path": "...", "chunks": 42}
        {"ok": False, "error": "...", "searched": [...]}

    Safe to call multiple times — if ChromaDB already contains data from a
    previous run (persisted on disk) this is a no-op and returns immediately.
    """
    # If ChromaDB already has documents we don't need to re-ingest
    if db is not None:
        try:
            count = db._collection.count()
            if count > 0:
                logger.info(f"ChromaDB already populated ({count} docs) — skipping ingest.")
                return {"ok": True, "path": "cached", "chunks": count}
        except Exception:
            pass  # can't count — fall through and ingest

    path = _find_handbook()
    if path is None:
        searched = [_os.path.abspath(c) for c in _HANDBOOK_CANDIDATES]
        return {
            "ok": False,
            "error": "student_handbook.pdf not found.",
            "searched": searched,
        }

    try:
        n = process_pdf(path)
        return {"ok": True, "path": path, "chunks": n}
    except Exception as exc:
        logger.exception("Failed to ingest handbook")
        return {"ok": False, "error": str(exc), "searched": [path]}