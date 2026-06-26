"""
LLMOps layer: structured JSON logging + MLFlow tracing.

One JSON line is emitted per LLM call to stdout (satisfies the grading
requirement) and every attribute is also stored as a span in the local
SQLite-backed MLFlow database so you can inspect traces via:

    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import json
import time
import logging
import mlflow
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("LLMOps")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

import os as _os
_mlflow_uri = _os.environ.get(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db",  # local default
)
mlflow.set_tracking_uri(_mlflow_uri)
mlflow.set_experiment("School_Handbook_Agent")


class OpsCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback that fires around every LLM call.

    Lifecycle
    ---------
    on_llm_start  → record wall-clock start; open an MLFlow span
    on_llm_end    → compute latency & tokens; log JSON; close span
    on_llm_error  → close span on failure so it isn't left dangling
    """

    def __init__(self):
        self._start_time: float | None = None
        self._span = None

    # ------------------------------------------------------------------
    # LLM start
    # ------------------------------------------------------------------
    def on_llm_start(self, serialized, prompts, **kwargs):
        self._start_time = time.time()
        try:
            if mlflow.active_run() is None:
                mlflow.start_run(run_name="llm_call", nested=True)
            self._span = mlflow.start_span(name="llm_inference", span_type="LLM")
        except Exception as exc:
            logger.debug(f"MLFlow span open error (non-fatal): {exc}")
            self._span = None

    # ------------------------------------------------------------------
    # LLM end — main logging point
    # ------------------------------------------------------------------
    def on_llm_end(self, response, **kwargs):
        latency_ms = round((time.time() - self._start_time) * 1000, 2) if self._start_time else 0.0

        # Token counts live in generation_info for Ollama
        prompt_tokens, completion_tokens = 0, 0
        try:
            gen_info = response.generations[0][0].generation_info or {}
            prompt_tokens     = gen_info.get("prompt_eval_count", 0)
            completion_tokens = gen_info.get("eval_count", 0)
        except (IndexError, AttributeError):
            pass

        total_tokens = prompt_tokens + completion_tokens

        # Ollama is local/free; cost is 0.
        estimated_cost_usd = 0.0

        log_record = {
            "latency_ms":        latency_ms,
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      total_tokens,
            "model":             "gemma3:1b",
            "estimated_cost_usd": estimated_cost_usd,
        }

        # Requirement 3: one JSON line per request to stdout
        logger.info(json.dumps(log_record))

        # Store in MLFlow span
        if self._span is not None:
            try:
                self._span.set_attributes(log_record)
                self._span.end()
            except Exception as exc:
                logger.debug(f"MLFlow span close error (non-fatal): {exc}")
            finally:
                self._span = None

    # ------------------------------------------------------------------
    # Error path — always close the span
    # ------------------------------------------------------------------
    def on_llm_error(self, error, **kwargs):
        if self._span is not None:
            try:
                self._span.set_attribute("error", str(error))
                self._span.end()
            except Exception:
                pass
            finally:
                self._span = None


# --------------------------------------------------------------------------
# Convenience context manager used by api.py / app.py to wrap an entire
# request (RAG retrieval → LLM inference) in one parent MLFlow span so the
# trace tree in the UI mirrors the architecture diagram from the slides.
# --------------------------------------------------------------------------
class RequestTrace:
    """
    Usage::

        with RequestTrace("chat_request") as trace:
            with trace.child("rag_retrieval", "RETRIEVER") as rag_span:
                rag_span.set_attribute("query", query)
            # llm inference span opened automatically by OpsCallbackHandler
    """

    def __init__(self, name: str = "chat_request"):
        self.name = name
        self._run = None
        self._span = None

    def __enter__(self):
        try:
            self._run = mlflow.start_run(run_name=self.name)
            self._span = mlflow.start_span(name=self.name, span_type="CHAIN")
        except Exception as exc:
            logger.debug(f"RequestTrace open error (non-fatal): {exc}")
        return self

    def child(self, name: str, span_type: str = "TOOL"):
        """Return an MLFlow span context manager for a child step."""
        try:
            return mlflow.start_span(name=name, span_type=span_type)
        except Exception:
            return _NoOpSpan()

    def __exit__(self, *_):
        if self._span is not None:
            try:
                self._span.end()
            except Exception:
                pass
        if self._run is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass


class _NoOpSpan:
    """Fallback when MLFlow is unavailable — absorbs all attribute calls."""
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def set_attribute(self, *_): pass
    def set_attributes(self, *_): pass
    def end(self): pass