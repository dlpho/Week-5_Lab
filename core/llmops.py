import json
import time
import logging
import mlflow
from langchain.callbacks.base import BaseCallbackHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLMOps")

class OpsCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.start_time = None
        self.mlflow_run = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start_time = time.time()
        # Start MLFlow span
        mlflow.set_tracking_uri("sqlite:///mlflow_traces.db")
        mlflow.set_experiment("School_Handbook_Chatbot")
        self.mlflow_run = mlflow.start_span(name="llm_inference", span_type="LLM")

    def on_llm_end(self, response, **kwargs):
        latency_ms = (time.time() - self.start_time) * 1000
        
        # Estimate tokens (Ollama local usage)
        prompt_tokens = response.llm_output.get("prompt_eval_count", 0) if response.llm_output else 0
        completion_tokens = response.llm_output.get("eval_count", 0) if response.llm_output else 0
        
        log_data = {
            "latency_ms": round(latency_ms, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "model": "gemma3:1b",
            "estimated_cost_usd": 0.0 # Local models are free
        }
        
        # 1. Structured JSON Log
        logger.info(json.dumps(log_data))
        
        # 2. MLFlow Trace Logging
        if self.mlflow_run:
            self.mlflow_run.set_attributes(log_data)
            self.mlflow_run.end()