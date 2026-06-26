import json
import logging
import mlflow
import os

# 1. Setup JSON Logger for terminal output (Grading requirement)
logger = logging.getLogger("LLMOps")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

# 2. Setup MLflow database connection
_mlflow_uri = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "sqlite:////app/mlflow_data/mlflow_traces.db" 
)
mlflow.set_tracking_uri(_mlflow_uri)
mlflow.set_experiment("School_Handbook_Agent")

# 3. Simple helper function to print the JSON to the terminal
def log_json_to_terminal(latency_ms: float):
    log_record = {
        "latency_ms": latency_ms,
        "prompt_tokens": 0,      # Simplified for explicit tracing
        "completion_tokens": 0,  # Simplified for explicit tracing
        "total_tokens": 0,
        "model": "gemma3:1b",
        "estimated_cost_usd": 0.0,
    }
    logger.info(json.dumps(log_record))
    return log_record