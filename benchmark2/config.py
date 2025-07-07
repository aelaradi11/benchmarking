

# Configuration for LLM Benchmark - Local HuggingFace Edition
import os

# ============================================================================
# LOCAL HUGGINGFACE SETTINGS
# ============================================================================

# Use local models instead of API
USE_LOCAL_MODELS = True

# Device configuration
DEVICE = "auto"  # Options: "auto", "cuda", "mps", "cpu"
TORCH_DTYPE = "auto"  # Options: "auto", "float16", "bfloat16", "float32"

# Quantization settings (helps with memory usage)
USE_QUANTIZATION = False  # Set to True to use 4-bit quantization
# QUANTIZATION_BITS = 4    # Options: 4, 8

# ============================================================================
# MODEL CONFIGURATION (Smaller models optimized for local use)
# ============================================================================

# Available models for benchmarking (smaller versions for local inference)
MODELS = {
    'qwen': 'Qwen/Qwen2.5-1.5B-Instruct',        # 0.5B parameters - very fast
}

# ============================================================================
# GENERATION PARAMETERS
# ============================================================================

# Parameters for text generation (optimized for speed and accuracy)
GENERATION_PARAMS = {
    'temperature': 0.1,
    'max_new_tokens': 1000,
    'do_sample': True,
    'top_p': 0.9,
    'pad_token_id': None,
}

# ============================================================================
# REQUEST SETTINGS
# ============================================================================

# Default parameters (used as fallback)
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1000

# Timing settings
DELAY_BETWEEN_REQUESTS = 1  # Shorter delay for local models

# ============================================================================
# FILE PATHS
# ============================================================================

# Path to the questions file
QUESTIONS_FILE = 'final_bloom_questions_cleaned.jsonl'

# Path to the Quran data file
QURAN_FILE = 'quran_cleaned.json'

# Output directory for results
OUTPUT_DIR = 'results'

# ============================================================================
# LEGACY SETTINGS (kept for compatibility)
# ============================================================================

# API settings (not used for local models)
USE_INFERENCE_API = False


# Ollama settings (legacy)
USE_OLLAMA = False
OLLAMA_BASE_URL = 'http://localhost:11434'