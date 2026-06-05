import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "assets" / "uploads"
RENDERED_PAGES_DIR = BASE_DIR / "assets" / "rendered_pages"
ANNOTATIONS_DIR = BASE_DIR / "outputs" / "annotations"
REPOSITORIES_DIR = BASE_DIR / "outputs" / "repositories"

# Create directories if they do not exist
for directory in [UPLOADS_DIR, RENDERED_PAGES_DIR, ANNOTATIONS_DIR, REPOSITORIES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configuration values
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
QWEN_API_BASE = os.getenv("QWEN_API_BASE", "http://localhost:11434/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen2.5-vl")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_ENDPOINT = os.getenv("NVIDIA_ENDPOINT", "https://integrate.api.nvidia.com/v1/chat/completions")
NVIDIA_PRIMARY_MODEL = os.getenv("NVIDIA_PRIMARY_MODEL", "moonshotai/kimi-k2.6")
NVIDIA_PAYLOAD_FORMAT = os.getenv("NVIDIA_PAYLOAD_FORMAT", "array")

# Default values with fallbacks
try:
    RENDER_DPI = int(os.getenv("RENDER_DPI", "200"))
except ValueError:
    RENDER_DPI = 200

try:
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
except ValueError:
    CONFIDENCE_THRESHOLD = 0.70
