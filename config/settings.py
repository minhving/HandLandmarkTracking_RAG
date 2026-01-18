"""
Configuration settings for the Hand Landmark Tracking RAG application.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CHROMA_DB_DIR = BASE_DIR / "data" / "chroma_db"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)

# OpenAI Configuration
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

# MediaPipe Configuration
MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = float(os.getenv("MEDIAPIPE_MIN_DETECTION_CONFIDENCE", "0.7"))
MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = float(os.getenv("MEDIAPIPE_MIN_TRACKING_CONFIDENCE", "0.7"))
MEDIAPIPE_MAX_NUM_HANDS: int = int(os.getenv("MEDIAPIPE_MAX_NUM_HANDS", "1"))

# Drawing Configuration
DRAWING_THRESHOLD: int = int(os.getenv("DRAWING_THRESHOLD", "50"))  # pixels
MIN_DRAWING_POINTS: int = int(os.getenv("MIN_DRAWING_POINTS", "10"))
DRAWING_LINE_THICKNESS: int = int(os.getenv("DRAWING_LINE_THICKNESS", "3"))
DRAWING_POINT_RADIUS: int = int(os.getenv("DRAWING_POINT_RADIUS", "5"))

# OCR Configuration
TESSERACT_CONFIG: str = os.getenv("TESSERACT_CONFIG", "--psm 7")
OCR_DILATION_KERNEL_SIZE: int = int(os.getenv("OCR_DILATION_KERNEL_SIZE", "3"))
OCR_DILATION_ITERATIONS: int = int(os.getenv("OCR_DILATION_ITERATIONS", "1"))

# Recognition Configuration
RECOGNITION_STABILITY_FRAMES: int = int(os.getenv("RECOGNITION_STABILITY_FRAMES", "50"))
GESTURE_SELECTION_FRAMES: int = int(os.getenv("GESTURE_SELECTION_FRAMES", "20"))

# RAG Configuration
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "openai_embeddings")
RAG_N_RESULTS: int = int(os.getenv("RAG_N_RESULTS", "5"))
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))
TRAIN_TEST_SPLIT_RATIO: float = float(os.getenv("TRAIN_TEST_SPLIT_RATIO", "0.2"))
TRAIN_TEST_RANDOM_STATE: int = int(os.getenv("TRAIN_TEST_RANDOM_STATE", "42"))

# File Paths
YOUTUBE_TITLES_CSV: Path = DATA_DIR / os.getenv("YOUTUBE_TITLES_CSV", "youtube_titles.csv")
OUTPUT_JSON: Path = DATA_DIR / os.getenv("OUTPUT_JSON", "output.json")

# Video Configuration
YOUTUBE_BASE_URL: str = "https://www.youtube.com/watch?v="

# Camera Configuration
CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
FRAME_MIRROR: bool = os.getenv("FRAME_MIRROR", "true").lower() == "true"
FRAME_DELAY: float = float(os.getenv("FRAME_DELAY", "0.1"))

# UI Configuration
WINDOW_NAME: str = os.getenv("WINDOW_NAME", "Hand Drawing Recognition")
FULLSCREEN_MODE: bool = os.getenv("FULLSCREEN_MODE", "false").lower() == "true"

# Logging Configuration
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: Path = LOGS_DIR / "app.log"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# Validate required settings
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in environment variables or .env file")
