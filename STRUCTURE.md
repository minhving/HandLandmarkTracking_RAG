# Project Structure Documentation

## Overview

This document provides a detailed overview of the project structure and architecture.

## Directory Structure

```
HandLandmarkTracking_RAG/
│
├── src/                          # Source code directory
│   ├── __init__.py
│   ├── app/                      # Application layer
│   │   ├── __init__.py
│   │   └── application.py        # Main HandDrawingApp class
│   ├── services/                 # Business logic services
│   │   ├── __init__.py
│   │   ├── hand_tracker.py       # Hand tracking using MediaPipe
│   │   ├── ocr_service.py        # OCR text recognition
│   │   ├── rag_service.py        # RAG recommendation engine
│   │   └── display_service.py    # UI rendering and display
│   ├── models/                   # Data models
│   │   ├── __init__.py
│   │   └── video_recommendation.py  # Video recommendation models
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── logger.py             # Logging configuration
│       └── gesture_parser.py     # Parse AI responses
│
├── config/                       # Configuration
│   ├── __init__.py
│   └── settings.py              # Centralized settings
│
├── data/                         # Data files (gitignored)
│   ├── .gitkeep
│   ├── youtube_titles.csv        # Input CSV (user-provided)
│   ├── output.json              # Generated embeddings (auto)
│   └── chroma_db/               # ChromaDB storage (auto)
│
├── logs/                         # Application logs (gitignored)
│   └── app.log                  # Main log file
│
├── tests/                        # Unit tests (to be implemented)
│
├── main.py                       # Application entry point
├── requirements.txt              # Python dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── README.md                    # Main documentation
├── MIGRATION.md                 # Migration guide
└── STRUCTURE.md                 # This file
```

## Component Descriptions

### Application Layer (`src/app/`)

**application.py** - `HandDrawingApp`
- Main application orchestrator
- Manages application state and lifecycle
- Coordinates between services
- Handles user input and frame processing

### Services Layer (`src/services/`)

**hand_tracker.py** - `HandTracker`
- MediaPipe hand detection and tracking
- Gesture recognition (finger states)
- Hand landmark processing

**ocr_service.py** - `OCRService`
- Text recognition from drawing points
- Canvas creation and preprocessing
- Tesseract OCR integration

**rag_service.py** - `RAGService`
- OpenAI API integration
- ChromaDB vector database management
- Embedding generation and storage
- Similarity search and recommendations

**display_service.py** - `DisplayService`
- Frame rendering and display
- UI element drawing
- Video recommendation display

### Models Layer (`src/models/`)

**video_recommendation.py**
- `VideoRecommendation`: Single recommendation model
- `RecommendationResult`: Container for multiple recommendations

### Utilities (`src/utils/`)

**logger.py**
- Centralized logging configuration
- File and console handlers
- Log level management

**gesture_parser.py**
- Parse AI response text into structured recommendations
- Extract titles and video IDs

### Configuration (`config/`)

**settings.py**
- Centralized configuration management
- Environment variable loading
- Default value definitions
- Path management

## Data Flow

1. **Camera** → `HandDrawingApp.process_frame()`
2. **Frame Processing** → `HandTracker.process_frame()`
3. **Drawing Points** → Collected in application state
4. **OCR Trigger** → `OCRService.recognize_text()`
5. **Text Recognition** → Stored in application state
6. **RAG Query** → `RAGService.get_recommendations()`
7. **Recommendations** → Parsed by `gesture_parser`
8. **Display** → `DisplayService.display_recommendations()`
9. **Selection** → Gesture detection → Browser opens video

## State Management

The application uses an `AppState` enum:
- `DRAWING`: User is drawing
- `RECOGNIZING`: OCR recognition in progress
- `PROCESSING`: RAG query in progress
- `DISPLAYING_RESULTS`: Showing recommendations

## Configuration Management

All configuration is managed through:
1. Environment variables (`.env` file)
2. `config/settings.py` (defaults and validation)
3. Runtime configuration via environment

## Error Handling Strategy

- **Service Level**: Try-except blocks with logging
- **Application Level**: Graceful degradation
- **User Level**: Clear error messages and state reset

## Logging Strategy

- **Console**: INFO level and above
- **File**: DEBUG level and above (`logs/app.log`)
- **Format**: Timestamp, logger name, level, message

## Extension Points

To add new features:

1. **New Service**: Add to `src/services/` following existing patterns
2. **New Model**: Add to `src/models/` as dataclass
3. **New Utility**: Add to `src/utils/` with documentation
4. **Configuration**: Add to `config/settings.py` and `.env.example`

## Testing Strategy (Future)

- Unit tests for each service
- Integration tests for application flow
- Mock external dependencies (OpenAI, ChromaDB, Camera)
