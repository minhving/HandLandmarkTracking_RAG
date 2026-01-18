# HandLandmarkTracking_RAG

A production-ready, real-time hand gesture-based drawing and video recommendation system that combines computer vision, OCR, and RAG (Retrieval-Augmented Generation) to enable users to draw text with their hand and receive YouTube video recommendations based on their drawings.

## Features

- **Hand Tracking**: Real-time hand landmark detection using MediaPipe
- **Air Drawing**: Draw text in the air using your index finger
- **OCR Recognition**: Automatic text recognition from drawn patterns using Tesseract
- **RAG-Powered Recommendations**: Uses OpenAI embeddings and ChromaDB to find relevant YouTube videos
- **Gesture-Based Selection**: Select video recommendations using hand gestures
- **Automatic Video Opening**: Opens selected YouTube videos in your default browser
- **Production-Ready Architecture**: Modular design with proper separation of concerns, logging, error handling, and configuration management

## Architecture

The project follows a clean, business-ready architecture:

```
HandLandmarkTracking_RAG/
├── src/                          # Source code
│   ├── app/                      # Application layer
│   │   └── application.py        # Main application class
│   ├── services/                 # Business logic services
│   │   ├── hand_tracker.py       # Hand tracking service
│   │   ├── ocr_service.py        # OCR recognition service
│   │   ├── rag_service.py        # RAG recommendation service
│   │   └── display_service.py    # UI/Display service
│   ├── models/                   # Data models
│   │   └── video_recommendation.py
│   └── utils/                    # Utility functions
│       ├── logger.py             # Logging utility
│       └── gesture_parser.py     # Gesture parsing utilities
├── config/                       # Configuration
│   └── settings.py              # Centralized configuration
├── data/                         # Data files (gitignored)
│   ├── youtube_titles.csv        # YouTube video metadata
│   ├── output.json              # Generated embeddings
│   └── chroma_db/               # ChromaDB storage
├── logs/                        # Application logs
├── tests/                       # Unit tests
├── main.py                      # Application entry point
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## How It Works

1. **Drawing Phase**: Point your index finger upward and draw text in the air. The system tracks your finger movement and displays a blue trail.
2. **Recognition Phase**: Close all fingers (fold them) to trigger OCR recognition of your drawing.
3. **Recommendation Phase**: The recognized text is sent to a RAG system that:
   - Generates embeddings using OpenAI's `text-embedding-3-large` model
   - Queries a ChromaDB vector database containing YouTube video metadata
   - Uses GPT-4o to select the top 3 most relevant videos
4. **Selection Phase**: Use hand gestures to select a video:
   - **Index finger up**: Select first video (highlighted in red)
   - **Index + Middle fingers up**: Select second video (highlighted in red)
   - **Index + Middle + Ring fingers up**: Select third video (highlighted in red)
   - Hold the gesture for ~2 seconds to open the selected video

## Requirements

### Python Dependencies

- Python 3.8+
- See `requirements.txt` for complete list

### System Dependencies

- **Tesseract OCR**: Must be installed on your system
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

- **Webcam**: A working webcam for hand tracking

## Installation

1. **Clone the repository**:
   ```bash
   cd HandLandmarkTracking_RAG
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

5. **Prepare Data Files**:
   - Place your `youtube_titles.csv` file in the `data/` directory
   - The CSV should have columns: `title`, `category_1`, `vid_id`
   - The system will automatically generate embeddings on first run

## Usage

### Running the Application

```bash
python main.py
```

### Controls

- **Draw**: Point index finger up and move it to draw
- **Recognize**: Close all fingers to trigger OCR
- **Select Video**: Use finger gestures (see "How It Works" section)
- **Clear**: Press `c` key to clear the drawing and reset
- **Quit**: Press `q` key to exit

## Configuration

All configuration is managed through environment variables (see `.env.example`) and can be set in:

1. **Environment variables**: Set directly in your shell
2. **`.env` file**: Create from `.env.example` template
3. **`config/settings.py`**: Default values and configuration logic

### Key Configuration Options

- **OpenAI**: API key, model selection, embedding model
- **MediaPipe**: Detection/tracking confidence, max hands
- **Drawing**: Threshold, minimum points, line thickness
- **OCR**: Tesseract config, dilation parameters
- **RAG**: Collection name, number of results, train/test split
- **Camera**: Camera index, frame mirroring, delay
- **UI**: Window name, fullscreen mode
- **Logging**: Log level, log file location

## Project Structure Details

### Services Layer

- **HandTracker**: MediaPipe-based hand tracking and gesture recognition
- **OCRService**: Text recognition from drawn patterns
- **RAGService**: Vector search and GPT-based recommendations
- **DisplayService**: UI rendering and frame display

### Models Layer

- **VideoRecommendation**: Data model for individual recommendations
- **RecommendationResult**: Container for multiple recommendations

### Application Layer

- **HandDrawingApp**: Main application class managing state and orchestration
- **AppState**: Enumeration of application states

## Development

### Adding New Features

1. **New Service**: Add to `src/services/` with proper error handling and logging
2. **New Model**: Add to `src/models/` as a dataclass
3. **New Utility**: Add to `src/utils/` with appropriate documentation
4. **Configuration**: Add to `config/settings.py` and `.env.example`

### Logging

The application uses structured logging:
- Console output: INFO level and above
- File logging: DEBUG level and above (stored in `logs/app.log`)
- Log format includes timestamp, logger name, level, and message

### Error Handling

All services include comprehensive error handling:
- Try-except blocks around critical operations
- Logging of errors with context
- Graceful degradation where possible

## Testing

```bash
# Run tests (when implemented)
pytest tests/
```

## Troubleshooting

- **Tesseract not found**: Ensure Tesseract is installed and in your system PATH
- **No hand detected**: Ensure good lighting and keep your hand visible to the camera
- **Poor OCR recognition**: Try drawing larger, clearer letters with slower movements
- **OpenAI API errors**: Check your API key in the `.env` file and ensure you have sufficient credits
- **Import errors**: Ensure you're running from the project root directory
- **Configuration errors**: Check `config/settings.py` for required environment variables

## Performance Considerations

- **First Run**: Initial embedding generation may take time depending on dataset size
- **API Calls**: RAG queries require OpenAI API calls (costs apply)
- **Camera Performance**: Frame rate depends on hardware capabilities
- **Memory**: ChromaDB stores embeddings in memory/disk

## Security Notes

- Never commit `.env` file with real API keys
- Use environment variables for sensitive configuration
- Review `.gitignore` to ensure sensitive files are excluded

## License

This project is provided as-is for educational and research purposes.

## Contributing

1. Follow the existing code structure and patterns
2. Add proper error handling and logging
3. Update configuration documentation
4. Add tests for new features
5. Update README as needed
