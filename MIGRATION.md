# Migration Guide: Old Structure to New Structure

This document explains the migration from the old monolithic structure to the new business-ready architecture.

## Old Structure

The old structure had:
- `main.py` - Monolithic script with all logic
- `RAG.py` - RAG class mixed with initialization logic
- `imports.py` - Centralized imports file

## New Structure

The new structure separates concerns:

### File Mapping

| Old File | New Location | Notes |
|----------|-------------|-------|
| `main.py` (old) | `src/app/application.py` | Refactored into `HandDrawingApp` class |
| `RAG.py` | `src/services/rag_service.py` | Refactored into `RAGService` class |
| `imports.py` | Removed | Imports now in respective modules |
| Configuration (hardcoded) | `config/settings.py` | Centralized configuration |

### Key Changes

1. **Configuration Management**
   - Old: Hardcoded values in code
   - New: Environment variables via `config/settings.py`

2. **Service Separation**
   - Old: All logic in `main.py`
   - New: Separate services (`HandTracker`, `OCRService`, `RAGService`, `DisplayService`)

3. **State Management**
   - Old: Global variables
   - New: `HandDrawingApp` class with proper state management

4. **Error Handling**
   - Old: Minimal error handling
   - New: Comprehensive try-except blocks with logging

5. **Logging**
   - Old: Print statements
   - New: Structured logging system

6. **Data Models**
   - Old: No data models
   - New: `VideoRecommendation` and `RecommendationResult` dataclasses

## Migration Steps

1. **Backup old files** (if needed):
   ```bash
   mkdir old_code_backup
   cp main.py old_code_backup/
   cp RAG.py old_code_backup/
   cp imports.py old_code_backup/
   ```

2. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Move data files**:
   ```bash
   mv youtube_titles.csv data/
   mv output.json data/  # if exists
   ```

4. **Run new application**:
   ```bash
   python main.py
   ```

## Breaking Changes

1. **Import paths**: All imports now use absolute paths from project root
2. **Configuration**: Must use `.env` file or environment variables
3. **Data location**: Data files should be in `data/` directory
4. **Initialization**: RAG service initialization is automatic on first run

## Backward Compatibility

The old files (`RAG.py`, old `main.py`, `imports.py`) are kept for reference but are no longer used. You can safely remove them after verifying the new structure works.

## Questions?

If you encounter issues during migration:
1. Check `logs/app.log` for detailed error messages
2. Verify `.env` file is properly configured
3. Ensure all dependencies are installed: `pip install -r requirements.txt`
4. Check that data files are in the correct location
