"""
Main entry point for Hand Landmark Tracking RAG application.
"""
from src.app.application import HandDrawingApp
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Main entry point."""
    try:
        app = HandDrawingApp()
        app.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
