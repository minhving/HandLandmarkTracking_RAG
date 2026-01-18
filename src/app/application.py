"""
Main application class for Hand Landmark Tracking RAG.
"""
import cv2
import numpy as np
import threading
import time
import webbrowser
from typing import List, Tuple, Optional
from enum import Enum

from config.settings import (
    CAMERA_INDEX,
    FRAME_MIRROR,
    FRAME_DELAY,
    MIN_DRAWING_POINTS,
    RECOGNITION_STABILITY_FRAMES,
    GESTURE_SELECTION_FRAMES
)
from src.services.hand_tracker import HandTracker
from src.services.ocr_service import OCRService
from src.services.rag_service import RAGService
from src.services.display_service import DisplayService
from src.models.video_recommendation import RecommendationResult
from src.utils.gesture_parser import parse_recommendation_response
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AppState(Enum):
    """Application state enumeration."""
    DRAWING = "drawing"
    RECOGNIZING = "recognizing"
    PROCESSING = "processing"
    DISPLAYING_RESULTS = "displaying_results"


class HandDrawingApp:
    """Main application class for hand drawing recognition and video recommendations."""
    
    def __init__(self):
        """Initialize the application."""
        # Services
        self.hand_tracker = HandTracker()
        self.ocr_service = OCRService()
        self.rag_service = RAGService()
        self.display_service = DisplayService()
        
        # Camera
        self.cap: Optional[cv2.VideoCapture] = None
        
        # State
        self.state = AppState.DRAWING
        self.drawing_points: List[Tuple[int, int]] = []
        self.recognized_text = ""
        self.old_recognized_text = ""
        self.recommendation_result: Optional[RecommendationResult] = None
        self.processing = False
        
        # Counters
        self.recognition_stability_count = 0
        self.gesture_selection_counts = [0, 0, 0]  # For 3 recommendations
        
        logger.info("HandDrawingApp initialized")
    
    def initialize(self) -> bool:
        """
        Initialize all services and camera.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize RAG service
            logger.info("Initializing RAG service...")
            self.rag_service.initialize()
            
            # Initialize camera
            logger.info(f"Initializing camera {CAMERA_INDEX}...")
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {CAMERA_INDEX}")
                return False
            
            logger.info("Application initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing application: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Processed frame
        """
        h, w, _ = frame.shape
        
        # Mirror frame if configured
        if FRAME_MIRROR:
            frame = cv2.flip(frame, 1)
        
        # Process hand landmarks
        hand_landmarks = self.hand_tracker.process_frame(frame)
        
        if hand_landmarks:
            # Draw hand landmarks
            frame = self.hand_tracker.draw_landmarks(frame, hand_landmarks)
            
            # Get gesture state
            gesture_state = self.hand_tracker.get_gesture_state(hand_landmarks)
            
            # Handle different states
            if self.state == AppState.DRAWING:
                self._handle_drawing(frame, hand_landmarks, w, h, gesture_state)
            elif self.state == AppState.RECOGNIZING:
                self._handle_recognition(frame, hand_landmarks, w, h, gesture_state)
            elif self.state == AppState.PROCESSING:
                frame = self.display_service.display_processing(frame)
            elif self.state == AppState.DISPLAYING_RESULTS:
                self._handle_selection(frame, hand_landmarks, gesture_state)
        
        # Draw trail
        if self.drawing_points:
            frame = self.display_service.draw_trail(frame, self.drawing_points)
        
        # Display current recognized text or recommendations
        if self.state == AppState.DISPLAYING_RESULTS and self.recommendation_result:
            selected_index = self._get_selected_index()
            frame = self.display_service.display_recommendations(
                frame, 
                self.recommendation_result,
                selected_index
            )
        elif self.state != AppState.PROCESSING:
            if self.recognized_text:
                frame = self.display_service.display_text(
                    frame, 
                    self.recognized_text,
                    position=(10, 100)
                )
        
        return frame
    
    def _handle_drawing(
        self, 
        frame: np.ndarray, 
        hand_landmarks, 
        width: int, 
        height: int,
        gesture_state: dict
    ) -> None:
        """Handle drawing state."""
        if gesture_state["index_raised"]:
            # Get index finger position
            x, y = self.hand_tracker.get_index_finger_position(hand_landmarks, width, height)
            self.drawing_points.append((x, y))
            frame = self.display_service.draw_point(frame, (x, y))
            self.recognized_text = ""  # Reset recognition if drawing resumes
    
    def _handle_recognition(
        self, 
        frame: np.ndarray, 
        hand_landmarks, 
        width: int, 
        height: int,
        gesture_state: dict
    ) -> None:
        """Handle recognition state."""
        if gesture_state["all_folded"] and len(self.drawing_points) > MIN_DRAWING_POINTS:
            # Perform OCR
            self.recognized_text = self.ocr_service.recognize_text(
                self.drawing_points, 
                width, 
                height
            )
            self.state = AppState.DRAWING  # Return to drawing after recognition
        
        # Check if text is stable and trigger RAG
        if gesture_state["all_folded"] and self.recognized_text:
            if self.recognized_text == self.old_recognized_text:
                self.recognition_stability_count += 1
                if (self.recognition_stability_count > RECOGNITION_STABILITY_FRAMES 
                    and not self.processing):
                    self._trigger_rag_processing()
                    self.recognition_stability_count = 0
            else:
                self.old_recognized_text = self.recognized_text
                self.recognition_stability_count = 0
    
    def _handle_selection(
        self, 
        frame: np.ndarray, 
        hand_landmarks,
        gesture_state: dict
    ) -> None:
        """Handle video selection via gestures."""
        if not self.recommendation_result:
            return
        
        # Determine which gesture is active
        if (gesture_state["index_raised"] and 
            gesture_state["middle_raised"] and 
            gesture_state["ring_raised"]):
            # Three fingers: select third video (index 2)
            self.gesture_selection_counts[2] += 1
            self.gesture_selection_counts[0] = 0
            self.gesture_selection_counts[1] = 0
            
            if self.gesture_selection_counts[2] >= GESTURE_SELECTION_FRAMES:
                self._open_video(2)
                self.gesture_selection_counts[2] = 0
                
        elif gesture_state["index_raised"] and gesture_state["middle_raised"]:
            # Two fingers: select second video (index 1)
            self.gesture_selection_counts[1] += 1
            self.gesture_selection_counts[0] = 0
            self.gesture_selection_counts[2] = 0
            
            if self.gesture_selection_counts[1] >= GESTURE_SELECTION_FRAMES:
                self._open_video(1)
                self.gesture_selection_counts[1] = 0
                
        elif gesture_state["index_raised"]:
            # One finger: select first video (index 0)
            self.gesture_selection_counts[0] += 1
            self.gesture_selection_counts[1] = 0
            self.gesture_selection_counts[2] = 0
            
            if self.gesture_selection_counts[0] >= GESTURE_SELECTION_FRAMES:
                self._open_video(0)
                self.gesture_selection_counts[0] = 0
        else:
            # Reset all counters if no gesture
            self.gesture_selection_counts = [0, 0, 0]
    
    def _get_selected_index(self) -> Optional[int]:
        """Get currently selected recommendation index based on gesture counts."""
        max_count = max(self.gesture_selection_counts)
        if max_count > 0:
            return self.gesture_selection_counts.index(max_count)
        return None
    
    def _trigger_rag_processing(self) -> None:
        """Trigger RAG processing in a separate thread."""
        if self.processing:
            return
        
        self.processing = True
        self.state = AppState.PROCESSING
        
        def process():
            try:
                response = self.rag_service.get_recommendations(self.recognized_text)
                self.recommendation_result = parse_recommendation_response(
                    response, 
                    self.recognized_text
                )
                self.state = AppState.DISPLAYING_RESULTS
                self.drawing_points = []  # Clear drawing after processing
                logger.info(f"Generated {len(self.recommendation_result)} recommendations")
            except Exception as e:
                logger.error(f"Error in RAG processing: {e}")
                self.state = AppState.DRAWING
            finally:
                self.processing = False
        
        threading.Thread(target=process, daemon=True).start()
    
    def _open_video(self, index: int) -> None:
        """Open selected video in browser."""
        if not self.recommendation_result:
            return
        
        rec = self.recommendation_result.get_recommendation(index)
        if rec:
            url = rec.youtube_url
            logger.info(f"Opening video: {url}")
            webbrowser.open(url)
    
    def handle_keypress(self, key: int) -> bool:
        """
        Handle keyboard input.
        
        Args:
            key: Key code from cv2.waitKey
            
        Returns:
            True if should continue, False if should quit
        """
        if key == ord('c'):  # Clear
            self.drawing_points = []
            self.recognized_text = ""
            self.old_recognized_text = ""
            self.recommendation_result = None
            self.recognition_stability_count = 0
            self.gesture_selection_counts = [0, 0, 0]
            self.state = AppState.DRAWING
            logger.info("Cleared drawing and reset state")
            return True
        elif key == ord('q'):  # Quit
            return False
        return True
    
    def run(self) -> None:
        """Run the main application loop."""
        if not self.initialize():
            logger.error("Failed to initialize application")
            return
        
        logger.info("Starting application loop...")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break
                
                # Process frame
                frame = self.process_frame(frame)
                
                # Display frame
                self.display_service.show_frame(frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keypress(key):
                    break
                
                time.sleep(FRAME_DELAY)
                
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Error in application loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up all resources."""
        logger.info("Cleaning up application...")
        
        if self.cap:
            self.cap.release()
        
        self.hand_tracker.cleanup()
        self.rag_service.cleanup()
        self.display_service.cleanup()
        
        logger.info("Application cleaned up")
