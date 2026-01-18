"""
Display service for rendering UI elements on video frames.
"""
import cv2
import numpy as np
from typing import List, Optional
from src.models.video_recommendation import RecommendationResult, VideoRecommendation
from config.settings import (
    WINDOW_NAME,
    FULLSCREEN_MODE,
    DRAWING_LINE_THICKNESS,
    DRAWING_POINT_RADIUS,
    DRAWING_THRESHOLD
)
from src.utils.logger import setup_logger
import math

logger = setup_logger(__name__)


class DisplayService:
    """Service for displaying UI elements and drawing on frames."""
    
    def __init__(self):
        """Initialize display service."""
        self.window_name = WINDOW_NAME
        self._setup_window()
        logger.info("DisplayService initialized")
    
    def _setup_window(self) -> None:
        """Set up the display window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if FULLSCREEN_MODE:
            cv2.setWindowProperty(
                self.window_name, 
                cv2.WND_PROP_FULLSCREEN, 
                cv2.WINDOW_FULLSCREEN
            )
    
    def draw_trail(
        self, 
        frame: np.ndarray, 
        points: List[tuple], 
        color: tuple = (255, 0, 0)
    ) -> np.ndarray:
        """
        Draw drawing trail on frame.
        
        Args:
            frame: Frame to draw on
            points: List of (x, y) coordinates
            color: RGB color tuple
            
        Returns:
            Frame with trail drawn
        """
        for i in range(1, len(points)):
            pt1 = points[i - 1]
            pt2 = points[i]
            distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
            if distance < DRAWING_THRESHOLD:
                cv2.line(frame, pt1, pt2, color, DRAWING_LINE_THICKNESS)
        return frame
    
    def draw_point(self, frame: np.ndarray, point: tuple, color: tuple = (0, 0, 255)) -> np.ndarray:
        """
        Draw a single point on frame.
        
        Args:
            frame: Frame to draw on
            point: (x, y) coordinates
            color: RGB color tuple
            
        Returns:
            Frame with point drawn
        """
        cv2.circle(frame, point, DRAWING_POINT_RADIUS, color, cv2.FILLED)
        return frame
    
    def display_text(
        self, 
        frame: np.ndarray, 
        text: str, 
        position: tuple = (10, 100),
        font_scale: float = 2.0,
        color: tuple = (0, 255, 0),
        thickness: int = 3
    ) -> np.ndarray:
        """
        Display text on frame.
        
        Args:
            frame: Frame to draw on
            text: Text to display
            position: (x, y) position
            font_scale: Font scale
            color: RGB color tuple
            thickness: Text thickness
            
        Returns:
            Frame with text drawn
        """
        cv2.putText(
            frame, 
            text, 
            position, 
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, 
            color, 
            thickness, 
            cv2.LINE_AA
        )
        return frame
    
    def display_recommendations(
        self, 
        frame: np.ndarray, 
        recommendations: RecommendationResult,
        selected_index: Optional[int] = None
    ) -> np.ndarray:
        """
        Display video recommendations on frame.
        
        Args:
            frame: Frame to draw on
            recommendations: RecommendationResult object
            selected_index: Index of selected recommendation (0-2)
            
        Returns:
            Frame with recommendations drawn
        """
        if not recommendations or len(recommendations) == 0:
            return frame
        
        y_positions = [100, 200, 300]
        
        for i, rec in enumerate(recommendations.recommendations[:3]):
            if i >= len(y_positions):
                break
            
            # Color: red if selected, green otherwise
            color = (255, 0, 0) if selected_index == i else (0, 255, 0)
            
            # Truncate title if too long
            title = rec.title
            if len(title) > 50:
                title = title[:47] + "..."
            
            self.display_text(
                frame,
                title,
                position=(10, y_positions[i]),
                font_scale=0.5,
                color=color,
                thickness=2
            )
        
        return frame
    
    def display_processing(self, frame: np.ndarray) -> np.ndarray:
        """
        Display processing indicator.
        
        Args:
            frame: Frame to draw on
            
        Returns:
            Frame with processing indicator
        """
        return self.display_text(
            frame,
            "Processing...",
            position=(10, 100),
            font_scale=2.0,
            color=(0, 255, 255),
            thickness=3
        )
    
    def show_frame(self, frame: np.ndarray) -> None:
        """
        Display frame in window.
        
        Args:
            frame: Frame to display
        """
        cv2.imshow(self.window_name, frame)
    
    def cleanup(self) -> None:
        """Clean up display resources."""
        cv2.destroyAllWindows()
        logger.info("DisplayService cleaned up")
