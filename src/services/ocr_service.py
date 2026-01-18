"""
OCR service for recognizing drawn text.
"""
import cv2
import numpy as np
import pytesseract
from typing import List, Tuple
from config.settings import (
    TESSERACT_CONFIG,
    OCR_DILATION_KERNEL_SIZE,
    OCR_DILATION_ITERATIONS,
    DRAWING_THRESHOLD
)
from src.utils.logger import setup_logger
import math

logger = setup_logger(__name__)


class OCRService:
    """Service for OCR text recognition from drawn patterns."""
    
    def __init__(self):
        """Initialize OCR service."""
        logger.info("OCRService initialized")
    
    def create_canvas_from_points(
        self, 
        points: List[Tuple[int, int]], 
        width: int, 
        height: int
    ) -> np.ndarray:
        """
        Create a binary canvas from drawing points.
        
        Args:
            points: List of (x, y) coordinates
            width: Canvas width
            height: Canvas height
            
        Returns:
            Binary canvas image
        """
        canvas = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(1, len(points)):
            pt1 = points[i - 1]
            pt2 = points[i]
            
            # Only draw line if points are close enough
            distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
            if distance < DRAWING_THRESHOLD:
                cv2.line(canvas, pt1, pt2, 255, 3)
        
        return canvas
    
    def preprocess_canvas(self, canvas: np.ndarray) -> np.ndarray:
        """
        Preprocess canvas for better OCR recognition.
        
        Args:
            canvas: Binary canvas image
            
        Returns:
            Preprocessed canvas
        """
        # Dilate to make lines thicker
        kernel = np.ones((OCR_DILATION_KERNEL_SIZE, OCR_DILATION_KERNEL_SIZE), np.uint8)
        canvas = cv2.dilate(canvas, kernel, iterations=OCR_DILATION_ITERATIONS)
        
        # Invert (OCR expects white text on black background)
        canvas_inv = cv2.bitwise_not(canvas)
        
        return canvas_inv
    
    def recognize_text(
        self, 
        points: List[Tuple[int, int]], 
        width: int, 
        height: int
    ) -> str:
        """
        Recognize text from drawing points.
        
        Args:
            points: List of (x, y) coordinates
            width: Canvas width
            height: Canvas height
            
        Returns:
            Recognized text string
        """
        try:
            if len(points) < 2:
                return ""
            
            # Create and preprocess canvas
            canvas = self.create_canvas_from_points(points, width, height)
            processed_canvas = self.preprocess_canvas(canvas)
            
            # Perform OCR
            recognized_text = pytesseract.image_to_string(
                processed_canvas, 
                config=TESSERACT_CONFIG
            ).strip()
            
            logger.info(f"OCR recognized: '{recognized_text}'")
            return recognized_text
            
        except Exception as e:
            logger.error(f"Error in OCR recognition: {e}")
            return ""
