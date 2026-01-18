"""
Hand tracking service using MediaPipe.
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional
from config.settings import (
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    MEDIAPIPE_MAX_NUM_HANDS
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class HandTracker:
    """Service for tracking hand landmarks using MediaPipe."""
    
    def __init__(self):
        """Initialize MediaPipe Hands solution."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=MEDIAPIPE_MAX_NUM_HANDS,
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils
        logger.info("HandTracker initialized")
    
    def process_frame(self, frame: np.ndarray) -> Optional[mp.solutions.hands.HandLandmark]:
        """
        Process a frame and detect hand landmarks.
        
        Args:
            frame: BGR frame from camera
            
        Returns:
            Hand landmarks if detected, None otherwise
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                return results.multi_hand_landmarks[0]
            return None
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None
    
    def draw_landmarks(self, frame: np.ndarray, hand_landmarks) -> np.ndarray:
        """
        Draw hand landmarks on frame.
        
        Args:
            frame: Frame to draw on
            hand_landmarks: Hand landmarks from MediaPipe
            
        Returns:
            Frame with landmarks drawn
        """
        self.mp_draw.draw_landmarks(
            frame, 
            hand_landmarks, 
            self.mp_hands.HAND_CONNECTIONS
        )
        return frame
    
    def get_index_finger_position(
        self, 
        hand_landmarks, 
        frame_width: int, 
        frame_height: int
    ) -> Tuple[int, int]:
        """
        Get index finger tip position in pixel coordinates.
        
        Args:
            hand_landmarks: Hand landmarks from MediaPipe
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            (x, y) coordinates of index finger tip
        """
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x = int(index_tip.x * frame_width)
        y = int(index_tip.y * frame_height)
        return x, y
    
    def is_index_finger_raised(self, hand_landmarks) -> bool:
        """
        Check if index finger is raised (tip above PIP joint).
        
        Args:
            hand_landmarks: Hand landmarks from MediaPipe
            
        Returns:
            True if index finger is raised
        """
        lm = hand_landmarks.landmark
        index_tip = lm[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = lm[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        return index_tip.y < index_pip.y
    
    def are_fingers_folded(self, hand_landmarks) -> bool:
        """
        Check if index, middle, ring, and pinky fingers are folded.
        
        Args:
            hand_landmarks: Hand landmarks from MediaPipe
            
        Returns:
            True if all fingers are folded
        """
        lm = hand_landmarks.landmark
        
        # Check each finger: folded if tip is below PIP joint
        fingers_folded = [
            lm[8].y >= lm[6].y,   # Index finger
            lm[12].y >= lm[10].y, # Middle finger
            lm[16].y >= lm[14].y, # Ring finger
            lm[20].y >= lm[18].y  # Pinky finger
        ]
        
        return all(fingers_folded)
    
    def get_gesture_state(self, hand_landmarks) -> dict:
        """
        Get current gesture state (which fingers are raised).
        
        Args:
            hand_landmarks: Hand landmarks from MediaPipe
            
        Returns:
            Dictionary with gesture state information
        """
        lm = hand_landmarks.landmark
        
        return {
            "index_raised": lm[8].y < lm[6].y,
            "middle_raised": lm[12].y < lm[10].y,
            "ring_raised": lm[16].y < lm[14].y,
            "pinky_raised": lm[20].y < lm[18].y,
            "all_folded": self.are_fingers_folded(hand_landmarks)
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.hands:
            self.hands.close()
        logger.info("HandTracker cleaned up")
