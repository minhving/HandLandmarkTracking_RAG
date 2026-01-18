"""
Utility functions for parsing gesture results and video recommendations.
"""
import re
from typing import List, Tuple
from src.models.video_recommendation import VideoRecommendation, RecommendationResult
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_recommendation_response(response_text: str, query: str) -> RecommendationResult:
    """
    Parse the raw AI response into structured RecommendationResult.
    
    Args:
        response_text: Raw response from GPT model
        query: Original query text
        
    Returns:
        RecommendationResult with parsed recommendations
    """
    try:
        # Pattern to match "Title: <title> Video ID: <id>"
        pattern = r'Title:\s*(.+?)\s*Video ID:\s*([a-zA-Z0-9_-]+)'
        matches = re.findall(pattern, response_text, re.IGNORECASE | re.DOTALL)
        
        recommendations = []
        for title, video_id in matches:
            title = title.strip()
            video_id = video_id.strip()
            if title and video_id:
                recommendations.append(VideoRecommendation(
                    title=title,
                    video_id=video_id
                ))
        
        if not recommendations:
            logger.warning(f"No recommendations found in response: {response_text[:200]}")
        
        return RecommendationResult(
            recommendations=recommendations,
            query=query,
            raw_response=response_text
        )
    except Exception as e:
        logger.error(f"Error parsing recommendation response: {e}")
        return RecommendationResult(
            recommendations=[],
            query=query,
            raw_response=response_text
        )
