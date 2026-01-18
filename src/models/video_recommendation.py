"""
Data models for video recommendations.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class VideoRecommendation:
    """Represents a single video recommendation."""
    title: str
    video_id: str
    category: Optional[str] = None
    similarity_score: Optional[float] = None

    @property
    def youtube_url(self) -> str:
        """Generate YouTube URL from video ID."""
        from config.settings import YOUTUBE_BASE_URL
        return f"{YOUTUBE_BASE_URL}{self.video_id}"


@dataclass
class RecommendationResult:
    """Container for multiple video recommendations."""
    recommendations: List[VideoRecommendation]
    query: str
    raw_response: Optional[str] = None

    def __len__(self) -> int:
        return len(self.recommendations)

    def get_recommendation(self, index: int) -> Optional[VideoRecommendation]:
        """Get recommendation by index."""
        if 0 <= index < len(self.recommendations):
            return self.recommendations[index]
        return None
