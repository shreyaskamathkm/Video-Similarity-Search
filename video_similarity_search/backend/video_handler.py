import logging

import numpy as np

from video_similarity_search.backend.query_result_formatter import QueryResultFormatter
from video_similarity_search.backend.video_processor import VideoProcessor
from video_similarity_search.backend.video_segment_extractor import VideoSegmentExtractor

logger = logging.getLogger(__name__)


class VideoHandler:
    """A class to handle video operations."""

    def __init__(
        self,
        video_processor: VideoProcessor,
        video_segment_extractor: VideoSegmentExtractor,
        query_result_formatter: QueryResultFormatter,
    ) -> None:
        """Initializes the VideoHandler.

        Args:
            video_processor: An instance of VideoProcessor.
            video_segment_extractor: An instance of VideoSegmentExtractor.
            query_result_formatter: An instance of QueryResultFormatter.
        """
        self.video_processor = video_processor
        self.video_segment_extractor = video_segment_extractor
        self.query_result_formatter = query_result_formatter

    def extract_frame_embeddings(
        self, video_path: str, frame_skip: int = 2
    ) -> tuple[np.ndarray, list[int]]:
        """Extracts frame embeddings from a video.

        Args:
            video_path: The path to the video.
            frame_skip: The number of frames to skip between embeddings.

        Returns:
            A tuple of (embeddings, frame_indices).
        """
        return self.video_processor.extract_frame_embeddings(video_path, frame_skip)

    def present_query_results(
        self,
        results: list[tuple[str, int, float]],
        duration: int = 5,
        min_time_distance: float = 2.0,
    ) -> None:
        """Presents the query results.

        Args:
            results: A list of tuples, where each tuple contains the video path,
                frame index, and score.
            duration: The duration of the video segment to extract.
            min_time_distance: The minimum time distance between consecutive segments.
        """
        self.query_result_formatter.present_query_results(results, duration, min_time_distance)
