import logging

from video_similarity_search.backend.query_result_formatter import QueryResultFormatter
from video_similarity_search.backend.video_processor import VideoProcessor
from video_similarity_search.backend.video_segment_extractor import VideoSegmentExtractor

logger = logging.getLogger(__name__)


class VideoHandler:
    def __init__(
        self,
        video_processor: VideoProcessor,
        video_segment_extractor: VideoSegmentExtractor,
        query_result_formatter: QueryResultFormatter,
    ) -> None:
        self.video_processor = video_processor
        self.video_segment_extractor = video_segment_extractor
        self.query_result_formatter = query_result_formatter

    def extract_frame_embeddings(self, video_path: str, frame_skip: int = 2):
        return self.video_processor.extract_frame_embeddings(video_path, frame_skip)

    def present_query_results(
        self,
        results: list[tuple[str, int, float]],
        duration: int = 5,
        min_time_distance: float = 2.0,
    ) -> None:
        self.query_result_formatter.present_query_results(
            results, duration, min_time_distance
        )
