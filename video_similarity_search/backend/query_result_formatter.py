import logging
from collections import defaultdict

import cv2

from video_similarity_search.backend.video_segment_extractor import VideoSegmentExtractor


class QueryResultFormatter:
    """A class to format and present query results."""

    def __init__(self, video_segment_extractor: VideoSegmentExtractor) -> None:
        """Initializes the QueryResultFormatter.

        Args:
            video_segment_extractor: An instance of VideoSegmentExtractor.
        """
        self.video_segment_extractor = video_segment_extractor

    def present_query_results(
        self,
        results: list[tuple[str, int, float]],
        duration: int = 5,
        min_time_distance: float = 2.0,  # Minimum distance in seconds
    ) -> None:
        """Presents the query results by extracting and saving video segments.

        Args:
            results: A list of tuples, where each tuple contains the video path,
                frame index, and score.
            duration: The duration of the video segment to extract.
            min_time_distance: The minimum time distance between consecutive segments.
        """
        # Group results by video_path
        grouped = defaultdict(list)
        for video_path, frame_idx, score in results:
            grouped[video_path].append((frame_idx, score))

        filtered_results = []
        for video_path, frames_scores in grouped.items():
            # Get FPS for this video
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            min_frame_distance = int(fps * min_time_distance)
            frames_scores.sort()
            last_frame_idx = -min_frame_distance
            for frame_idx, score in frames_scores:
                if frame_idx - last_frame_idx >= min_frame_distance:
                    filtered_results.append((video_path, frame_idx, score))
                    last_frame_idx = frame_idx

        for video_path, frame_idx, score in filtered_results:
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()

            timestamp = self.video_segment_extractor.frame_to_timestamp(frame_idx, fps)
            segment_path = self.video_segment_extractor.extract_video_segment(
                video_path, frame_idx, fps, duration
            )
            logging.info(f"Match found in video: {video_path}")
            logging.info(f"Timestamp: {timestamp}, Score: {score:.2f}")
            logging.info(f"Segment saved at: {segment_path}")
