import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np
from PIL import Image

from video_similarity_search.backend.model import BaseModel
from video_similarity_search.backend.schema import FrameEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingExtractor(ABC):
    def __init__(self, model: BaseModel):
        self.model = model

    @abstractmethod
    def extract_embeddings(self, *args: Any):  # type: ignore
        raise NotImplementedError("This should be implemented in the subclass")


# Class for Video operations
class VideoEmbeddingExtractor(EmbeddingExtractor):
    def __init__(self, model: BaseModel, frame_skip: int = 2):
        super().__init__(model)
        self.frame_skip = frame_skip

    def extract_embeddings(self, path: str) -> FrameEmbeddings:
        embeddings, frame_indices = self._extract_frame_embeddings(path)
        return FrameEmbeddings(embeddings=np.vstack(embeddings), frame_indices=frame_indices)

    def _extract_frame_embeddings(self, path: str) -> tuple[list[np.ndarray], list[int]]:
        cap = cv2.VideoCapture(path)
        embeddings: list[np.ndarray] = []
        frame_indices: list[int] = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.frame_skip == 0:
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                logger.debug(f"Extracting Features for Fame Index {frame_idx}")
                frame_embedding = self.model.extract_image_features(frame_pil)
                embeddings.append(frame_embedding.flatten())
                frame_indices.append(frame_idx)

            frame_idx += 1

        cap.release()
        return embeddings, frame_indices

    def present_query_results(
        self,
        results: list[tuple[str, int, float]],
        duration: int = 5,
        time_threshold: int = 10,
    ):
        grouped_results = self._group_close_frames(results, time_threshold)

        for video_path, frames_group in grouped_results.items():
            fps = self._get_video_fps(video_path)
            first_frame_idx = min(frames_group)
            last_frame_idx = max(frames_group)

            start_frame = first_frame_idx
            end_frame = last_frame_idx + int(fps * duration)
            segment_path = self._extract_video_segment(video_path, start_frame, end_frame, fps)

            logger.info(f"Match found in video: {video_path}")
            logger.info(f"Frames: {frames_group}")
            logger.info(f"Segment saved at: {segment_path}")

    def _group_close_frames(
        self, results: list[tuple[str, int, float]], time_threshold: int
    ) -> dict[str, list[int]]:
        """Groups frame indices by proximity within the same video."""
        grouped: dict[str, list[int]] = {}
        for video_path, frame_idx, _ in results:
            if video_path not in grouped:
                grouped[video_path] = []
            grouped[video_path].append(frame_idx)

        for video_path, frames in grouped.items():
            frames.sort()
            grouped[video_path] = self._group_frames_by_time(frames, time_threshold)

        return grouped

    def _group_frames_by_time(self, frames: list[int], time_threshold: int) -> list[int]:
        """Groups frames if they are within the specified time threshold."""
        grouped = []
        temp_group = [frames[0]]

        for i in range(1, len(frames)):
            if frames[i] - frames[i - 1] <= time_threshold:
                temp_group.append(frames[i])
            else:
                grouped.extend(temp_group)
                temp_group = [frames[i]]

        grouped.extend(temp_group)
        return grouped

    def _get_video_fps(self, video_path: str) -> int:
        """Gets the FPS of the video."""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return fps

    def _extract_video_segment(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        fps: int,
        output_dir: str = "./segments",
    ) -> str:
        """Extracts a video segment from start_frame to end_frame."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"segment_{os.path.basename(video_path).split('.')[0]}_{start_frame}_{end_frame}.mp4",
        )

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = min(end_frame, total_frames - 1)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type:ignore
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()
        return output_path
