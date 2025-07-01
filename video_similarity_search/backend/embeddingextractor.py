from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel

from video_similarity_search.backend.model import VLMBaseModel

logger = logging.getLogger(__name__)


class FrameEmbeddings(BaseModel):
    """A Pydantic model for storing frame embeddings and their indices."""

    embeddings: np.ndarray
    frame_indices: list[int]


class EmbeddingExtractor(ABC):
    """An abstract base class for embedding extractors."""

    def __init__(self, model: VLMBaseModel):
        """Initializes the EmbeddingExtractor.

        Args:
            model: An instance of VLMBaseModel.
        """
        self.model = model

    @abstractmethod
    def extract_embeddings(self, *args: Any):  # type: ignore
        """Extracts embeddings from a source.

        Raises:
            NotImplementedError: This method should be implemented in the subclass.
        """
        raise NotImplementedError("This should be implemented in the subclass")


class VideoEmbeddingExtractor(EmbeddingExtractor):
    """A class for extracting embeddings from videos."""

    def __init__(self, model: VLMBaseModel, frame_skip: int = 5):
        """Initializes the VideoEmbeddingExtractor.

        Args:
            model: An instance of VLMBaseModel.
            frame_skip: The number of frames to skip between embeddings.
        """
        super().__init__(model)
        self.frame_skip = frame_skip

    def extract_embeddings(self, path: str) -> FrameEmbeddings:
        """Extracts embeddings from a video file.

        Args:
            path: The path to the video file.

        Returns:
            A FrameEmbeddings object containing the embeddings and frame indices.
        """
        embeddings, frame_indices = self._extract_frame_embeddings(path)
        return FrameEmbeddings(embeddings=np.vstack(embeddings), frame_indices=frame_indices)

    def _extract_frame_embeddings(self, path: str) -> tuple[list[np.ndarray], list[int]]:
        """Extracts frame embeddings from a video.

        Args:
            path: The path to the video.

        Returns:
            A tuple containing:
                - A list of numpy arrays, where each array is the embedding of a frame.
                - A list of frame indices corresponding to the embeddings.
        """
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