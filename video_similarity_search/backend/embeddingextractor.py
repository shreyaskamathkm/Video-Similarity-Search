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
    embeddings: np.ndarray
    frame_indices: list[int]


class EmbeddingExtractor(ABC):
    def __init__(self, model: VLMBaseModel):
        self.model = model

    @abstractmethod
    def extract_embeddings(self, *args: Any):  # type: ignore
        raise NotImplementedError("This should be implemented in the subclass")


class VideoEmbeddingExtractor(EmbeddingExtractor):
    def __init__(self, model: VLMBaseModel, frame_skip: int = 5):
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
