import cv2
import numpy as np
from PIL import Image

from video_similarity_search.backend.model import VLMBaseModel


class VideoProcessor:
    """A class to process videos and extract frame embeddings."""

    def __init__(self, model: VLMBaseModel) -> None:
        """Initializes the VideoProcessor.

        Args:
            model: An instance of VLMBaseModel.
        """
        self.model = model

    def extract_frame_embeddings(
        self, video_path: str, frame_skip: int = 2
    ) -> tuple[np.ndarray, list[int]]:
        """Extracts frame embeddings from a video.

        Args:
            video_path: The path to the video.
            frame_skip: The number of frames to skip between embeddings.

        Returns:
            A tuple containing:
                - A numpy array of frame embeddings.
                - A list of frame indices corresponding to the embeddings.
        """
        cap = cv2.VideoCapture(video_path)
        embeddings = []
        frame_indices = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_embedding = self.model.extract_image_features(frame_pil)
                embeddings.append(frame_embedding.flatten())
                frame_indices.append(frame_idx)

            frame_idx += 1

        cap.release()
        return np.vstack(embeddings), frame_indices
