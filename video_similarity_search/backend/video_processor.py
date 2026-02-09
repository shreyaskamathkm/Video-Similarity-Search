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
        self, video_path: str, frame_skip: int = 2, batch_size: int = 32
    ) -> tuple[np.ndarray, list[int]]:
        """Extracts frame embeddings from a video.

        Args:
            video_path: The path to the video.
            frame_skip: The number of frames to skip between embeddings.
            batch_size: The number of frames to process at once.

        Returns:
            A tuple containing:
                - A numpy array of frame embeddings.
                - A list of frame indices corresponding to the embeddings.
        """
        cap = cv2.VideoCapture(video_path)
        embeddings: list[np.ndarray] = []
        frame_indices: list[int] = []
        batch_frames: list[Image.Image] = []
        batch_indices: list[int] = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                batch_frames.append(frame_pil)
                batch_indices.append(frame_idx)

                if len(batch_frames) == batch_size:
                    batch_embeddings = self.model.extract_image_features(batch_frames)
                    embeddings.append(batch_embeddings)
                    frame_indices.extend(batch_indices)
                    batch_frames = []
                    batch_indices = []

            frame_idx += 1

        # Process remaining frames
        if batch_frames:
            batch_embeddings = self.model.extract_image_features(batch_frames)
            embeddings.append(batch_embeddings)
            frame_indices.extend(batch_indices)

        cap.release()
        if not embeddings:
            return np.empty((0, self.model.get_embedding_length())), []
        return np.vstack(embeddings), frame_indices
