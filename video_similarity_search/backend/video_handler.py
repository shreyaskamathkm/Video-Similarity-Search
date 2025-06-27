import logging
import os

import cv2
import numpy as np
from PIL import Image

from video_similarity_search.backend.model import VLMBaseModel

logger = logging.getLogger(__name__)


# Class for Video operations
class VideoHandler:
    def __init__(self, model: VLMBaseModel) -> None:
        self.model = model

    def extract_frame_embeddings(
        self, video_path: str, frame_skip: int = 2
    ) -> tuple[np.ndarray, list[int]]:
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

    def present_query_results(
        self,
        results: list[tuple[str, int, float]],
        duration: int = 5,
        min_frame_distance: int = 30,  # Minimum distance between frames to consider as separate
    ) -> None:
        # Sort results by video_path, then by frame_idx (or by score if you prefer)
        results = sorted(results, key=lambda x: (x[0], x[1]))
        filtered_results = []
        last_frame_idx = {}
        for video_path, frame_idx, score in results:
            if (
                video_path not in last_frame_idx
                or abs(frame_idx - last_frame_idx[video_path]) >= min_frame_distance
            ):
                filtered_results.append((video_path, frame_idx, score))
                last_frame_idx[video_path] = frame_idx
        for video_path, frame_idx, score in filtered_results:
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()

            timestamp = self.frame_to_timestamp(frame_idx, fps)
            segment_path = self.extract_video_segment(video_path, frame_idx, fps, duration)
            logging.info(f"Match found in video: {video_path}")
            logging.info(f"Timestamp: {timestamp}, Score: {score:.2f}")
            logging.info(f"Segment saved at: {segment_path}")

    # Step 5: Convert frame index to timestamp
    def frame_to_timestamp(self, frame_idx: int, fps: int) -> str:
        seconds = frame_idx / fps
        mins, secs = divmod(seconds, 60)
        hrs, mins = divmod(mins, 60)
        return f"{int(hrs):02}:{int(mins):02}:{int(secs):02}"

    # Step 6: Extract video segments
    def extract_video_segment(
        self,
        video_path: str,
        frame_idx: int,
        fps: int,
        duration: int = 5,
        output_dir: str = "./segments",
    ) -> str:
        start_frame = max(0, frame_idx - int(fps * duration // 2))
        end_frame = frame_idx + int(fps * duration // 2)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = min(end_frame, total_frames - 1)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"segment_{os.path.basename(video_path).split('.')[0]}_{frame_idx}.mp4",
        )

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
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
