
import numpy as np
from PIL import Image

from video_similarity_search.backend.database_handler import MilvusHandler
from video_similarity_search.backend.model import Model

# Class for Search operations


class VideoSearch:
    def __init__(self, model: Model, milvus_handler: MilvusHandler):
        self.model = model
        self.milvus_handler = milvus_handler

    def _search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple[str, int, float]]:
        results = self.milvus_handler.search(query_embedding=query_embedding, top_k=top_k)
        matches: list[tuple[str, int, float]] = []
        for hit in results[0]:
            video_name = hit.entity.get("video_name")
            frame_idx = hit.entity.get("frame_idx")
            distance = hit.distance
            if video_name is None or frame_idx is None or distance is None:
                raise ValueError(
                    "Milvus search result contains None for video_name, frame_idx, or distance."
                )
            matches.append(
                (
                    str(video_name),
                    int(frame_idx),
                    float(distance),
                )
            )
        return matches

    def search_by_text(
        self, query: str, top_k: int = 5
    ) -> list[tuple[str, int, float]]:
        query_embedding = self.model.extract_text_features(query).flatten()
        return self._search(query_embedding=query_embedding, top_k=top_k)

    def search_by_image(
        self, image: Image.Image, top_k: int = 5
    ) -> list[tuple[str, int, float]]:
        query_embedding = self.model.extract_image_features(image).flatten()
        return self._search(query_embedding=query_embedding, top_k=top_k)
