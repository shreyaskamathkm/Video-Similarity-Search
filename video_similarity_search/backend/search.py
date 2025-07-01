import logging

import numpy as np
from PIL import Image

from video_similarity_search.backend.database_handler import DatabaseHandler
from video_similarity_search.backend.model import VLMBaseModel

logger = logging.getLogger(__name__)


class VideoSearch:
    """A class for performing video search."""

    def __init__(self, model: VLMBaseModel, database_handler: DatabaseHandler):
        """Initializes the VideoSearch object.

        Args:
            model: The VLMBaseModel to use for extracting features.
            database_handler: The DatabaseHandler to use for searching.
        """
        self.model = model
        self.database_handler = database_handler

    def _search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[str, int, float]]:
        """Performs a search using a query embedding.

        Args:
            query_embedding: The query embedding to search for.
            top_k: The number of results to return.

        Returns:
            A list of tuples, where each tuple contains the video name, frame index,
            and distance.
        """
        results = self.database_handler.search(query_embedding=query_embedding, top_k=top_k)
        matches: list[tuple[str, int, float]] = []
        for hit in results[0]:
            video_name = hit["entity"].get("video_name")
            frame_idx = hit["entity"].get("frame_idx")
            distance = hit["distance"]
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

    def search_by_text(self, query: str, top_k: int = 5) -> list[tuple[str, int, float]]:
        """Performs a search using a text query.

        Args:
            query: The text query to search for.
            top_k: The number of results to return.

        Returns:
            A list of tuples, where each tuple contains the video name, frame index,
            and distance.
        """
        query_embedding = self.model.extract_text_features(query).flatten()
        return self._search(query_embedding=query_embedding, top_k=top_k)

    def search_by_image(self, image: Image.Image, top_k: int = 5) -> list[tuple[str, int, float]]:
        """Performs a search using an image query.

        Args:
            image: The image query to search for.
            top_k: The number of results to return.

        Returns:
            A list of tuples, where each tuple contains the video name, frame index,
            and distance.
        """
        query_embedding = self.model.extract_image_features(image).flatten()
        return self._search(query_embedding=query_embedding, top_k=top_k)
