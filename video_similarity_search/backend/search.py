import logging

import numpy as np
from PIL import Image

from video_similarity_search.backend.database_handler import Database
from video_similarity_search.backend.model import BaseModel

logger = logging.getLogger(__name__)


class Search:
    def __init__(self, model: BaseModel, database: Database):
        self.model = model
        self.database = database

    def _search(self, query_embedding: np.ndarray, top_k: int = 5):
        results = self.database.search(query_embedding=query_embedding, top_k=top_k)

        matches = []
        for hit in results[0]:
            matches.append(
                (
                    hit["entity"].get("path"),
                    hit["entity"].get("frame_idx"),
                    hit["distance"],
                )
            )
        return matches

    def search_by_text(self, query: str, top_k: int = 5):
        query_embedding = self.model.extract_text_features(query).flatten()
        return self._search(query_embedding=query_embedding, top_k=top_k)

    def search_by_image(self, image: Image.Image, top_k: int = 5):
        query_embedding = self.model.extract_image_features(image).flatten()
        return self._search(query_embedding=query_embedding, top_k=top_k)
