from PIL import Image

from video_similarity_search.backend.database_handler import MilvusHandler
from video_similarity_search.backend.model import Model

# Class for Search operations


class VideoSearch:
    def __init__(self, model: Model, milvus_handler: MilvusHandler):
        self.model = model
        self.milvus_handler = milvus_handler

    def search_by_text(self, query: str, top_k: int = 5):
        query_embedding = self.model.extract_text_features(query).flatten()

        results = self.milvus_handler.search(query_embedding=query_embedding, top_k=top_k)

        matches = []
        for hit in results[0]:
            matches.append(
                (
                    hit["entity"].get("video_name"),
                    hit["entity"].get("frame_idx"),
                    hit["distance"],
                )
            )
        return matches

    def search_by_image(self, image: Image.Image, top_k: int = 5):
        query_embedding = self.model.extract_image_features(image).flatten()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        results = self.milvus_handler.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["video_name", "frame_idx"],
        )

        matches = []
        for hit in results[0]:
            matches.append(
                (
                    hit.entity.get("video_name"),
                    hit.entity.get("frame_idx"),
                    hit.distance,
                )
            )
        return matches
