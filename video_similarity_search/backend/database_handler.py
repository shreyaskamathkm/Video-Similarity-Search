from typing import List

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)

from video_similarity_search.backend.model import Model
from video_similarity_search.backend.video_handler import VideoHandler

VIDEO_SUFFIXES = ["mp4", "mov"]


class MilvusHandler:
    def __init__(self, drop_old: bool = True):
        self.drop_old = drop_old
        self.search_params = {"nprobe": 128}
        try:
            # Initialize the client directly
            self.client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
            print("Connected to Milvus.")
        except Exception as e:
            print(f"Error connecting to Milvus: {e}")
            raise

        self.collection_name = "video_embeddings"
        self._create_or_get_collection()

        self._create_index()  # Create index if it doesn't exist

        self.client.load_collection(self.collection_name)  # Load collection after initialization
        print(f"Collection '{self.collection_name}' loaded into memory.")

    def _create_or_get_collection(self):
        if self.client.has_collection(self.collection_name) and self.drop_old:
            self.client.drop_collection(self.collection_name)

        if self.client.has_collection(self.collection_name):
            raise RuntimeError(
                f"Collection {self.collection_name} already exists. Set drop_old=True to create a new one instead."
            )
        # Check if the collection exists without 'using'
        if not self.client.has_collection(self.collection_name):
            # Define schema for the collection
            fields = [
                FieldSchema(
                    name="id", dtype=DataType.INT64, is_primary=True, auto_id=True
                ),  # Auto-generated unique ID
                FieldSchema(
                    name="video_name",
                    dtype=DataType.VARCHAR,
                    max_length=255,
                    is_primary=False,
                    auto_id=False,
                ),
                FieldSchema(name="frame_idx", dtype=DataType.INT64, is_primary=False),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=512,
                    is_primary=False,
                ),
            ]
            schema = CollectionSchema(fields=fields, description="Video frame embeddings")

            # Create the collection
            self.client.create_collection(collection_name=self.collection_name, schema=schema)
            print(f"Collection '{self.collection_name}' created.")

    def _create_index(self):
        # Create index for the 'embedding' field
        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="embedding",
            metric_type="COSINE",
            index_type="IVF_FLAT",
            index_name="vector_index",
            params={"nlist": 128},
        )

        self.client.create_index(collection_name=self.collection_name, index_params=index_params)
        print(f"Index created for '{self.collection_name}'.")

    def save_embeddings(self, video_path: str, embeddings: np.ndarray, frame_indices: List[int]):
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Embeddings should be a numpy ndarray")
        if embeddings.shape[1] != 512:
            raise ValueError(
                f"Embeddings should have 512 dimensions, but got {embeddings.shape[1]}"
            )
        if not isinstance(video_path, str):
            video_path = str(video_path)
        if not all(isinstance(idx, int) for idx in frame_indices):
            raise ValueError("frame_indices should be a list of integers")

        # Prepare data for insertion
        data = [
            {
                "video_name": video_path,
                "frame_idx": frame_idx,
                "embedding": embedding.tolist(),
            }
            for frame_idx, embedding in zip(frame_indices, embeddings)
        ]

        try:
            # collection = Collection(name=self.collection_name)
            self.client.insert(collection_name=self.collection_name, data=data)
            print(f"Inserted {len(data)} records into '{self.collection_name}'.")
        except Exception as e:
            print(f"Error saving embeddings: {e}")
            raise

    def search(self, query_embedding, top_k=5):
        if len(query_embedding) != 512:
            raise ValueError(
                f"Query embedding should have 512 dimensions, but got {len(query_embedding)}"
            )

        try:
            # Ensure the collection is loaded
            self.client.load_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' loaded into memory.")

            # Perform the search
            return self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                anns_field="embedding",
                search_params=self.search_params,
                limit=top_k,
                output_fields=["video_name", "frame_idx"],
            )
        except Exception as e:
            print(f"Error during search: {e}")
            raise

    def query(self, expr: str):
        result = self.client.query(collection_name=self.collection_name, filter=expr)
        print(result)

    def delete_video(self, video_name: str):
        try:
            collection = Collection(name=self.collection_name)
            collection.delete(expr=f"video_name == '{video_name}'")
            print(f"Deleted video '{video_name}' from '{self.collection_name}'.")
        except Exception as e:
            print(f"Error deleting video: {e}")
            raise


# Class for managing video database operations
class VideoDatabase:
    def __init__(self, model: Model, video_handler: VideoHandler, milvus_handler: MilvusHandler):
        self.model = model
        self.video_handler = video_handler
        self.milvus_handler = milvus_handler

    def add_video_to_database(self, video_path: str):
        embeddings, frame_indices = self.video_handler.extract_frame_embeddings(video_path)

        self.milvus_handler.save_embeddings(video_path, embeddings, frame_indices)
        print(f"Video {video_path.name} added to the database.")

    def add_videos_from_folder(self, folder_path: str):
        paths = [path for i in VIDEO_SUFFIXES for path in folder_path.glob("*." + i)]
        for video_path in [paths[0]]:
            self.add_video_to_database(video_path.resolve())
