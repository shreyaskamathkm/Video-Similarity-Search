import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)

from video_similarity_search.backend.model import VLMBaseModel
from video_similarity_search.backend.video_handler import VideoHandler

logger = logging.getLogger(__name__)
VIDEO_SUFFIXES = ["mp4", "mov"]


# Base class for database operations
class Database:
    def __init__(self, collection_name: str, reset_dataset: bool, embedding_size: int) -> None:
        self.reset_dataset = reset_dataset
        self.collection_name = collection_name
        self.embedding_size = embedding_size

    def insert_video_embeddings(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("This should be implemented in the subclass")

    def search(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("This should be implemented in the subclass")

    def query(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("This should be implemented in the subclass")

    def delete_file(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("This should be implemented in the subclass")


class MilvusHandler(Database):
    def __init__(self, collection_name: str, reset_dataset: bool, embedding_size: int) -> None:
        super().__init__(
            collection_name=collection_name,
            reset_dataset=reset_dataset,
            embedding_size=embedding_size,
        )

        self.search_params = {"nprobe": 128}
        try:
            # Initialize the client directly using environment variable for token
            milvus_token = os.environ.get("MILVUS_TOKEN", "root:Milvus")
            self.client = MilvusClient(uri="http://localhost:19530", token=milvus_token)
            logging.info("Connected to Milvus.")
        except Exception as e:
            logging.error(f"Error connecting to Milvus: {e}")
            raise

        self._create_or_get_collection()

        self._create_index()  # Create index if it doesn't exist

        self.client.load_collection(self.collection_name)  # Load collection after initialization
        logging.info(f"Collection '{self.collection_name}' loaded into memory.")

    def _create_or_get_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            if self.reset_dataset:
                self.client.drop_collection(self.collection_name)
            else:
                raise RuntimeError(
                    f"Collection {self.collection_name} already exists.\
                        Set reset_dataset=True to create a new one instead."
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
                    dim=self.embedding_size,
                    is_primary=False,
                ),
            ]
            schema = CollectionSchema(fields=fields, description="Video frame embeddings")

            # Create the collection
            self.client.create_collection(collection_name=self.collection_name, schema=schema)
            logging.info(f"Collection '{self.collection_name}' created.")

    def _create_index(self) -> None:
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
        logging.info(f"Index created for '{self.collection_name}'.")

    def save_embeddings(
        self, video_path: str, embeddings: np.ndarray, frame_indices: list[int]
    ) -> None:
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Embeddings should be a numpy ndarray")
        if embeddings.shape[1] != self.embedding_size:
            raise ValueError(
                f"Embeddings should have {self.embedding_size} dimensions, but got {embeddings.shape[1]}"
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
            logging.info(f"Inserted {len(data)} records into '{self.collection_name}'.")
        except Exception as e:
            logging.error(f"Error saving embeddings: {e}")
            raise

    def search(self, query_embedding: Any, top_k: int = 5) -> list[dict[str, Any]]:
        if len(query_embedding) != self.embedding_size:
            raise ValueError(
                f"Query embedding should have {self.embedding_size} dimensions, but got {len(query_embedding)}"
            )

        try:
            # Ensure the collection is loaded
            self.client.load_collection(self.collection_name)
            logging.info(f"Collection '{self.collection_name}' loaded into memory.")

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
            logging.error(f"Error during search: {e}")
            raise

    def query(self, expr: str) -> list[dict[str, Any]]:
        result = self.client.query(collection_name=self.collection_name, filter=expr)
        logging.info(result)
        return result

    def video_exists(self, path: Path) -> bool:
        try:
            # Check if a video with the given path exists in the collection
            result = self.client.query(
                collection_name=self.collection_name,
                filter=f"video_name == '{str(path)}'",
                output_fields=["id"],
            )
            exists = len(result) > 0
            logging.info(f"Video exists: {exists}")
            return exists
        except Exception as e:
            logging.error(f"Error checking if video exists: {e}")
            raise

    def delete_file(self, video_name: str) -> None:
        try:
            collection = Collection(name=self.collection_name)
            collection.delete(expr=f"video_name == '{video_name}'")
            logging.info(f"Deleted video '{video_name}' from '{self.collection_name}'.")
        except Exception as e:
            logging.error(f"Error deleting video: {e}")
            raise


# Class for managing video database operations
class VideoDatabase:
    def __init__(
        self,
        model: VLMBaseModel,
        video_handler: VideoHandler,
        milvus_handler: MilvusHandler,
        frame_skip: int,
    ):
        self.model = model
        self.frame_skip = frame_skip
        self.video_handler = video_handler
        self.milvus_handler = milvus_handler

    def add_video_to_database(self, video_path: Path) -> None:
        embeddings, frame_indices = self.video_handler.extract_frame_embeddings(
            str(video_path), self.frame_skip
        )

        self.milvus_handler.save_embeddings(str(video_path), embeddings, frame_indices)
        logging.info(f"Video {video_path.name} added to the database.")

    def add_videos_from_folder(self, folder_path: Path) -> None:
        paths = [path for i in VIDEO_SUFFIXES for path in folder_path.glob("*." + i)]
        for video_path in paths:
            self.add_video_to_database(video_path.resolve())
