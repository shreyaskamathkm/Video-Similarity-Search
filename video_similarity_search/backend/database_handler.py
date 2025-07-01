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
    """A base class for database operations."""

    def __init__(self, collection_name: str, reset_dataset: bool, embedding_size: int) -> None:
        """Initializes the Database object.
        Args:
            collection_name: The name of the collection.
            reset_dataset: Whether to reset the dataset.
            embedding_size: The size of the embeddings.
        """
        self.reset_dataset = reset_dataset
        self.collection_name = collection_name
        self.embedding_size = embedding_size

    def insert_video_embeddings(self, *args: Any, **kwargs: Any) -> None:
        """Inserts video embeddings into the database.
        Raises:
            NotImplementedError: This method should be implemented in the subclass.
        """
        raise NotImplementedError("This should be implemented in the subclass")

    def search(self, *args: Any, **kwargs: Any) -> Any:
        """Searches the database.
        Raises:
            NotImplementedError: This method should be implemented in the subclass.
        """
        raise NotImplementedError("This should be implemented in the subclass")

    def query(self, *args: Any, **kwargs: Any) -> Any:
        """Queries the database.
        Raises:
            NotImplementedError: This method should be implemented in the subclass.
        """
        raise NotImplementedError("This should be implemented in the subclass")

    def delete_file(self, *args: Any, **kwargs: Any) -> None:
        """Deletes a file from the database.
        Raises:
            NotImplementedError: This method should be implemented in the subclass.
        """
        raise NotImplementedError("This should be implemented in the subclass")


class MilvusHandler(Database):
    """A class for handling Milvus database operations."""

    def __init__(self, collection_name: str, reset_dataset: bool, embedding_size: int) -> None:
        """Initializes the MilvusHandler object.
        Args:
            collection_name: The name of the collection.
            reset_dataset: Whether to reset the dataset.
            embedding_size: The size of the embeddings.
        """
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
            logger.info("Connected to Milvus.")
        except Exception as e:
            logger.error(f"Error connecting to Milvus: {e}")
            raise

        self._create_or_get_collection()

        self._create_index()
        self.client.load_collection(self.collection_name)
        logger.info(f"Collection '{self.collection_name}' loaded into memory.")

    def _create_or_get_collection(self) -> None:
        """Creates a new collection or gets an existing one."""
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
                    name="path",
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
            schema = CollectionSchema(fields=fields, description="Embedding Space")

            # Create the collection
            self.client.create_collection(collection_name=self.collection_name, schema=schema)
            logger.info(f"Collection '{self.collection_name}' created.")
            logger.info(f"Collection '{self.collection_name}' created.")

    def _create_index(self) -> None:
        """Creates an index for the 'embedding' field."""
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
        logger.info(f"Index created for '{self.collection_name}'.")
        logger.info(f"Index created for '{self.collection_name}'.")

    def save_embeddings(
        self, video_path: str, embeddings: np.ndarray, frame_indices: list[int]
    ) -> None:
        """Saves embeddings to the database.
        Args:
            video_path: The path to the video.
            embeddings: The embeddings to save.
            frame_indices: The frame indices corresponding to the embeddings.
        Raises:
            ValueError: If the embeddings are not a numpy ndarray, have the wrong
                dimensions, or if frame_indices is not a list of integers.
        """
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Embeddings should be a numpy ndarray")
        if embeddings.shape[1] != self.embedding_size:
            raise ValueError(
                f"Embeddings should have {self.embedding_size} dimensions, \
                but got {embeddings.shape[1]}"
            )
        if not isinstance(path, str):
            path = str(path)
        if not all(isinstance(idx, int) for idx in embeddings.frame_indices):
            raise ValueError("frame_indices should be a list of integers")

        # Prepare data for insertion
        data = [
            {
                "path": path,
                "frame_idx": frame_idx,
                "embedding": embedding.tolist(),
            }
            for frame_idx, embedding in zip(embeddings.frame_indices, embeddings.embeddings)
        ]

        try:
            # collection = Collection(name=self.collection_name)
            self.client.insert(collection_name=self.collection_name, data=data)
            logger.info(f"Inserted {len(data)} records into '{self.collection_name}'.")
            logger.info(f"Inserted {len(data)} records into '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            raise

    def search(self, query_embedding: Any, top_k: int = 5) -> list[list[dict]]:
        """Searches for similar embeddings in the database.
        Args:
            query_embedding: The embedding to search for.
            top_k: The number of results to return.
        Returns:
            A list of search results.
        Raises:
            ValueError: If the query embedding has the wrong dimensions.
        """
        if len(query_embedding) != self.embedding_size:
            raise ValueError(
                f"Query embedding should have {self.embedding_size} \
                    dimensions, but got {len(query_embedding)}"
            )

        try:
            # Ensure the collection is loaded
            self.client.load_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' loaded into memory.")
            logger.info(f"Collection '{self.collection_name}' loaded into memory.")

            # Perform the search
            return self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                anns_field="embedding",
                search_params=self.search_params,
                limit=top_k,
                output_fields=["path", "frame_idx"],
            )
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise

    def query(self, expr: str) -> list[dict[str, Any]]:
        """Queries the database.
        Args:
            expr: The query expression.
        Returns:
            A list of query results.
        """
        result = self.client.query(collection_name=self.collection_name, filter=expr)
        logger.info(result)
        return result

    def video_exists(self, path: Path) -> bool:
        """Checks if a video exists in the database.
        Args:
            path: The path to the video.
        Returns:
            True if the video exists, False otherwise.
        """
        try:
            # Check if a video with the given path exists in the collection
            result = self.client.query(
                collection_name=self.collection_name,
                filter=f"video_name == '{str(path)}'",
                output_fields=["id"],
            )
            exists = len(result) > 0
            logger.info(f"Video exists: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Error checking if video exists: {e}")
            raise

    def get_all_videos_and_frame_indices(self) -> dict[str, list[int]]:
        """Retrieves all video names and their corresponding frame indices from the database.
        Returns:
            A dictionary where keys are video names and values are lists of frame indices.
        """
        try:
            # A filter to select all entries.
            # Note: The default limit for query is 100, so we set it to the max value.
            results = self.client.query(
                collection_name=self.collection_name,
                filter="id >= 0",
                output_fields=["video_name", "frame_idx"],
                limit=16384,
            )

            video_data: dict[str, list[int]] = {}
            for res in results:
                video_name = res["video_name"]
                frame_idx = res["frame_idx"]
                if video_name not in video_data:
                    video_data[video_name] = []
                video_data[video_name].append(frame_idx)

            # Sort frame indices for each video
            for video_name in video_data:
                video_data[video_name].sort()

            logger.info(f"Retrieved data for {len(video_data)} videos.")
            return video_data
        except Exception as e:
            logger.error(f"Error retrieving all videos and frame indices: {e}")
            raise

    def delete_file(self, video_name: str) -> None:
        """Deletes a video from the database.
        Args:
            video_name: The name of the video to delete.
        """
        try:
            collection = Collection(name=self.collection_name)
            collection.delete(expr=f"video_name == '{video_name}'")
            logger.info(f"Deleted video '{video_name}' from '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Error deleting video: {e}")
            raise


# Class for managing video database operations
class VideoDatabase:
    """A class for managing video database operations."""

    def __init__(
        self,
        model: VLMBaseModel,
        video_handler: VideoHandler,
        database_handler: MilvusHandler,
        frame_skip: int,
    ):
        """Initializes the VideoDatabase object.
        Args:
            model: The VLMBaseModel to use for extracting embeddings.
            video_handler: The VideoHandler to use for processing videos.
            database_handler: The MilvusHandler to use for database operations.
            frame_skip: The number of frames to skip between embeddings.
        """
        self.model = model
        self.frame_skip = frame_skip
        self.video_handler = video_handler
        self.database_handler = database_handler

    def add_video_to_database(self, video_path: Path) -> None:
        """Adds a video to the database.
        Args:
            video_path: The path to the video.
        """
        embeddings, frame_indices = self.video_handler.extract_frame_embeddings(
            str(video_path), self.frame_skip
        )

        self.database_handler.save_embeddings(str(video_path), embeddings, frame_indices)
        logger.info(f"Video {video_path.name} added to the database.")

    def add_videos_from_folder(self, folder_path: Path) -> None:
        """Adds all videos from a folder to the database.
        Args:
            folder_path: The path to the folder.
        """
        paths = [path for i in VIDEO_SUFFIXES for path in folder_path.glob("*." + i)]
        for video_path in paths:
            self.add_video_to_database(video_path.resolve())
