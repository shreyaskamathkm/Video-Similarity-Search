from pathlib import Path

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)

from video_similarity_search.backend.embeddingextractor import EmbeddingExtractor

VIDEO_SUFFIXES = ["mp4", "mov"]


class Database:
    def __init__(self, remove_old_data: bool):
        self.remove_old_data = remove_old_data

    def insert_embeddings(self, path: str, embeddings: np.ndarray, frame_indices: list[int]):
        raise NotImplementedError("This should be implemented in the subclass")

    def search(self, query_embedding, top_k):
        raise NotImplementedError("This should be implemented in the subclass")

    def query(self, expr: str):
        raise NotImplementedError("This should be implemented in the subclass")

    def delete_file(self, name: str):
        raise NotImplementedError("This should be implemented in the subclass")


class MilvusDatabase(Database):
    def __init__(self, remove_old_data: bool = True):
        super().__init__(remove_old_data)

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

        # Create index if it doesn't exist
        self._create_index()
        # Load collection after initialization
        self.client.load_collection(self.collection_name)
        print(f"Collection '{self.collection_name}' loaded into memory.")

    def _create_or_get_collection(self):
        if self.client.has_collection(self.collection_name) and self.remove_old_data:
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

    def insert_embeddings(self, video_path: str, embeddings: np.ndarray, frame_indices: list[int]):
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

    def delete_file(self, video_name: str):
        try:
            collection = Collection(name=self.collection_name)
            collection.delete(expr=f"video_name == '{video_name}'")
            print(f"Deleted video '{video_name}' from '{self.collection_name}'.")
        except Exception as e:
            print(f"Error deleting video: {e}")
            raise


class ModalityToDatabase:
    def __init__(self, embeddings_extractor: EmbeddingExtractor, database: Database):
        self.embeddings_extractor = embeddings_extractor
        self.database = database

    def add_files_from_folder(self, folder_path: str):
        raise NotImplementedError("This should be implemented in the subclass")


class VideoToDatabase(ModalityToDatabase):
    def __init__(self, embeddings_extractor: EmbeddingExtractor, database: Database):
        super().__init__(embeddings_extractor, database)

    def _add_video_to_database(self, path: Path):
        embeddings, frame_indices = self.embeddings_extractor.extract_embeddings(path)

        self.database.insert_embeddings(path, embeddings, frame_indices)
        print(f"Video {path.name} added to the database.")

    def add_files_from_folder(self, folder_path: str):
        paths = [path for i in VIDEO_SUFFIXES for path in folder_path.glob("*." + i)]
        for video_path in [paths[0]]:
            self._add_video_to_database(video_path.resolve())
