from pathlib import Path

import yaml
from cloudpathlib import AnyPath, S3Path
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """A class to store the application configuration."""

    video_folder: AnyPath = Field(
        description="Path to the folder containing videos to process.",
    )
    query: str = Field(description="Text query for video similarity search.")
    collection_name: str = Field(
        default="video_search_similarity", description="Name of the collection."
    )
    reset_dataset: bool = Field(
        default=True, description="Wether to drop reset the Milvus dataset."
    )
    frame_skip: int = Field(default=2, description="Number of frames to skip in a video.")
    model_name: str = Field(
        default="clip", description="Name of the model to use (e.g., 'clip', 'siglip2')."
    )
    model_architecture: str = Field(description="Architecture of the model.")
    model_pretrained: str = Field(description="Pretrained weights for the model.")

    @staticmethod
    def _read_yaml(config_path: S3Path | Path) -> dict:
        """Reads a YAML file.

        Args:
            config_path: The path to the YAML file.

        Returns:
            The content of the YAML file.
        """
        with open(config_path) as f:
            return yaml.safe_load(f)

    @classmethod
    def from_yaml(cls, config_path: S3Path | Path) -> "AppConfig":
        """Creates an AppConfig object from a YAML file.

        Args:
            config_path: The path to the YAML file.

        Returns:
            An AppConfig object.
        """
        config = cls._read_yaml(config_path)
        return cls(**config)
