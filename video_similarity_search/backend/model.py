import logging
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# Base class for Model operations
class VLMBaseModel:
    """A base class for Vision-Language Models."""

    def __init__(self) -> None:
        """Initializes the VLMBaseModel."""
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_text_features(self, text: str) -> np.ndarray:  # type: ignore[override]
        """Extracts features from text.

        Args:
            text: The input text.

        Raises:
            NotImplementedError: This method should be implemented in the subclass.
        """
        raise NotImplementedError("This should be implemented in the subclass")

    def extract_image_features(self, image: Image.Image) -> np.ndarray:  # type: ignore[override]
        """Extracts features from an image.

        Args:
            image: The input image.

        Raises:
            NotImplementedError: This method should be implemented in the subclass.
        """
        raise NotImplementedError("This should be implemented in the subclass")

    def get_embedding_length(self):  # type: ignore[override]
        """Gets the length of the embedding vector.

        Raises:
            NotImplementedError: This method should be implemented in the subclass.
        """
        raise NotImplementedError("This should be implemented in the subclass")


class ClipModel(VLMBaseModel):
    """A class for using CLIP models."""

    def __init__(self, model_architecture: str, model_pretrained: str) -> None:
        """Initializes the ClipModel.

        Args:
            model_architecture: The architecture of the CLIP model.
            model_pretrained: The pretrained weights for the CLIP model.
        """
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_architecture, pretrained=model_pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_architecture)
        self._embedding_length = self.model.text_projection.shape[1]

    def extract_text_features(self, text: str) -> np.ndarray:  # type: ignore[override]
        """Extracts features from text using the CLIP model.

        Args:
            text: The input text.

        Returns:
            A numpy array of text features.
        """
        tokens = self.tokenizer([text])
        tokens = tokens.to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
        return features.cpu().numpy()

    def extract_image_features(self, image: Image.Image) -> np.ndarray:  # type: ignore[override]
        """Extracts features from an image using the CLIP model.

        Args:
            image: The input image.

        Returns:
            A numpy array of image features.
        """
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)  # type: ignore[override]
        with torch.no_grad():
            features = self.model.encode_image(image_input)
        return features.cpu().numpy()

    def get_embedding_length(self) -> int:
        """Gets the length of the embedding vector for the CLIP model.

        Returns:
            The length of the embedding vector.
        """
        return self._embedding_length


class Siglip2Model(VLMBaseModel):
    """A class for using Siglip2 models."""

    def __init__(self, model_architecture: str, model_pretrained: str, **kwargs: Any) -> None:
        """Initializes the Siglip2Model.

        Args:
            model_architecture: The architecture of the Siglip2 model.
            model_pretrained: The pretrained weights for the Siglip2 model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_architecture, pretrained=model_pretrained, device=self.device, **kwargs
        )
        self.tokenizer = open_clip.get_tokenizer(model_architecture)
        self._embedding_length = self.model.text_projection.shape[1]

    def extract_text_features(self, text: str) -> np.ndarray:
        """Extracts features from text using the Siglip2 model.

        Args:
            text: The input text.

        Returns:
            A numpy array of text features.
        """
        tokens = self.tokenizer([text])
        tokens = tokens.to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
        return features.cpu().numpy()

    def extract_image_features(self, image: Image.Image) -> np.ndarray:
        """Extracts features from an image using the Siglip2 model.

        Args:
            image: The input image.

        Returns:
            A numpy array of image features.
        """
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_input)
        return features.cpu().numpy()

    def get_embedding_length(self) -> int:
        """Gets the length of the embedding vector for the Siglip2 model.

        Returns:
            The length of the embedding vector.
        """
        return self._embedding_length


_MODEL_MAP = {
    "clip": ClipModel,
    "siglip2": Siglip2Model,
}


def model_factory(model_name: str, *args: Any, **kwargs: Any) -> VLMBaseModel:
    """A factory function for creating VLM models.

    Args:
        model_name: The name of the model to create.
        *args: Positional arguments to pass to the model constructor.
        **kwargs: Keyword arguments to pass to the model constructor.

    Returns:
        An instance of a VLMBaseModel subclass.

    Raises:
        ValueError: If the model name is unknown.
    """
    model_class = _MODEL_MAP.get(model_name)
    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}")
    return model_class(*args, **kwargs)
