import logging
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# Base class for Model operations
class VLMBaseModel:
    def __init__(self) -> None:
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_text_features(self, text: str) -> np.ndarray:  # type: ignore[override]
        raise NotImplementedError("This should be implemented in the subclass")

    def extract_image_features(self, image: Image.Image) -> np.ndarray:  # type: ignore[override]
        raise NotImplementedError("This should be implemented in the subclass")

    def get_embedding_length(self):  # type: ignore[override]
        raise NotImplementedError("This should be implemented in the subclass")


class ClipModel(VLMBaseModel):
    def __init__(self, model_architecture: str, model_pretrained: str) -> None:
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_architecture, pretrained=model_pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_architecture)
        self._embedding_length = self.model.text_projection.shape[1]

    def extract_text_features(self, text: str) -> np.ndarray:  # type: ignore[override]
        tokens = self.tokenizer([text])
        tokens = tokens.to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
        return features.cpu().numpy()

    def extract_image_features(self, image: Image.Image) -> np.ndarray:  # type: ignore[override]
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)  # type: ignore[override]
        with torch.no_grad():
            features = self.model.encode_image(image_input)
        return features.cpu().numpy()

    def get_embedding_length(self) -> int:
        return self._embedding_length


class Siglip2Model(VLMBaseModel):
    def __init__(self, model_architecture: str, model_pretrained: str, **kwargs: Any) -> None:
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_architecture, pretrained=model_pretrained, device=self.device, **kwargs
        )
        self.tokenizer = open_clip.get_tokenizer(model_architecture)
        self._embedding_length = self.model.text_projection.shape[1]

    def extract_text_features(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text])
        tokens = tokens.to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
        return features.cpu().numpy()

    def extract_image_features(self, image: Image.Image) -> np.ndarray:
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_input)
        return features.cpu().numpy()

    def get_embedding_length(self) -> int:
        return self._embedding_length


_MODEL_MAP = {
    "clip": ClipModel,
    "siglip2": Siglip2Model,
}


def model_factory(model_name: str, *args: Any, **kwargs: Any) -> VLMBaseModel:
    model_class = _MODEL_MAP.get(model_name)
    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}")
    return model_class(*args, **kwargs)
