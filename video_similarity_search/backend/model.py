import logging

import numpy as np
import open_clip
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# Base class for Model operations
class VLMBaseModel:
    def __init__(self) -> None:
        self.model = None

    def extract_text_features(self, text: str) -> np.ndarray:
        raise NotImplementedError("This should be implemented in the subclass")

    def extract_image_features(self, image: Image.Image) -> np.ndarray:
        raise NotImplementedError("This should be implemented in the subclass")


class ClipModel(VLMBaseModel):
    def __init__(self, ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
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

    def get_embedding_length(self):
        return self._embedding_length
