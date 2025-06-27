import logging

import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


# Base class for Model operations
class VLMBaseModel:
    def __init__(self) -> None:
        self.model = None

    def extract_text_features(self, text: str) -> np.ndarray:
        raise NotImplementedError("This should be implemented in the subclass")

    def extract_image_features(self, image: Image.Image) -> np.ndarray:
        raise NotImplementedError("This should be implemented in the subclass")


class Model(VLMBaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def extract_text_features(self, text: str) -> np.ndarray:
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        return self.model.get_text_features(**inputs).detach().numpy()

    def extract_image_features(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        return self.model.get_image_features(**inputs).detach().numpy()
