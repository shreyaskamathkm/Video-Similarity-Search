import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class BaseModel:
    def __init__(self):
        self.model = None

    def extract_text_features(self, text: str) -> np.ndarray:
        raise NotImplementedError("This should be implemented in the subclass")

    def extract_image_features(self, image: Image.Image) -> np.ndarray:
        raise NotImplementedError("This should be implemented in the subclass")


class CLIPModelProcessor(BaseModel):
    """Handles feature extraction using a pre-trained CLIP model."""

    def __init__(self):
        """Initializes the CLIP model and processor."""
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def extract_text_features(self, text: str) -> np.ndarray:
        """Extracts text embeddings using the CLIP model.

        Args:
            text (str): Input text.

        Returns:
            np.ndarray: Text feature vector.
        """
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        return self.model.get_text_features(**inputs).detach().numpy()

    def extract_image_features(self, image: Image.Image) -> np.ndarray:
        """Extracts image embeddings using the CLIP model.

        Args:
            image (Image.Image): Input image.

        Returns:
            np.ndarray: Image feature vector.
        """
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        return self.model.get_image_features(**inputs).detach().numpy()
