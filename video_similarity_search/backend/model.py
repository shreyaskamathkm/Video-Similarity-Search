from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# SOLID Principle Implementation


# Base class for Model operations
class Model:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def extract_text_features(self, text: str):
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        return self.model.get_text_features(**inputs).detach().numpy()

    def extract_image_features(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        return self.model.get_image_features(**inputs).detach().numpy()
