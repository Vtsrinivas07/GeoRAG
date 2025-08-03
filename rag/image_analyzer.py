# rag/image_analyzer.py
from PIL import Image
import os

class ImageAnalyzer:
    def __init__(self):
        pass  # Load models, etc.

    def analyze(self, image_path, location=None):
        if not os.path.exists(image_path):
            return f"Image not found: {image_path}"
        try:
            img = Image.open(image_path)
            info = {
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
            }
            return f"Image info: {info}"
        except Exception as e:
            return f"Error analyzing image: {e}"