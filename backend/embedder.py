import os
import logging
from typing import List, Any
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

class MultimodalEmbedder:
    """
    Expert-level embedder for research papers.
    Uses lazy loading for heavyweight libraries to ensure Streamlit stability.
    """
    def __init__(self, 
                 text_model_name: str = "BAAI/bge-large-en-v1.5", 
                 image_model_name: str = "ViT-B-32", 
                 pretrained: str = "laion2b_s34b_b79k"):
        
        # Delayed imports to avoid Streamlit file-watcher conflicts
        import torch
        from sentence_transformers import SentenceTransformer
        import open_clip
        
        logger.info(f"Initializing MultimodalEmbedder with {text_model_name} and {image_model_name}...")
        
        # Load Text Embedding Model
        self.text_model = SentenceTransformer(text_model_name)
        
        # Load CLIP Model
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            image_model_name, 
            pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(image_model_name)
        self.clip_model.eval()
        
        # Detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)
        logger.info(f"Embedder models loaded on {self.device}.")

    def embed_text(self, text_list: List[str]) -> Any:
        import torch
        with torch.no_grad():
            embeddings = self.text_model.encode(text_list, normalize_embeddings=True)
        return embeddings

    def embed_image(self, image_path: str) -> Any:
        import torch
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return None
            
        try:
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()
        except Exception as e:
            logger.error(f"Failed to embed image: {str(e)}")
            return None

    def embed_query_clip(self, query: str) -> Any:
        import torch
        try:
            text = self.tokenizer([query]).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()
        except Exception as e:
            logger.error(f"Failed to embed query: {str(e)}")
            return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    embedder = MultimodalEmbedder()
    print("Embedder verified successfully.")
