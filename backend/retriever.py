from .embedder import MultimodalEmbedder
from .vectorstore import VectorStore
import os

class MultimodalRetriever:
    def __init__(self, index_dir="indexes"):
        self.embedder = MultimodalEmbedder()
        self.vector_store = VectorStore(index_dir=index_dir)
        self.vector_store.load_indices()

    def retrieve(self, query, text_k=5, image_k=2):
        """Performs multimodal retrieval: both text chunks and images."""
        
        # 1. Embed query for text retrieval (BGE)
        text_query_embedding = self.embedder.embed_text([query])
        text_results = self.vector_store.search_text(text_query_embedding, k=text_k)
        
        # 2. Embed query for image retrieval (CLIP)
        image_query_embedding = self.embedder.embed_query_clip(query)
        image_results = self.vector_store.search_image(image_query_embedding, k=image_k)
        
        return {
            "text_chunks": text_results,
            "images": image_results
        }

if __name__ == "__main__":
    # Test stub
    retriever = MultimodalRetriever()
    print("Retriever initialized.")
