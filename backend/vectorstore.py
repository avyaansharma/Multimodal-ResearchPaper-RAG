import faiss
import numpy as np
import os
import pickle

class VectorStore:
    def __init__(self, index_dir="indexes", dimension_text=1024, dimension_image=512):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        
        # FAISS indices
        self.text_index = faiss.IndexFlatIP(dimension_text)
        self.image_index = faiss.IndexFlatIP(dimension_image)
        
        # Metadata storage
        self.text_metadata = []
        self.image_metadata = []

    def add_text_embeddings(self, embeddings, metadata_list):
        """Add text embeddings and their metadata to the index."""
        embeddings = np.array(embeddings).astype('float32')
        self.text_index.add(embeddings)
        self.text_metadata.extend(metadata_list)

    def add_image_embeddings(self, embeddings, metadata_list):
        """Add image embeddings and their metadata to the index."""
        embeddings = np.array(embeddings).astype('float32')
        self.image_index.add(embeddings)
        self.image_metadata.extend(metadata_list)

    def search_text(self, query_embedding, k=5):
        """Search for top-k similar text chunks."""
        query_embedding = np.array(query_embedding).astype('float32')
        distances, indices = self.text_index.search(query_embedding, k)
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:
                results.append({
                    "metadata": self.text_metadata[idx],
                    "score": float(distances[0][i])
                })
        return results

    def search_image(self, query_embedding, k=3):
        """Search for top-k similar images."""
        query_embedding = np.array(query_embedding).astype('float32')
        distances, indices = self.image_index.search(query_embedding, k)
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:
                results.append({
                    "metadata": self.image_metadata[idx],
                    "score": float(distances[0][i])
                })
        return results

    def save_indices(self):
        """Save FAISS indices and metadata to disk."""
        faiss.write_index(self.text_index, os.path.join(self.index_dir, "text_index.faiss"))
        faiss.write_index(self.image_index, os.path.join(self.index_dir, "image_index.faiss"))
        
        with open(os.path.join(self.index_dir, "metadata.pkl"), "wb") as f:
            pickle.dump({
                "text": self.text_metadata,
                "image": self.image_metadata
            }, f)

    def load_indices(self):
        """Load FAISS indices and metadata from disk."""
        text_path = os.path.join(self.index_dir, "text_index.faiss")
        image_path = os.path.join(self.index_dir, "image_index.faiss")
        meta_path = os.path.join(self.index_dir, "metadata.pkl")
        
        if os.path.exists(text_path):
            self.text_index = faiss.read_index(text_path)
        if os.path.exists(image_path):
            self.image_index = faiss.read_index(image_path)
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
                self.text_metadata = data["text"]
                self.image_metadata = data["image"]

if __name__ == "__main__":
    # Test stub
    vs = VectorStore()
    print("Vector Store initialized.")
