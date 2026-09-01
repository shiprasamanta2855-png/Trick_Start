import chromadb
from chromadb.config import Settings
import numpy as np

class VectorDB:
    def __init__(self, db_path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        # Using cosine similarity as requested by the user.
        # In ChromaDB, you specify this in the metadata when creating the collection.
        self.collection = self.client.get_or_create_collection(
            name="faces",
            metadata={"hnsw:space": "cosine"}
        )

    def add_face(self, face_id: str, name: str, embedding: np.ndarray):
        """
        Adds a face embedding to the database.
        embedding: a 1D numpy array representing the face vector.
        """
        self.collection.add(
            ids=[face_id],
            embeddings=[embedding.tolist()],
            metadatas=[{"name": name}]
        )

    def search_face(self, embedding: np.ndarray, threshold: float = 0.5):
        """
        Searches for the closest face using cosine similarity.
        Returns (name, score) if found within threshold, else None.
        """
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=1
        )
        
        if not results['ids'][0]:
            return None, None
            
        distance = results['distances'][0][0]
        name = results['metadatas'][0][0]['name']
        
        # In ChromaDB with cosine space, distance = 1 - cosine_similarity.
        # Thus, smaller distance means higher similarity.
        if distance < threshold:
            return name, distance
        return "Unknown", distance

    def get_all_faces(self):
        return self.collection.get()

db = VectorDB()
