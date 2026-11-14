import os
import shutil
import chromadb
from chromadb.config import Settings
import numpy as np

class VectorDB:
    def __init__(self, db_path="./chroma_db"):
        self.db_path = db_path
        self._ensure_storage()
        
        print(f"[DB] Initializing ChromaDB PersistentClient at '{self.db_path}'...")
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name="faces",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[DB] ChromaDB collection 'faces' ready. Current face count: {self.collection.count()}")

    def _ensure_storage(self):
        """Ensures the directory structure is set up even if Chroma files were uploaded to the root dir."""
        os.makedirs(self.db_path, exist_ok=True)
        # If chroma.sqlite3 exists in the root directory but not in db_path, copy it over
        root_chroma = "chroma.sqlite3"
        target_chroma = os.path.join(self.db_path, "chroma.sqlite3")
        if os.path.exists(root_chroma) and not os.path.exists(target_chroma):
            print(f"[DB] Copying root {root_chroma} into {self.db_path}/")
            shutil.copy2(root_chroma, target_chroma)
            # Also copy any .bin files if present
            for fname in os.listdir("."):
                if fname.endswith(".bin"):
                    shutil.copy2(fname, os.path.join(self.db_path, fname))

    def normalize(self, emb: np.ndarray) -> np.ndarray:
        """L2 normalizes a 1D embedding vector with zero-division protection."""
        emb = np.asarray(emb, dtype=np.float32).flatten()
        norm = np.linalg.norm(emb)
        if norm > 1e-10:
            return emb / norm
        return emb

    def add_face(self, face_id: str, name: str, embedding: np.ndarray) -> bool:
        """
        Adds a face embedding to the database.
        Returns True if insertion is verified, False otherwise.
        """
        norm_emb = self.normalize(embedding)
        try:
            self.collection.add(
                ids=[face_id],
                embeddings=[norm_emb.tolist()],
                metadatas=[{"name": name.strip()}]
            )
            # Verify persistence
            check = self.collection.get(ids=[face_id])
            if check and check['ids'] and face_id in check['ids']:
                print(f"[DB] Successfully stored & verified face: '{name}' (ID: {face_id})")
                return True
            else:
                print(f"[DB] Verification failed for face ID {face_id}")
                return False
        except Exception as e:
            print(f"[DB] Error adding face {face_id}: {e}")
            return False

    def search_face(self, embedding: np.ndarray, threshold: float = 0.5):
        """
        Searches for the closest face using cosine distance (distance = 1 - cosine_similarity).
        Smaller distance means higher similarity.
        Returns: (name, distance)
        """
        if self.collection.count() == 0:
            print("[RECOGNITION] ChromaDB is empty (0 registered faces). Result: Unknown")
            return "Unknown", 1.0

        norm_emb = self.normalize(embedding)
        results = self.collection.query(
            query_embeddings=[norm_emb.tolist()],
            n_results=1
        )

        if not results['ids'] or not results['ids'][0]:
            return "Unknown", 1.0

        distance = float(results['distances'][0][0])
        name = results['metadatas'][0][0].get('name', 'Unknown')
        similarity = max(0.0, 1.0 - distance)

        print(f"[RECOGNITION] Nearest candidate: '{name}' | Cosine Dist: {distance:.4f} | Sim: {similarity:.4f} | Threshold: {threshold:.2f}")

        if distance < threshold:
            print(f"[RECOGNITION] Match verified: '{name}' (ACCESS GRANTED)")
            return name, distance
        else:
            print(f"[RECOGNITION] Distance {distance:.4f} >= {threshold:.2f} -> Unknown (ACCESS DENIED)")
            return "Unknown", distance

    def get_all_faces(self):
        """Returns structured list of registered identities."""
        data = self.collection.get(include=['metadatas'])
        faces = []
        if data and 'ids' in data:
            for i, fid in enumerate(data['ids']):
                meta = data['metadatas'][i] if data.get('metadatas') else {}
                faces.append({
                    "id": fid,
                    "name": meta.get("name", "Unknown") if meta else "Unknown"
                })
        return {
            "count": len(faces),
            "faces": faces
        }

    def delete_face(self, face_id: str) -> bool:
        try:
            self.collection.delete(ids=[face_id])
            return True
        except Exception as e:
            print(f"[DB] Failed to delete face {face_id}: {e}")
            return False

db = VectorDB()
