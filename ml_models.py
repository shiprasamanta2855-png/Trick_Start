import cv2
import numpy as np
import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceRecognitionModel:
    """Core Face Recognition Model using MTCNN (detection) and FaceNet InceptionResnetV1 (512-d embeddings)."""
    def __init__(self):
        print("[MODEL] Initializing Face Detection (MTCNN)...")
        # CPU-safe MTCNN configuration
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=device)
        
        print("[MODEL] Initializing Face Recognition (FaceNet InceptionResnetV1 pretrained on VGGFace2)...")
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        print("[MODEL] Core FaceNet recognition model loaded successfully!")

    def preprocess(self, face_img):
        # Resize to standard 160x160 for InceptionResnetV1
        img = cv2.resize(face_img, (160, 160))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # Standard FaceNet fixed normalization: (x - 0.5) / 0.5
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return torch.tensor(img, dtype=torch.float32).to(self.device)

    def get_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """Extracts and returns a 512-dim L2-normalized face embedding."""
        input_tensor = self.preprocess(face_img)
        with torch.no_grad():
            emb = self.resnet(input_tensor).cpu().numpy()[0]
        
        # Explicit L2 normalization with zero-division safeguard
        norm = np.linalg.norm(emb)
        if norm > 1e-10:
            emb = emb / norm
        return emb.astype(np.float32)

    def detect_faces(self, frame: np.ndarray):
        """Returns list of (x, y, w, h) bounding boxes."""
        if frame is None or frame.size == 0:
            return []
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            boxes, _ = self.mtcnn.detect(rgb_frame)
        except Exception as e:
            print(f"[MODEL] Face detection error: {e}")
            return []
        
        faces = []
        if boxes is not None:
            h_frame, w_frame = frame.shape[:2]
            for box in boxes:
                x1, y1, x2, y2 = box
                x = max(0, int(x1))
                y = max(0, int(y1))
                w = min(w_frame - x, int(x2 - x1))
                h = min(h_frame - y, int(y2 - y1))
                if w >= 20 and h >= 20: # Filter out tiny noise artifacts
                    faces.append((x, y, w, h))
        return faces


class EnvironmentAnalyzer:
    """Optional Environment Captioning Model (BLIP or lightweight fallback)."""
    def __init__(self):
        # On cloud free-tiers with <=512MB RAM, loading Salesforce/BLIP causes immediate OOM crashes.
        # We enable it only if explicitly configured via ENABLE_BLIP=true.
        self.enabled = os.environ.get("ENABLE_BLIP", "false").lower() in ("true", "1", "yes")
        self.processor = None
        self.model = None
        self.is_loaded = False
        
        if self.enabled:
            print("[OPTIONAL MODEL] Loading Salesforce/blip-image-captioning-base...")
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.is_loaded = True
                print("[OPTIONAL MODEL] BLIP loaded successfully.")
            except Exception as e:
                print(f"[OPTIONAL MODEL] BLIP could not be loaded ({e}). Using lightweight environment analyzer.")
                self.is_loaded = False
        else:
            print("[OPTIONAL MODEL] BLIP disabled to preserve cloud RAM (set ENABLE_BLIP=true to activate).")

    def analyze(self, image) -> str:
        if self.is_loaded and self.processor and self.model:
            try:
                inputs = self.processor(image, return_tensors="pt")
                out = self.model.generate(**inputs, max_new_tokens=25)
                caption = self.processor.decode(out[0], skip_special_tokens=True)
                return caption
            except Exception as e:
                print(f"[OPTIONAL MODEL] BLIP inference error: {e}")
        return "person visible in camera field"


class FaceAnalyzer:
    """Optional Anti-Spoofing and Demographic Analysis (DeepFace or fallback)."""
    def __init__(self):
        self.enabled = os.environ.get("ENABLE_DEEPFACE", "false").lower() in ("true", "1", "yes")
        self.is_loaded = False
        
        if self.enabled:
            print("[OPTIONAL MODEL] Initializing DeepFace models...")
            try:
                from deepface import DeepFace
                self.DeepFace = DeepFace
                self.is_loaded = True
                print("[OPTIONAL MODEL] DeepFace initialized.")
            except Exception as e:
                print(f"[OPTIONAL MODEL] DeepFace initialization failed: {e}")
                self.is_loaded = False
        else:
            print("[OPTIONAL MODEL] DeepFace disabled to preserve cloud RAM (set ENABLE_DEEPFACE=true to activate).")

    def analyze_face(self, frame_bgr, face_box):
        """
        Analyzes face for anti-spoofing and gender.
        Returns: (is_real: bool, gender_str: str)
        """
        if not self.is_loaded:
            return True, "Person"

        x, y, w, h = face_box
        padding = 20
        h_img, w_img = frame_bgr.shape[:2]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)
        crop = frame_bgr[y1:y2, x1:x2]

        is_real = True
        gender = "Person"

        try:
            results = self.DeepFace.analyze(
                crop,
                actions=['gender'],
                enforce_detection=False,
                anti_spoofing=True,
                silent=True
            )
            res = results[0] if isinstance(results, list) else results
            if 'is_real' in res:
                is_real = bool(res['is_real'])
            if 'dominant_gender' in res:
                gender = str(res['dominant_gender'])
        except Exception as e:
            # DeepFace errors should never block surveillance
            pass

        return is_real, gender
