import cv2
import numpy as np
import onnxruntime as ort
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from deepface import DeepFace

class FaceRecognitionModel:
    def __init__(self, model_path="models/face_recognition.onnx"):
        # Face detector using MTCNN from facenet-pytorch
        self.mtcnn = MTCNN(keep_all=True)
        # Replacing the missing ONNX model with a real pre-trained model so embeddings actually work.
        print("Loading FaceNet InceptionResnetV1 model...")
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def preprocess(self, face_img):
        # Resize to 160x160 as expected by InceptionResnetV1
        img = cv2.resize(face_img, (160, 160))
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # Normalize with fixed mean and std
        img = (img - 0.5) / 0.5
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return torch.tensor(img, dtype=torch.float32)

    def get_embedding(self, face_img):
        input_tensor = self.preprocess(face_img)
        with torch.no_grad():
            embedding = self.resnet(input_tensor).numpy()[0]
        return embedding

    def detect_faces(self, frame):
        """Returns list of (x, y, w, h) bounding boxes."""
        # MTCNN expects a PIL Image or RGB numpy array
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = self.mtcnn.detect(rgb_frame)
        
        faces = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box
                # Ensure coordinates are within frame boundaries and valid
                x = max(0, int(x1))
                y = max(0, int(y1))
                w = int(x2 - x1)
                h = int(y2 - y1)
                if w > 0 and h > 0:
                    faces.append((x, y, w, h))
        return faces

class EnvironmentAnalyzer:
    def __init__(self):
        # Using a small version of BLIP for fast inference
        print("Loading BLIP model for environment checking...")
        try:
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.is_loaded = True
            print("BLIP model loaded successfully.")
        except Exception as e:
            print(f"Failed to load BLIP: {e}. Will attempt to download on first run.")
            self.is_loaded = False

    def analyze(self, image: Image.Image) -> str:
        if not self.is_loaded:
            return "Environment checking not available."
            
        inputs = self.processor(image, return_tensors="pt")
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

class FaceAnalyzer:
    def __init__(self):
        # Trigger an initial dummy run to force DeepFace to download models if missing
        print("Initializing DeepFace models for Gender and Anti-Spoofing...")
        try:
            # Create a dummy image
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            DeepFace.analyze(dummy_img, actions=['gender'], enforce_detection=False, silent=True)
            print("DeepFace models loaded successfully.")
        except Exception as e:
            print(f"DeepFace initialization note: {e}")

    def analyze_face(self, frame_bgr, face_box):
        """
        Analyzes a face for anti-spoofing and gender.
        face_box: (x, y, w, h)
        Returns: (is_real, gender_str)
        """
        x, y, w, h = face_box
        
        # We need a slightly larger crop for anti-spoofing context if possible, 
        # but DeepFace handles extracting internally if we pass the whole frame.
        # However, passing the cropped face is faster. Let's pass the whole frame 
        # and tell it where the face is, or just pass the crop.
        # DeepFace anti-spoofing requires facial context. It's best to pass the full image.
        
        # To save compute and avoid DeepFace re-detecting faces across the whole image,
        # we pass a padded crop.
        padding = 30
        h_img, w_img = frame_bgr.shape[:2]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)
        
        padded_crop = frame_bgr[y1:y2, x1:x2]
        
        is_real = True
        gender = "Unknown"
        
        try:
            # anti_spoofing parameter requires DeepFace >= 0.0.80
            # analyze() returns a list of dictionaries if multiple faces, but we pass a crop.
            results = DeepFace.analyze(
                padded_crop, 
                actions=['gender'], 
                enforce_detection=False,
                anti_spoofing=True,
                silent=True
            )
            
            res = results[0] if isinstance(results, list) else results
            
            # Anti-spoofing check
            if 'is_real' in res:
                is_real = res['is_real']
                
            # Gender check (DeepFace returns a dict of probabilities, we take the dominant one)
            if 'dominant_gender' in res:
                gender = res['dominant_gender']
                
        except Exception as e:
            print(f"DeepFace analysis error: {e}")
            
        return is_real, gender

face_model = FaceRecognitionModel()
env_analyzer = EnvironmentAnalyzer()
face_analyzer = FaceAnalyzer()
