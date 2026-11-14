import io
import cv2
import numpy as np
import base64
import os
import uuid
import time
import datetime
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image

import database

app = FastAPI(title="AI Surveillance System API", version="2.0.0")

# Cross-Origin Resource Sharing (CORS) Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("captured_faces", exist_ok=True)
app.mount("/captured_faces", StaticFiles(directory="captured_faces"), name="captured_faces")

# System Status Tracker
SYSTEM_STATUS = {
    "server": "online",
    "models_ready": False,
    "face_model_ready": False,
    "database_ready": True,
    "vector_database_ready": False,
    "environment_model_ready": False,
    "optional_models_ready": False,
    "error": None
}

face_model = None
env_analyzer = None
face_analyzer = None
tts = None
db = None

# Configurable Similarity Threshold (Default 0.5 for cosine distance on FaceNet)
SIMILARITY_THRESHOLD = float(os.environ.get("FACE_SIMILARITY_THRESHOLD", 0.50))

@app.on_event("startup")
def startup_event():
    def load_all_models():
        global face_model, env_analyzer, face_analyzer, tts, db
        print("[STARTUP] Initializing AI Surveillance System backend...")
        
        # 1. Initialize SQLite Database
        try:
            database.init_db()
            SYSTEM_STATUS["database_ready"] = True
            print("[STARTUP] SQLite Database initialized.")
        except Exception as e:
            SYSTEM_STATUS["error"] = f"Database error: {e}"
            print(f"[STARTUP] SQLite Database error: {e}")

        # 2. Initialize ChromaDB Vector Database
        try:
            from vector_db import db as vector_db
            db = vector_db
            SYSTEM_STATUS["vector_database_ready"] = True
            print("[STARTUP] ChromaDB Vector Database ready.")
        except Exception as e:
            SYSTEM_STATUS["error"] = f"ChromaDB error: {e}"
            print(f"[STARTUP] ChromaDB error: {e}")

        # 3. Initialize Core FaceNet Model (MTCNN + InceptionResnetV1)
        try:
            from ml_models import FaceRecognitionModel
            face_model = FaceRecognitionModel()
            SYSTEM_STATUS["face_model_ready"] = True
            SYSTEM_STATUS["models_ready"] = True
            print("[STARTUP] Core Face Recognition Model is ONLINE and READY!")
        except Exception as e:
            SYSTEM_STATUS["error"] = f"Face model error: {e}"
            print(f"[STARTUP] Core Face Recognition Model failed to load: {e}")

        # 4. Initialize Optional Models (BLIP, DeepFace, TTS) - fail-safe
        try:
            from ml_models import EnvironmentAnalyzer, FaceAnalyzer
            env_analyzer = EnvironmentAnalyzer()
            face_analyzer = FaceAnalyzer()
            SYSTEM_STATUS["environment_model_ready"] = True
            SYSTEM_STATUS["optional_models_ready"] = True
            print("[STARTUP] Optional Models initialized.")
        except Exception as e:
            print(f"[STARTUP] Optional models note: {e}")

        try:
            from tts import tts as t
            tts = t
            print("[STARTUP] TTS system initialized.")
        except Exception as e:
            print(f"[STARTUP] TTS note: {e}")

        print("[STARTUP] Server startup complete. Full pipeline ready for connections.")

    threading.Thread(target=load_all_models, daemon=True).start()


@app.get("/health")
async def health_check():
    """Health check endpoint for cloud load balancers and frontends."""
    return {
        "status": "ok",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "models_ready": SYSTEM_STATUS["models_ready"]
    }

@app.get("/status")
async def get_system_status():
    """Detailed subsystem readiness status."""
    return SYSTEM_STATUS

@app.get("/faces")
async def get_faces():
    """Lists all registered faces in the vector database."""
    if not db:
        return {"count": 0, "faces": []}
    return db.get_all_faces()

@app.get("/stats")
async def get_stats():
    """Returns overview statistics for dashboard."""
    if not db:
        return database.get_stats(0)
    all_faces = db.get_all_faces()
    total_known = all_faces.get("count", 0)
    return database.get_stats(total_known)

@app.get("/logs")
async def get_logs(limit: int = 50):
    """Returns historic detection logs."""
    return database.get_logs(limit=limit)


# ==========================================
# WEBSOCKET: FACE REGISTRATION
# ==========================================
@app.websocket("/ws/register")
async def register_websocket(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected to /ws/register")
    
    if not face_model or not db:
        print("[WS] Registration rejected: Core models still loading")
        await websocket.send_json({
            "status": "error",
            "message": "AI models are still initializing in the cloud. Please try again in a few seconds."
        })
        await websocket.close()
        return

    embeddings = []
    identity_name = "Unknown"
    saved_face_crop = None
    
    try:
        # First message is the person's name
        data = await websocket.receive_text()
        identity_name = data.strip()
        print(f"[REGISTER] Starting registration for identity: '{identity_name}'")
        
        while True:
            data = await websocket.receive_text()
            if data == "DONE":
                print(f"[REGISTER] Received DONE signal. Frames collected: {len(embeddings)}")
                break
                
            if data.startswith("data:image"):
                _, data = data.split(",", 1)
            
            img_bytes = base64.b64decode(data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            faces = face_model.detect_faces(frame)
            if faces:
                x, y, w, h = faces[0]
                face_crop = frame[y:y+h, x:x+w]
                if saved_face_crop is None:
                    saved_face_crop = face_crop.copy()
                
                # Extract L2-normalized embedding
                emb = face_model.get_embedding(face_crop)
                embeddings.append(emb)
                
            await websocket.send_json({"status": "captured", "count": len(embeddings)})
            
        if len(embeddings) > 0:
            print(f"[REGISTER] Averaging {len(embeddings)} normalized embeddings for '{identity_name}'...")
            avg_emb = np.mean(embeddings, axis=0)
            avg_norm = np.linalg.norm(avg_emb)
            if avg_norm > 1e-10:
                avg_emb = avg_emb / avg_norm
            
            face_id = str(uuid.uuid4())
            
            # Save reference image
            if saved_face_crop is not None:
                image_filename = f"registered_{face_id}.jpg"
                image_path = os.path.join("captured_faces", image_filename)
                cv2.imwrite(image_path, saved_face_crop)
            else:
                image_filename = None

            # Add to ChromaDB and verify
            verified = db.add_face(face_id, identity_name, avg_emb)
            
            if verified:
                # Log registration event
                database.log_event(
                    "REGISTER",
                    f"{identity_name} | REGISTERED | Face enrolled with {len(embeddings)} angles",
                    "Registration Camera",
                    1.0,
                    f"/captured_faces/{image_filename}" if image_filename else None
                )
                print(f"[REGISTER] Success: {identity_name} successfully enrolled and verified in ChromaDB!")
                await websocket.send_json({
                    "status": "success",
                    "message": f"Successfully registered and verified {identity_name} using {len(embeddings)} frames!"
                })
            else:
                await websocket.send_json({
                    "status": "error",
                    "message": "Database storage verification failed. Please try again."
                })
        else:
            await websocket.send_json({
                "status": "error",
                "message": "No valid face could be detected in the captured frames. Please ensure your face is well-lit and centered."
            })
            
    except WebSocketDisconnect:
        print(f"[WS] Registration client disconnected for '{identity_name}'")
    except Exception as e:
        print(f"[WS] Registration error: {e}")
        try:
            await websocket.send_json({"status": "error", "message": f"Server error: {str(e)}"})
        except Exception:
            pass


# ==========================================
# WEBSOCKET: LIVE SURVEILLANCE
# ==========================================
@app.websocket("/ws/surveillance")
async def surveillance_websocket(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected to /ws/surveillance")

    last_spoken_time = 0
    last_log_time = 0
    frame_count = 0
    
    try:
        while True:
            data = await websocket.receive_text()
            
            if not face_model or not db:
                await websocket.send_json({
                    "faces": [],
                    "environment": "Loading AI Models in Cloud...",
                    "status": "loading"
                })
                continue
                
            if data.startswith("data:image"):
                _, data = data.split(",", 1)
            
            try:
                img_bytes = base64.b64decode(data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                continue

            if frame is None:
                continue

            frame_count += 1
            faces = face_model.detect_faces(frame)
            results = []
            current_time = time.time()
            
            env_caption = "Active Surveillance Zone"
            if env_analyzer and faces and (frame_count % 30 == 0):
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    env_caption = env_analyzer.analyze(Image.fromarray(rgb_frame))
                except Exception:
                    pass

            for (x, y, w, h) in faces:
                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size == 0:
                    continue

                # Optional Anti-Spoofing and Gender Check
                is_real = True
                gender = "Person"
                if face_analyzer:
                    is_real, gender = face_analyzer.analyze_face(frame, (x, y, w, h))

                if not is_real:
                    results.append({
                        "box": [int(x), int(y), int(w), int(h)],
                        "name": "Spoof Attempt",
                        "score": 0.0,
                        "access": "DENIED",
                        "gender": "N/A"
                    })
                    continue

                # Core Face Recognition Embedding & Search
                embedding = face_model.get_embedding(face_crop)
                name, score = db.search_face(embedding, threshold=SIMILARITY_THRESHOLD)
                
                access = "GRANTED" if name != "Unknown" else "DENIED"
                confidence = float(score) if score is not None else 0.0

                # Periodic Logging (Debounced to once every 5 seconds to avoid flooding SQLite/disk)
                if current_time - last_log_time > 5.0:
                    last_log_time = current_time
                    prefix = "known" if access == "GRANTED" else "unknown"
                    image_filename = f"{prefix}_{int(current_time)}.jpg"
                    image_path = os.path.join("captured_faces", image_filename)
                    cv2.imwrite(image_path, face_crop)
                    
                    event_type = "KNOWN" if access == "GRANTED" else "UNKNOWN"
                    log_text = f"{name} ({gender}) | {access} | {env_caption}"
                    database.log_event(event_type, log_text, "Main Camera", confidence, f"/captured_faces/{image_filename}")
                    
                    # Optional Audio Alerts
                    if tts:
                        if access == "DENIED":
                            tts.speak(f"Alert. Unknown individual detected.")
                        else:
                            tts.speak(f"Welcome {name}.")

                results.append({
                    "box": [int(x), int(y), int(w), int(h)],
                    "name": name,
                    "score": confidence,
                    "access": access,
                    "gender": gender
                })

            await websocket.send_json({
                "faces": results,
                "environment": env_caption,
                "status": "active"
            })
            
    except WebSocketDisconnect:
        print("[WS] Surveillance client disconnected")
    except Exception as e:
        print(f"[WS] Surveillance error: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"[STARTUP] Binding Uvicorn to 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
