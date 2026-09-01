import io
import cv2
import numpy as np
import base64
import os
import uuid
import time
try:
    import winsound
except ImportError:
    class DummyWinsound:
        def Beep(self, freq, duration):
            pass
    winsound = DummyWinsound()
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

import database

app = FastAPI(title="AI Surveillance System API")

face_model = None
env_analyzer = None
face_analyzer = None
tts = None
db = None

@app.on_event("startup")
def startup_event():
    def load_models():
        global face_model, env_analyzer, face_analyzer, tts, db
        print("Starting background load of ML models...")
        from vector_db import db as vector_db
        from ml_models import face_model as fm, env_analyzer as ea, face_analyzer as fa
        from tts import tts as t
        db = vector_db
        face_model = fm
        env_analyzer = ea
        face_analyzer = fa
        tts = t
        print("ML Models loaded!")
    threading.Thread(target=load_models, daemon=True).start()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("captured_faces", exist_ok=True)
app.mount("/captured_faces", StaticFiles(directory="captured_faces"), name="captured_faces")


@app.get("/stats")
async def get_stats():
    if not db:
        return database.get_stats(0)
    total_known = len(db.get_all_faces().get('ids', []))
    return database.get_stats(total_known)

@app.get("/logs")
async def get_logs():
    return database.get_logs()


@app.websocket("/ws/register")
async def register_websocket(websocket: WebSocket):
    await websocket.accept()
    if not face_model:
        await websocket.send_json({"status": "error", "message": "Models loading..."})
        await websocket.close()
        return
    embeddings = []
    identity_name = "Unknown"
    
    try:
        # First message should be the name
        data = await websocket.receive_text()
        identity_name = data
        
        while True:
            data = await websocket.receive_text()
            if data == "DONE":
                break
                
            if data.startswith("data:image"):
                header, data = data.split(",", 1)
            
            img_bytes = base64.b64decode(data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            faces = face_model.detect_faces(frame)
            if faces:
                x, y, w, h = faces[0] # Assume one person during registration
                face_crop = frame[y:y+h, x:x+w]
                emb = face_model.get_embedding(face_crop)
                embeddings.append(emb)
                
            await websocket.send_json({"status": "captured", "count": len(embeddings)})
            
        if len(embeddings) > 0:
            # Average the embeddings for a robust 3D-like profile
            avg_emb = np.mean(embeddings, axis=0)
            face_id = str(uuid.uuid4())
            db.add_face(face_id, identity_name, avg_emb)
            await websocket.send_json({"status": "success", "message": f"Successfully registered {identity_name} using {len(embeddings)} frames!"})
        else:
            await websocket.send_json({"status": "error", "message": "No faces could be captured."})
            
    except WebSocketDisconnect:
        print("Registration client disconnected")


@app.websocket("/ws/surveillance")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if not face_model:
        await websocket.send_json({"faces": [], "environment": "Models loading..."})

    frame_count = 0
    last_spoken_time = 0
    
    try:
        while True:
            data = await websocket.receive_text()
            if not face_model:
                await websocket.send_json({"faces": [], "environment": "Loading AI Models (~5 mins on free tier)..."})
                continue
            if data.startswith("data:image"):
                header, data = data.split(",", 1)
            
            img_bytes = base64.b64decode(data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue

            frame_count += 1
            faces = face_model.detect_faces(frame)
            results = []
            
            current_time = time.time()
            env_caption = ""
            
            # Compute environment context once per batch of faces
            if faces and frame_count % 30 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                env_caption = env_analyzer.analyze(pil_img)
            
            for (x, y, w, h) in faces:
                face_crop = frame[y:y+h, x:x+w]
                
                # Anti-Spoofing and Gender Check
                is_real, gender = face_analyzer.analyze_face(frame, (x, y, w, h))
                
                if not is_real:
                    event_type = "SPOOF"
                    name = "Spoof Attempt"
                    score = 0.0
                    
                    if current_time - last_spoken_time > 5:
                        image_filename = f"spoof_{int(current_time)}.jpg"
                        image_path = os.path.join("captured_faces", image_filename)
                        cv2.imwrite(image_path, face_crop)
                        database.log_event("SPOOF", "Spoof Attempt | DENIED | Fake Photo/Screen", "Main Camera", None, f"/captured_faces/{image_filename}")
                        
                        threading.Thread(target=winsound.Beep, args=(2500, 1000), daemon=True).start()
                        tts.speak("Alert. Spoofing attempt detected. Please show a real face.")
                        last_spoken_time = current_time
                        
                    results.append({
                        "box": [int(x), int(y), int(w), int(h)],
                        "name": "SPOOF",
                        "score": 0.0,
                        "access": "SPOOF",
                        "gender": "N/A"
                    })
                    continue
                
                # If Real Person, proceed with recognition
                embedding = face_model.get_embedding(face_crop)
                name, score = db.search_face(embedding, threshold=0.5)
                if not name:
                    name = "Unknown"
                
                event_type = "UNKNOWN" if name == "Unknown" else "KNOWN"
                
                # Ensure we have an env_caption before logging
                if not env_caption:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    env_caption = env_analyzer.analyze(Image.fromarray(rgb_frame))

                # Append Known/Unknown status and Environment to log identity
                access_status = "DENIED" if event_type == "UNKNOWN" else "GRANTED"
                base_name = "Unknown (Unknown)" if event_type == "UNKNOWN" else f"{name} (Known)"
                log_identity = f"{base_name} | {access_status} | {env_caption}"
                
                # Handling face saving and logging for ALL real people
                if current_time - last_spoken_time > 5:
                    prefix = "unknown" if event_type == "UNKNOWN" else "known"
                    image_filename = f"{prefix}_{int(current_time)}.jpg"
                    image_path = os.path.join("captured_faces", image_filename)
                    cv2.imwrite(image_path, face_crop)
                    database.log_event(event_type, log_identity, "Main Camera", score, f"/captured_faces/{image_filename}")
                
                # Debounced TTS Logic (prevent overlapping speech)
                if current_time - last_spoken_time > 10:
                    last_spoken_time = current_time
                        
                    if event_type == "UNKNOWN":
                        threading.Thread(target=winsound.Beep, args=(2500, 1000), daemon=True).start()
                        tts.speak(f"alert unknown {gender} person detected. The environment shows: {env_caption}")
                    else:
                        tts.speak(f"Welcome {name}, access granted. The environment shows: {env_caption}")
                        
                results.append({
                    "box": [int(x), int(y), int(w), int(h)],
                    "name": name,
                    "score": float(score) if score else 0.0,
                    "access": "GRANTED" if name != "Unknown" else "DENIED",
                    "gender": gender
                })

            await websocket.send_json({
                "faces": results,
                "environment": env_caption
            })
            
    except WebSocketDisconnect:
        print("Surveillance client disconnected")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
