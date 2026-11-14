import threading
import sys
import os

class TTSEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.engine = None
        self.enabled = False
        
        # Audio output is generally unavailable in headless cloud containers
        if os.environ.get("ENABLE_TTS", "false").lower() in ("true", "1", "yes"):
            try:
                import pyttsx3
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
                self.engine.setProperty('volume', 1.0)
                self.enabled = True
                print("[TTS] Text-to-speech engine initialized.")
            except Exception as e:
                print(f"[TTS] Audio device not found ({e}). Audio announcements disabled.")
                self.engine = None
                self.enabled = False
        else:
            print("[TTS] Audio announcements disabled in cloud mode (set ENABLE_TTS=true to enable).")

    def speak(self, text: str):
        if not self.enabled or not self.engine:
            return
            
        def _speak():
            with self.lock:
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception as e:
                    print(f"[TTS] Speech warning: {e}")

        threading.Thread(target=_speak, daemon=True).start()

tts = TTSEngine()

if __name__ == "__main__":
    tts.speak("System initialized.")
