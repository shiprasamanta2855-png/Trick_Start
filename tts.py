import pyttsx3
import threading

class TTSEngine:
    def __init__(self):
        # pyttsx3 can have issues if called from multiple threads.
        # So we create a lock to serialize speech requests.
        self.lock = threading.Lock()
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 1.0)
        except Exception as e:
            print(f"Failed to initialize TTS: {e}")
            self.engine = None

    def speak(self, text: str):
        if not self.engine:
            return
            
        def _speak():
            with self.lock:
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception as e:
                    print(f"TTS Error: {e}")

        # Run TTS in a daemon thread so it doesn't block the API
        t = threading.Thread(target=_speak, daemon=True)
        t.start()

# Global TTS instance
tts = TTSEngine()

if __name__ == "__main__":
    tts.speak("System initialized.")
