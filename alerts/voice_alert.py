import threading
import pyttsx3


def _speak(message):
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    engine.setProperty("volume", 1.0)
    engine.say(message)
    engine.runAndWait()


def speak_warning(message):
    threading.Thread(target=_speak, args=(message,), daemon=True).start()