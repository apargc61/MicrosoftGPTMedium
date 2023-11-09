import speech_recognition as sr

# Initialize the recognizer
recognizer = sr.Recognizer()

# Capture audio from the microphone
with sr.Microphone() as source:
    print("Say something:")
    audio = recognizer.listen(source)

# Recognize the speech using Google Web Speech API
try:
    text = recognizer.recognize_google(audio)
    print(f"You said: {text}")
except sr.UnknownValueError:
    print("Could not understand the audio")
except sr.RequestError as e:
    print(f"Request error: {e}")
