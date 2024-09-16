import google.generativeai as genai
from dotenv import load_dotenv
import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile

# Load API key from.env file
load_dotenv()
api_key = os.getenv("API_KEY")

# Set up Gemini model
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

def speech_to_text_gemini(audio_file):
    prompt = "Transcribe the speech."
    response = model.generate_content([prompt, audio_file])
    return response.text

def upload_audio_file(file_path):
    audio_file = genai.upload_file(path=file_path)
    return audio_file

def record_audio():
    recording = sd.rec(int(5 * 44100), samplerate=44100, channels=1)
    sd.wait()
    return recording

def save_audio_file(recording):
    wavfile.write("output.wav", 44100, recording.astype(np.int16))

def speech_to_text():
    print("Recording audio...")
    recording = record_audio()
    print("Saving audio file...")
    save_audio_file(recording)
    print("Uploading audio file...")
    uploaded_audio_file = upload_audio_file("output.wav")
    print("Transcribing speech...")
    transcribed_text = speech_to_text_gemini(uploaded_audio_file)
    print("Transcribed text:")
    print(transcribed_text)
    return transcribed_text

def main():
    input_text = speech_to_text()
    print("Input Text:", input_text)

if __name__ == "__main__":
    main()
