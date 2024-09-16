import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
from io import BytesIO
import tempfile
import os
import re
import time
import random
import requests
from gemini_speech_to_text import speech_to_text_gemini, upload_audio_file
from response import resp
from gtts import gTTS
from io import BytesIO

# Set page config
st.set_page_config(page_title="FlowBus Chatbot", page_icon="🚌", layout="wide")

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        audio_data = BytesIO()
        tts.write_to_fp(audio_data)
        audio_data.seek(0)
        return audio_data.read()
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        return None

# Custom CSS
st.markdown("""
<style>
.header {
        color: #1E88E5;
        font-size: 50px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }
.user-message {
        background-color: #E3F2FD;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
.bot-message {
        background-color: #BBDEFB;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
.error-message {
        background-color: #FFCDD2;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
.info-message {
        background-color: #C8E6C9;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
.recording-indicator {
        color: red;
        font-weight: bold;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        50% { opacity: 0; }
    }
.stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'last_input' not in st.session_state:
    st.session_state.last_input = ""
if 'recorded_text' not in st.session_state:
    st.session_state.recorded_text = ""

# Header
st.markdown('<div class="header">🚌 FlowBus Chatbot</div>', unsafe_allow_html=True)

# Main chat interface
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        if message['type'] == 'user':
            st.markdown(f"<div class='user-message'><strong>You:</strong> {message['text']}</div>", unsafe_allow_html=True)
        elif message['type'] == 'bot':
            st.markdown(f"<div class='bot-message'><strong>FlowBus:</strong> {message['text']}</div>", unsafe_allow_html=True)
            if 'audio' in message and message['audio']:
                st.audio(message['audio'], format='audio/mp3')
        elif message['type'] == 'error':
            st.markdown(f"<div class='error-message'>{message['text']}</div>", unsafe_allow_html=True)
        elif message['type'] == 'info':
            st.markdown(f"<div class='info-message'>{message['text']}</div>", unsafe_allow_html=True)

# Input section
col1, col2, col3 = st.columns([0.7, 0.15, 0.15])

with col1:
    text_input = st.text_input("", placeholder="Type your message here...", key="text_input")

with col2:
    send_button = st.button("Send", use_container_width=True)

with col3:
    if not st.session_state.is_recording:
        if st.button("🎤 Record", key="start_recording", help="Click to start recording"):
            st.session_state.is_recording = True
            recording = sd.rec(int(5 * 44100), samplerate=44100, channels=1)
            sd.wait()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wavfile.write(tmp.name, 44100, recording)
                audio_file = upload_audio_file(tmp.name)
                text = speech_to_text_gemini(audio_file)
                st.session_state.chat_history.append({'type': 'user', 'text': text})
                response = resp(text)
                audio_data = text_to_speech(response)
                st.session_state.chat_history.append({'type': 'bot', 'text': response, 'audio': audio_data})
            st.session_state.is_recording = False
            st.rerun()
    else:
        if st.button("🔴 Stop", key="stop_recording", help="Recording in progress. Click to stop."):
            st.session_state.is_recording = False
            st.rerun()

# Process text input
if send_button or (text_input and text_input!= st.session_state.last_input):
    st.session_state.chat_history.append({'type': 'user', 'text': text_input})
    response = resp(text_input)
    audio_data = text_to_speech(response)
    st.session_state.chat_history.append({'type': 'bot', 'text': response, 'audio': audio_data})
    st.session_state.last_input = text_input
    st.rerun()

# Clear chat history button
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# JavaScript to scroll chat container to bottom
st.markdown("""
<script>
    function scrollToBottom() {
        var chatContainer = document.querySelector('.stStreamlitApp');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }
    scrollToBottom();
    
    // Periodically check for new messages and scroll
    setInterval(scrollToBottom, 500);
</script>
""", unsafe_allow_html=True)
