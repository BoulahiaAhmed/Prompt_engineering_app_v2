import streamlit as st
import moviepy.editor as mp
from pydub import AudioSegment
import os
from groq import Groq
from dotenv import load_dotenv


# Load environment variables from a .env file
load_dotenv()

#GROQ_API_KEY = os.getenv('groq_api')
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# Initialize the Groq client
client = Groq(api_key=GROQ_API_KEY)


def extract_audio_from_video(video_path, output_audio_path):
    """Extracts audio from the video file and saves it as MP3."""
    # Load the video file
    video = mp.VideoFileClip(video_path)
    # Create a temporary path for the WAV file
    audio_path = output_audio_path.replace(".mp3", ".wav")
    # Extract and save the audio as a WAV file
    video.audio.write_audiofile(audio_path)
    return audio_path


def transcribe_audio_with_whisper(audio_path):
    """Transcribes the audio using the specified Whisper model."""
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(audio_path, audio_file.read()),
            model="whisper-large-v3",
            prompt="Specify context or spelling",
            response_format="json",
            temperature=0.0
        )
        return transcription.text
