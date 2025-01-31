import os
import tempfile
from typing import List
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from moviepy import VideoFileClip
import speech_recognition as sr

class VideoTranscriptionLoader(BaseLoader):
    """Load video files and transcribe their audio content."""

    def __init__(self, file_path: str):
        """Initialize with path to video file."""
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load and transcribe the video file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        # Extract audio from video
        try:
            video = VideoFileClip(self.file_path)
            audio = video.audio

            # Save audio to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                audio.write_audiofile(temp_audio.name, codec='pcm_s16le')
                temp_audio_path = temp_audio.name

            # Clean up video and audio objects
            audio.close()
            video.close()

            # Initialize speech recognition
            recognizer = sr.Recognizer()
            print("Recognizing audio...")

            # Load audio file and transcribe
            with sr.AudioFile(temp_audio_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)

            # Clean up temporary file
            os.unlink(temp_audio_path)

            # Create metadata
            metadata = {
                "source": self.file_path,
                "file_type": "video",
                "transcription_method": "google_speech_recognition"
            }

            return [Document(page_content=text, metadata=metadata)]

        except Exception as e:
            raise Exception(f"Error processing video file: {str(e)}")
        