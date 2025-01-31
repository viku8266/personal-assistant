import os
import whisper
class AudioTranscriber:
  def __init__(self, model_size="small"):
    """
    Initializes the transcriber with a specific model size.
    """
    self.model = whisper.load_model(model_size)
  def transcribe_audio(self, audio_path, language="English"):
    """
    Transcribes the audio file specified by audio_path and returns the transcription text.
    """
    result = self.model.transcribe(audio_path, language=language)
    return result["text"]
    