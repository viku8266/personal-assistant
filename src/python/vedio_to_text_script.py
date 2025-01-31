from moviepy import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
import speech_recognition as sr
import os
from transcript import AudioTranscriber
from pydub import AudioSegment

def create_audio_file(video_path):
    file_name = os.path.splitext(os.path.basename(video_path))[0]
    # Load the video file
    video = VideoFileClip(video_path)
    
    # Extract audio from the video
    audio = video.audio
    
    # Create a directory for audio files if it doesn't exist
    audio_dir = 'audio'
    os.makedirs(audio_dir, exist_ok=True)
    audio_file_name = os.path.join(audio_dir, file_name + '.wav')
    audio.write_audiofile(audio_file_name, codec='pcm_s16le')
    return audio_file_name

def transcribe_audio_chunks(audio_file_path):
    audio = AudioSegment.from_wav(audio_file_path)
    chunk_length_ms = 300000  # 5 minutes in milliseconds
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    
    # Create a directory for transcripts if it doesn't exist
    transcripts_dir = 'transcripts'
    os.makedirs(transcripts_dir, exist_ok=True)

    audio_chunck_dir = "sub_audios"
    os.makedirs(audio_chunck_dir, exist_ok=True)

    # Generate the transcript file name based on the audio file name
    file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    transcript_file_name = os.path.join(transcripts_dir, file_name + '.txt')

    # Initialize a variable to accumulate the transcript
    complete_transcript = ''

    chunk_number = 0
    for chunk in chunks:
        chunk_file_name = f"{file_name}-{chunk_number}.wav"
        chunk.export(os.path.join(audio_chunck_dir, chunk_file_name), format="wav")
        text = AudioTranscriber().transcribe_audio(os.path.join(audio_chunck_dir, chunk_file_name))
        print(text)
        complete_transcript += text + '\n'
        chunk_number += 1

    # Save the complete transcript to a file
    with open(transcript_file_name, 'w') as file:
        file.write(complete_transcript)
    print(f'Complete transcript saved to {transcript_file_name}')

def get_video_files(directory):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    return video_files

# Example usage
def transcript_all_videos(directory):
    video_files = get_video_files(directory)
    for video_file in video_files:
        audio_file_name = create_audio_file(video_file)
        transcribe_audio_chunks(audio_file_name)

video_directory = '/Users/vikasvashistha/github/personal-assistant/documents'
transcript_all_videos(video_directory)