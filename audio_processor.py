import os
import logging
from pydub import AudioSegment
from pydub.utils import which

logger = logging.getLogger(__name__)

# Configure ffmpeg path for pydub
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffmpeg = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

def process_audio(file_path):
    """
    Process audio file by splitting into chunks if necessary.
    Returns a list of file paths to process.
    """
    try:
        # Get file size in MB
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"Processing file: {file_path}, size: {file_size_mb:.2f} MB")
        
        # If file is smaller than 25MB, process as is
        if file_size_mb < 25:
            logger.info("File is small enough, processing without splitting")
            return [file_path]
        
        # Load audio file
        logger.info("Loading audio file for splitting...")
        audio = AudioSegment.from_file(file_path)
        
        # Calculate chunk duration (aim for ~20MB chunks)
        # Estimate based on file size and duration
        duration_ms = len(audio)
        target_chunk_size_mb = 20
        estimated_chunks = max(1, int(file_size_mb / target_chunk_size_mb))
        chunk_duration_ms = duration_ms // estimated_chunks
        
        # Ensure minimum chunk duration of 30 seconds
        min_chunk_duration_ms = 30 * 1000
        chunk_duration_ms = max(chunk_duration_ms, min_chunk_duration_ms)
        
        logger.info(f"Splitting into chunks of {chunk_duration_ms/1000:.1f} seconds")
        
        chunk_files = []
        chunk_number = 0
        
        for start_ms in range(0, duration_ms, chunk_duration_ms):
            end_ms = min(start_ms + chunk_duration_ms, duration_ms)
            chunk = audio[start_ms:end_ms]
            
            # Generate chunk filename
            base_name = os.path.splitext(file_path)[0]
            chunk_filename = f"{base_name}_chunk_{chunk_number:03d}.wav"
            
            # Export chunk as WAV for better compatibility with whisper
            chunk.export(chunk_filename, format="wav")
            chunk_files.append(chunk_filename)
            
            logger.info(f"Created chunk {chunk_number}: {chunk_filename}")
            chunk_number += 1
        
        logger.info(f"Successfully split into {len(chunk_files)} chunks")
        return chunk_files
        
    except Exception as e:
        logger.error(f"Error processing audio file {file_path}: {str(e)}")
        # If processing fails, return original file
        return [file_path]

def convert_to_wav(file_path):
    """
    Convert audio/video file to WAV format for better compatibility.
    """
    try:
        logger.info(f"Converting {file_path} to WAV format")
        
        # Load the file
        audio = AudioSegment.from_file(file_path)
        
        # Generate output filename
        base_name = os.path.splitext(file_path)[0]
        wav_filename = f"{base_name}_converted.wav"
        
        # Export as WAV
        audio.export(wav_filename, format="wav")
        
        logger.info(f"Successfully converted to {wav_filename}")
        return wav_filename
        
    except Exception as e:
        logger.error(f"Error converting {file_path} to WAV: {str(e)}")
        # Return original file if conversion fails
        return file_path

def get_audio_duration(file_path):
    """
    Get the duration of an audio file in seconds.
    """
    try:
        audio = AudioSegment.from_file(file_path)
        duration_seconds = len(audio) / 1000.0
        return duration_seconds
    except Exception as e:
        logger.error(f"Error getting duration for {file_path}: {str(e)}")
        return 0
