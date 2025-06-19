import os
import logging
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Initialize the Whisper model
# Using base model for balance between speed and accuracy
# You can change to "small", "medium", "large-v2", or "large-v3" for different quality/speed tradeoffs
MODEL_SIZE = "base"
model = None

def get_whisper_model():
    """
    Get or initialize the Whisper model.
    """
    global model
    if model is None:
        try:
            logger.info(f"Loading Whisper model: {MODEL_SIZE}")
            # Use CPU for compatibility, change device="cuda" if GPU is available
            model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            raise
    return model

def transcribe_audio_file(file_path):
    """
    Transcribe an audio file using faster-whisper.
    Returns the transcribed text.
    """
    try:
        logger.info(f"Starting transcription for: {file_path}")
        
        # Get the model
        whisper_model = get_whisper_model()
        
        # Transcribe the audio
        segments, info = whisper_model.transcribe(
            file_path,
            beam_size=5,
            language=None,  # Auto-detect language
            condition_on_previous_text=False,
            vad_filter=True,  # Voice Activity Detection to filter out silence
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Collect all text segments
        transcription_text = ""
        for segment in segments:
            transcription_text += segment.text + " "
        
        # Clean up the text
        transcription_text = transcription_text.strip()
        
        logger.info(f"Transcription completed for {file_path}")
        logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        logger.debug(f"Transcription preview: {transcription_text[:100]}...")
        
        return transcription_text
        
    except Exception as e:
        logger.error(f"Error transcribing {file_path}: {str(e)}")
        raise Exception(f"Transcription failed: {str(e)}")

def transcribe_with_timestamps(file_path):
    """
    Transcribe an audio file and return text with timestamps.
    Returns a list of segments with start/end times and text.
    """
    try:
        logger.info(f"Starting transcription with timestamps for: {file_path}")
        
        # Get the model
        whisper_model = get_whisper_model()
        
        # Transcribe the audio
        segments, info = whisper_model.transcribe(
            file_path,
            beam_size=5,
            language=None,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Collect segments with timestamps
        result_segments = []
        for segment in segments:
            result_segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip()
            })
        
        logger.info(f"Transcription with timestamps completed for {file_path}")
        logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
        return {
            'segments': result_segments,
            'language': info.language,
            'language_probability': info.language_probability
        }
        
    except Exception as e:
        logger.error(f"Error transcribing with timestamps {file_path}: {str(e)}")
        raise Exception(f"Transcription with timestamps failed: {str(e)}")

def get_supported_languages():
    """
    Get list of supported languages by Whisper.
    """
    # Common languages supported by Whisper
    return [
        'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh',
        'ar', 'tr', 'pl', 'nl', 'sv', 'da', 'no', 'fi', 'hu', 'cs',
        'sk', 'uk', 'bg', 'hr', 'sl', 'et', 'lv', 'lt', 'ro', 'el',
        'he', 'hi', 'th', 'vi', 'id', 'ms', 'tl', 'sw', 'mt', 'cy',
        'is', 'eu', 'ca', 'gl', 'ast', 'gn', 'qu', 'ay', 'nah'
    ]
