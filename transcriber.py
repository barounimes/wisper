from faster_whisper import WhisperModel
import torch

model_size = "base"
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

model = WhisperModel(model_size, device=device, compute_type=compute_type)

def transcribe_audio_file(file_path):
    segments, _ = model.transcribe(file_path)
    return " ".join([seg.text for seg in segments])
