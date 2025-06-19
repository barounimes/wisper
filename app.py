import os
import uuid
import logging
import math
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# ====================================
# Configuración de Faster-Whisper
# ====================================
from faster_whisper import WhisperModel
import torch

model_size = "base"
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

model = WhisperModel(model_size, device=device, compute_type=compute_type)

def transcribe_audio_file(file_path):
    segments, _ = model.transcribe(file_path)
    return " ".join([seg.text for seg in segments])

# ====================================
# Flask App
# ====================================

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "whisper-secret")

# Configuración de archivos
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'mp4', 'webm'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_audio(file_path):
    # Aquí podrías dividir o convertir el audio si deseas.
    return [file_path]

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head><title>Transcripción</title></head>
    <body>
        <h1>Sube un archivo de audio para transcribir</h1>
        <form method="POST" action="/transcribe" enctype="multipart/form-data">
            <input type="file" name="audio_file" required>
            <button type="submit">Transcribir</button>
        </form>
    </body>
    </html>
    '''

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': f'Tipo de archivo no soportado. Tipos válidos: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    try:
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        logger.info(f"Archivo guardado: {file_path}")

        chunks = process_audio(file_path)
        transcription = []

        for chunk in chunks:
            result = transcribe_audio_file(chunk)
            transcription.append(result)
            if chunk != file_path:
                try: os.remove(chunk)
                except: pass

        try: os.remove(file_path)
        except: pass

        return jsonify({'transcription': ' '.join(transcription)})
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    return transcribe()

@app.route('/keepalive', methods=['POST'])
def keepalive():
    data = request.json or {}
    valor = data.get("valor", 1000)
    texto = data.get("texto", "default")
    activo = data.get("activo", False)

    print(f"Keepalive recibido: valor={valor}, texto='{texto}', activo={activo}")

    result = 0
    for i in range(1, valor):
        result += math.sqrt(i) * math.sin(i)

    return f'Ping OK | Resultado: {result:.2f}', 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
