import os
import uuid
import logging
import math
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# ====================================
# Configuración de Faster-Whisper
# ====================================
from faster_whisper import WhisperModel
import torch

# Configuración del modelo
model_size = "tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "int8"
model = WhisperModel(model_size, device=device, compute_type=compute_type)

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Modelo Whisper cargado: {model_size}, dispositivo: {device}, tipo: {compute_type}")

# Transcripción
def transcribe_audio_file(file_path):
    try:
        logger.info(f"Iniciando transcripción del archivo: {file_path}")
        segments, _ = model.transcribe(file_path)
        texts = [seg.text for seg in segments if seg.text.strip()]
        logger.info(f"Segmentos encontrados: {len(texts)}")
        return " ".join(texts)
    except Exception as e:
        logger.error(f"Error al transcribir el archivo: {e}")
        return ""

# ====================================
# Configuración de Flask
# ====================================
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "whisper-secret")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'mp4', 'webm'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ====================================
# Rutas
# ====================================
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
        return jsonify({'error': 'No se subió ningún archivo'}), 400

    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': f'Tipo de archivo no soportado. Tipos válidos: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    transcription = ""
    file_path = None
    try:
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        logger.info(f"Archivo guardado: {file_path}")

        transcription = transcribe_audio_file(file_path)
        logger.info(f"Resultado de la transcripción: {transcription[:100]}...")

    except Exception as e:
        logger.error(f"Error en transcripción: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Archivo eliminado: {file_path}")
            except Exception as e:
                logger.warning(f"No se pudo eliminar el archivo: {e}")

    # Mostrar HTML si viene del navegador
    if request.content_type.startswith('multipart/form-data'):
        return f'''
        <h2>Transcripción:</h2>
        <pre>{transcription or "⚠️ No se pudo generar ninguna transcripción."}</pre>
        <a href="/">← Volver</a>
        '''

    # Si vino por API (JSON)
    return jsonify({'transcription': transcription})

@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    return transcribe()

@app.route('/keepalive', methods=['POST'])
def keepalive():
    data = request.json or {}
    valor = data.get("valor", 1000)
    texto = data.get("texto", "default")
    activo = data.get("activo", False)

    logger.info(f"Keepalive recibido: valor={valor}, texto='{texto}', activo={activo}")

    result = 0
    for i in range(1, valor):
        result += math.sqrt(i) * math.sin(i)

    return f'Ping OK | Resultado: {result:.2f}', 200

# ====================================
# Ejecución
# ====================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
