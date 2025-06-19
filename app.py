import os
import uuid
import logging
import math
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from utils.audio_processor import process_audio
from utils.transcriber import transcribe_audio_file

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "whisper-transcription-secret")

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'mp4', 'webm'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # Check if file is in the request
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['audio_file']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file has valid extension
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not supported. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Generate a unique filename
        original_filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{original_filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(file_path)
        logger.info(f"File saved at {file_path}")
        
        # Process audio file (split, compress if needed)
        processed_files = process_audio(file_path)
        logger.info(f"Processed into {len(processed_files)} chunks")
        
        # Transcribe all chunks
        transcription_results = []
        for chunk_file in processed_files:
            result = transcribe_audio_file(chunk_file)
            transcription_results.append(result)
            
            # Clean up processed chunks
            if chunk_file != file_path:  # Don't delete the original if it's the only chunk
                try:
                    os.remove(chunk_file)
                except Exception as e:
                    logger.warning(f"Could not delete chunk file {chunk_file}: {e}")
        
        # Clean up original file
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"Could not delete original file {file_path}: {e}")
        
        # Combine all transcriptions
        full_transcription = ' '.join(transcription_results)
        
        return jsonify({'transcription': full_transcription})
    
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        return jsonify({'error': f'Error during transcription: {str(e)}'}), 500

@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    # This endpoint serves the same function as the web interface but is designed for API access
    return transcribe()

@app.route('/keepalive', methods=['POST'])
def keepalive():
    data = request.json or {}
    valor = data.get("valor", 1000)
    texto = data.get("texto", "default")
    activo = data.get("activo", False)

    print(f"Keepalive recibido con valor={valor}, texto='{texto}', activo={activo}")

    # Trabajo artificial más realista con valor dinámico
    result = 0
    for i in range(1, valor):
        result += math.sqrt(i) * math.sin(i)

    print("Cálculo dinámico completado.")
    return f'Ping OK | Cálculo con valor {valor} -> Resultado: {result:.2f}', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
