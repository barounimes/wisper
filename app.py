import os
import uuid
import logging
import math
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from audio_processor import process_audio
from transcriber import transcribe_audio_file

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "whisper-transcription-secret")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'mp4', 'webm'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not supported. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        original_filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{original_filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        logger.info(f"File saved at {file_path}")
        
        processed_files = process_audio(file_path)
        logger.info(f"Processed into {len(processed_files)} chunks")
        
        results = []
        for chunk in processed_files:
            results.append(transcribe_audio_file(chunk))
            if chunk != file_path:
                try: os.remove(chunk)
                except: pass
        
        try: os.remove(file_path)
        except: pass
        
        return jsonify({'transcription': ' '.join(results)})
    
    except Exception as e:
        logger.error(str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    return transcribe()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
