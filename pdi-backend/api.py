from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os
from datetime import datetime
from pathlib import Path

app = Flask(__name__)

# Configurações
OUTPUT_DIR = 'processed_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image_to_grayscale(input_bgr, image_label):
    try:
        gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(Path(OUTPUT_DIR) / f"{image_label}_gray.png"), gray)
        output_path = str(Path(OUTPUT_DIR) / f"{image_label}_gray.png")
        return gray, output_path
    except Exception as e:
        raise Exception(f"Erro no processamento da imagem: {str(e)}")

@app.route('/hello', methods=['GET'])
def hello_world():
    return jsonify({
        'message': 'Hello World!',
        'status': 'success'
    })

#curl -X POST \
#  -H "Content-Type: multipart/form-data" \
#  -F "file=@/path/to/your/image.jpg" \
#  -v \
#  http://localhost:5000/process-image
@app.route('/process-image', methods=['POST'])
def process_image_route():
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'Nenhum arquivo foi enviado',
                'code': 'NO_FILE'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'Nome do arquivo vazio',
                'code': 'EMPTY_FILENAME'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Tipo de arquivo não permitido. Tipos aceitos: {", ".join(ALLOWED_EXTENSIONS)}',
                'code': 'INVALID_FILE_TYPE'
            }), 400
        
        try:
            # Ler arquivo como bytes
            file_bytes = file.read()
            
            # Converter bytes para numpy array
            nparr = np.frombuffer(file_bytes, np.uint8)
            
            # Decodificar imagem com OpenCV
            input_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if input_bgr is None:
                raise Exception("Não foi possível decodificar a imagem")
            
            # Gerar label da imagem baseado no nome do arquivo
            filename = secure_filename(file.filename)
            image_label = filename.rsplit('.', 1)[0]  # Remove extensão
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_label = f"{image_label}_{timestamp}"
            
            # Processar imagem (converter para escala de cinza e salvar)
            gray, output_path = process_image_to_grayscale(input_bgr, image_label)
            
            # Preparar resposta
            response_data = {
                'success': True,
                'message': 'Imagem convertida para escala de cinza e salva com sucesso',
                'original_filename': filename,
                'processed_filename': f"{image_label}_gray.png",
                'output_path': output_path,
                'original_size': {
                    'height': input_bgr.shape[0],
                    'width': input_bgr.shape[1],
                    'channels': input_bgr.shape[2]
                },
                'processed_size': {
                    'height': gray.shape[0],
                    'width': gray.shape[1],
                    'channels': 1
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(response_data), 200
            
        except Exception as e:
            return jsonify({
                'error': f'Erro ao processar imagem: {str(e)}',
                'code': 'PROCESSING_ERROR'
            }), 500
            
    except Exception as e:
        return jsonify({
            'error': f'Erro interno do servidor: {str(e)}',
            'code': 'INTERNAL_ERROR'
        }), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': f'Arquivo muito grande. Tamanho máximo: {MAX_FILE_SIZE // (1024*1024)}MB',
        'code': 'FILE_TOO_LARGE'
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Endpoint não encontrado',
        'code': 'NOT_FOUND'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'error': 'Erro interno do servidor',
        'code': 'INTERNAL_ERROR'
    }), 500

if __name__ == '__main__':
    print("🚀 Iniciando API de Processamento de Imagens...")
    print(f"📁 Pasta de saída: {os.path.abspath(OUTPUT_DIR)}")
    print(f"📏 Tamanho máximo de arquivo: {MAX_FILE_SIZE // (1024*1024)}MB")
    print(f"🎨 Tipos de arquivo aceitos: {', '.join(ALLOWED_EXTENSIONS)}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)