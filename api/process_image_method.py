from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os
from io import BytesIO
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import base64
import cv2
import io
from IPython.display import HTML, display

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def __img_to_html__(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.png', img_rgb)
    base64_img = base64.b64encode(buffer).decode('utf-8')
    return f'<img src="data:image/png;base64,{base64_img}" width="80" height="80" />'

def process_image(request):
    try:      
        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo foi enviado', 'code': 'NO_FILE'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Nome do arquivo vazio', 'code': 'EMPTY_FILENAME'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'Tipo de arquivo não permitido. Tipos aceitos: {", ".join(ALLOWED_EXTENSIONS)}', 'code': 'INVALID_FILE_TYPE'}), 400
        
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Não foi possível ler a imagem', 'code': 'INVALID_IMAGE'}), 400
        
        print(img)
        
        return jsonify({'message':'Ok'}), 200
    
    except Exception as e:
        return jsonify({'error': f'Erro interno do servidor: {str(e)}', 'code': 'INTERNAL_ERROR'}), 500