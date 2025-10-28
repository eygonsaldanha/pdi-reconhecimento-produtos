import cv2
import numpy as np
from flask import jsonify
from werkzeug.datastructures import FileStorage

from common import allowed_file, ALLOWED_EXTENSIONS
from knn_process_image import knn_process_image
from io_minio import upload_img


def process_image_exec(request):
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo foi enviado', 'code': 'NO_FILE'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'Nome do arquivo vazio', 'code': 'EMPTY_FILENAME'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': f'Tipo de arquivo não permitido. Tipos aceitos: {", ".join(ALLOWED_EXTENSIONS)}',
                            'code': 'INVALID_FILE_TYPE'}), 400

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Não foi possível ler a imagem', 'code': 'INVALID_IMAGE'}), 400

        upload_img(image=img, content_type=file.content_type)
        knn_result = knn_process_image(img)
        return jsonify({'path_image_result': knn_result}), 200

    except Exception as e:
        return jsonify({'error': f'Erro interno do servidor: {str(e)}', 'code': 'INTERNAL_ERROR'}), 500
