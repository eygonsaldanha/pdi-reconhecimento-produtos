import datetime

import cv2
import numpy as np
from flask import jsonify

from common import allowed_file, ALLOWED_EXTENSIONS
from common import generate_hash
from db_common import insert_data, select_data
from io_minio import upload_img


def process_image_exec(request, knn_default):
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

        not_is_this_products = []
        id_data = None

        if 'not-is' in request.form:
            if 'id_data' not in request.form:
                return jsonify({'error': 'Informe o identificador do dado', 'code': 'NO_ID_DATA'}), 400
            id_data = int(request.form['id_data'])

            not_is_this_products = request.form['not-is'].split(',')
            not_is_this_products = [int(x) for x in not_is_this_products if x.strip().isdigit()]
        else:
            path_data = generate_hash()
            id_data = select_data('SELECT COUNT(*) FROM DATA')['count'].iloc[0] + 1
            insert_data('data', [{'id_data': id_data,  #
                                  'path_data': path_data,  #
                                  'tp_data': 'IMG',  #
                                  'dt_inclusion': datetime.datetime.now()  #
                                  }])
            upload_img(image=img, content_type=file.content_type, key=path_data)

        knn_result = knn_default.knn_process_image(img, not_is_this_products)

        df_product_result = select_data(f"""
        SELECT p.* FROM data d
        JOIN product_data pd ON pd.id_data = d.id_data
        JOIN product p ON p.id_product = pd.id_product
        WHERE d.path_data = '{knn_result}'
        """)

        return jsonify({'id_data': int(id_data),  #
                        'id_product': int(df_product_result['id_product'].iloc[0]),  #
                        'nm_product': df_product_result['nm_product'].iloc[0],  #
                        'vl_product': float(df_product_result['vl_product'].iloc[0])  #
                        }), 200

    except Exception as e:
        return jsonify({'error': f'Erro interno do servidor: {str(e)}', 'code': 'INTERNAL_ERROR'}), 500


def process_image_confirm_exec(request):
    try:
        json = request.json
        if len(select_data(f"SELECT * FROM product_data WHERE id_data = {json['id_data']}")) == 0:
            insert_data('product_data', [{'id_product': json['id_product'], 'id_data': json['id_data']}])
        return jsonify(json), 200
    except Exception as e:
        return jsonify({'error': f'Erro interno do servidor: {str(e)}', 'code': 'INTERNAL_ERROR'}), 500
