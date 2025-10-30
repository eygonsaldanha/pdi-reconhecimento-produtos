from flask import Flask, request, jsonify
from knn_process_image import KNN
# from flask_cors import CORS
from process_image_method import process_image_exec,process_image_confirm_exec

app = Flask(__name__)
# CORS(app)

MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
knn_default = KNN()

@app.route('/process-image', methods=['POST'])
def process_image_route():
    return process_image_exec(request,knn_default)

@app.route('/process-image/confirm', methods=['POST'])
def process_image_confirm_route():
    return process_image_confirm_exec(request)

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': f'Arquivo muito grande. Tamanho máximo: {MAX_FILE_SIZE // (1024 * 1024)}MB',
                    'code': 'FILE_TOO_LARGE'}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint não encontrado', 'code': 'NOT_FOUND'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Erro interno do servidor', 'code': 'INTERNAL_ERROR'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
