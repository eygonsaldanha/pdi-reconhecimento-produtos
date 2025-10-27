from flask import Flask, request
from process_image_method import process_image

app = Flask(__name__)

MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

@app.route('/process-image', methods=['POST'])
def process_image_route():
    return process_image(request)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)