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

# Criar pasta de saída se não existir
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def allowed_file(filename):
    """Verifica se o arquivo tem uma extensão permitida"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image_complete(input_bgr, image_label):
    """
    Aplica processamento completo de imagem usando apenas OpenCV:
    1. Conversão para escala de cinza
    2. Suavização Gaussiana
    3. Suavização por mediana
    4. Limiarização por Otsu
    5. Detecção de bordas (Canny)
    6. Detecção de contornos
    7. Análise de histograma e equalização
    8. Análise de textura com filtros de Gabor
    9. Análise de gradientes (Sobel)
    
    Todas as etapas são salvas na pasta OUTPUT_DIR
    """
    try:
        output_paths = []
        processing_results = {}
        
        # Criar pasta de saída se não existir
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Salvar imagem original
        original_path = os.path.join(OUTPUT_DIR, f"{image_label}_original.png")
        cv2.imwrite(original_path, input_bgr)
        output_paths.append(original_path)
        
        # 1. Conversão para escala de cinza
        gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
        gray_path = os.path.join(OUTPUT_DIR, f"{image_label}_gray.png")
        cv2.imwrite(gray_path, gray)
        output_paths.append(gray_path)
        
        # 2. Suavização Gaussiana
        gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
        gaussian_path = os.path.join(OUTPUT_DIR, f"{image_label}_gaussian_blur.png")
        cv2.imwrite(gaussian_path, gaussian)
        output_paths.append(gaussian_path)
        
        # 3. Suavização por mediana
        median = cv2.medianBlur(gray, 5)
        median_path = os.path.join(OUTPUT_DIR, f"{image_label}_median_blur.png")
        cv2.imwrite(median_path, median)
        output_paths.append(median_path)
        
        # 4. Limiarização por Otsu
        _, otsu_mask = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_path = os.path.join(OUTPUT_DIR, f"{image_label}_otsu_mask.png")
        cv2.imwrite(otsu_path, otsu_mask)
        output_paths.append(otsu_path)
        
        # 5. Detecção de bordas (Canny)
        edges = cv2.Canny(gaussian, 100, 200)
        edges_path = os.path.join(OUTPUT_DIR, f"{image_label}_canny_edges.png")
        cv2.imwrite(edges_path, edges)
        output_paths.append(edges_path)
        
        # 6. Detecção de contornos
        contours, _ = cv2.findContours(otsu_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_overlay = input_bgr.copy()
        cv2.drawContours(contour_overlay, contours, -1, (0, 255, 0), 2)
        contours_path = os.path.join(OUTPUT_DIR, f"{image_label}_contours.png")
        cv2.imwrite(contours_path, contour_overlay)
        output_paths.append(contours_path)
        
        # Calcular características dos contornos
        shape_features = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
            shape_features.append({
                "area": float(area),
                "perimeter": float(perimeter),
                "bounding_box": (int(x), int(y), int(w), int(h)),
                "circularity": float(circularity),
            })
        
        processing_results['contours_count'] = len(contours)
        processing_results['shape_features'] = shape_features
        
        # 7. Histograma da imagem em escala de cinza
        try:
            # Calcular histograma da imagem
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_normalized = hist.flatten() / hist.sum()  # Normalizar
            processing_results['gray_histogram'] = hist_normalized[:50].tolist()  # Primeiros 50 bins
            
            # Aplicar equalização de histograma e salvar
            equalized = cv2.equalizeHist(gray)
            equalized_path = os.path.join(OUTPUT_DIR, f"{image_label}_histogram_equalized.png")
            cv2.imwrite(equalized_path, equalized)
            output_paths.append(equalized_path)
            
        except Exception as hist_error:
            print(f"Erro no processamento de histograma: {str(hist_error)}")
            processing_results['histogram_error'] = str(hist_error)
        
        # 8. Análise de textura com filtros de Gabor
        try:
            # Aplicar filtros de Gabor para análise de textura
            gabor_responses = []
            gabor_images = []
            
            # Diferentes orientações para os filtros de Gabor
            angles = [0, 45, 90, 135]
            for angle in angles:
                kernel = cv2.getGaborKernel((21, 21), 8, np.radians(angle), 2*np.pi*0.5, 0.5, 0, cv2.CV_32F)
                gabor_response = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                gabor_responses.append(gabor_response.mean())
                gabor_images.append(gabor_response)
            
            # Salvar uma combinação dos filtros de Gabor
            gabor_combined = np.mean(gabor_images, axis=0).astype(np.uint8)
            gabor_path = os.path.join(OUTPUT_DIR, f"{image_label}_gabor_texture.png")
            cv2.imwrite(gabor_path, gabor_combined)
            output_paths.append(gabor_path)
            
            processing_results['gabor_responses'] = gabor_responses
            
        except Exception as gabor_error:
            print(f"Erro no processamento de textura: {str(gabor_error)}")
            processing_results['gabor_error'] = str(gabor_error)
        
        # 9. Gradientes e detecção de características
        try:
            # Calcular gradientes nas direções X e Y
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Magnitude do gradiente
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            magnitude_norm = (magnitude * 255 / magnitude.max()).astype(np.uint8)
            
            # Direção do gradiente
            direction = np.arctan2(grad_y, grad_x)
            direction_norm = ((direction + np.pi) * 255 / (2 * np.pi)).astype(np.uint8)
            
            # Salvar imagens de gradiente
            grad_mag_path = os.path.join(OUTPUT_DIR, f"{image_label}_gradient_magnitude.png")
            cv2.imwrite(grad_mag_path, magnitude_norm)
            output_paths.append(grad_mag_path)
            
            grad_dir_path = os.path.join(OUTPUT_DIR, f"{image_label}_gradient_direction.png")
            cv2.imwrite(grad_dir_path, direction_norm)
            output_paths.append(grad_dir_path)
            
            processing_results['gradient_stats'] = {
                'mean_magnitude': float(magnitude.mean()),
                'max_magnitude': float(magnitude.max()),
                'std_magnitude': float(magnitude.std())
            }
            
        except Exception as grad_error:
            print(f"Erro no processamento de gradientes: {str(grad_error)}")
            processing_results['gradient_error'] = str(grad_error)
        
        return output_paths, processing_results
    
    except Exception as e:
        raise Exception(f"Erro no processamento da imagem: {str(e)}")

def process_image_to_grayscale(input_bgr, image_label):
    """
    Função mantida para compatibilidade - apenas converte para escala de cinza
    """
    try:
        gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(Path(OUTPUT_DIR) / f"{image_label}_gray.png"), gray)
        output_path = str(Path(OUTPUT_DIR) / f"{image_label}_gray.png")
        return gray, output_path
    except Exception as e:
        raise Exception(f"Erro no processamento da imagem: {str(e)}")

@app.route('/', methods=['GET'])
def home():
    """Rota de teste"""
    return jsonify({
        'message': 'API de Processamento de Imagens - PDI Reconhecimento de Produtos',
        'version': '1.0.0',
        'endpoints': {
            '/process-image': 'POST - Converte imagem para escala de cinza (simples)',
            '/process-image-complete': 'POST - Processamento completo de imagem (pipeline completo)',
            '/health': 'GET - Status da API',
            '/hello': 'GET - Retorna Hello World'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Verificação de saúde da API"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/hello', methods=['GET'])
def hello_world():
    """Endpoint simples que retorna Hello World"""
    return jsonify({
        'message': 'Hello World!',
        'status': 'success'
    })

@app.route('/process-image', methods=['POST'])
def process_image_route():
    """
    Endpoint que converte imagem para escala de cinza e salva
    
    Parâmetros:
    - file: arquivo de imagem
    """
    try:
        # Verificar se foi enviado um arquivo
        if 'file' not in request.files:
            return jsonify({
                'error': 'Nenhum arquivo foi enviado',
                'code': 'NO_FILE'
            }), 400
        
        file = request.files['file']
        
        # Verificar se o arquivo não está vazio
        if file.filename == '':
            return jsonify({
                'error': 'Nome do arquivo vazio',
                'code': 'EMPTY_FILENAME'
            }), 400
        
        # Verificar tipo de arquivo
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Tipo de arquivo não permitido. Tipos aceitos: {", ".join(ALLOWED_EXTENSIONS)}',
                'code': 'INVALID_FILE_TYPE'
            }), 400
        
        # Processar a imagem
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

@app.route('/process-image-complete', methods=['POST'])
def process_image_complete_route():
    """
    Endpoint que aplica processamento completo de imagem usando apenas OpenCV:
    - Conversão para escala de cinza
    - Suavização (Gaussiana e Mediana)
    - Limiarização (Otsu)
    - Detecção de bordas (Canny)
    - Detecção de contornos
    - Análise de histograma e equalização
    - Análise de textura com filtros de Gabor
    - Análise de gradientes (Sobel)
    
    Todas as etapas são salvas na pasta processed_images/
    
    Parâmetros:
    - file: arquivo de imagem
    """
    try:
        # Verificar se foi enviado um arquivo
        if 'file' not in request.files:
            return jsonify({
                'error': 'Nenhum arquivo foi enviado',
                'code': 'NO_FILE'
            }), 400
        
        file = request.files['file']
        
        # Verificar se o arquivo não está vazio
        if file.filename == '':
            return jsonify({
                'error': 'Nome do arquivo vazio',
                'code': 'EMPTY_FILENAME'
            }), 400
        
        # Verificar tipo de arquivo
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Tipo de arquivo não permitido. Tipos aceitos: {", ".join(ALLOWED_EXTENSIONS)}',
                'code': 'INVALID_FILE_TYPE'
            }), 400
        
        # Processar a imagem
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
            
            # Processar imagem completamente (pipeline completo)
            output_paths, processing_results = process_image_complete(input_bgr, image_label)
            
            # Preparar resposta
            response_data = {
                'success': True,
                'message': 'Imagem processada completamente com pipeline PDI e salva com sucesso',
                'original_filename': filename,
                'image_label': image_label,
                'output_paths': output_paths,
                'processing_steps': [
                    'original', 'grayscale', 'gaussian_blur', 'median_blur', 
                    'otsu_mask', 'canny_edges', 'contours', 'histogram_equalized',
                    'gabor_texture', 'gradient_magnitude', 'gradient_direction'
                ],
                'original_size': {
                    'height': input_bgr.shape[0],
                    'width': input_bgr.shape[1],
                    'channels': input_bgr.shape[2]
                },
                'processing_results': processing_results,
                'total_files_generated': len(output_paths),
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
    """Manipula erro de arquivo muito grande"""
    return jsonify({
        'error': f'Arquivo muito grande. Tamanho máximo: {MAX_FILE_SIZE // (1024*1024)}MB',
        'code': 'FILE_TOO_LARGE'
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Manipula erro 404"""
    return jsonify({
        'error': 'Endpoint não encontrado',
        'code': 'NOT_FOUND'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Manipula erro interno do servidor"""
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