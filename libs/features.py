import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops

def extrair_caracteristicas_forma(contornos):
    caracteristicas = []
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        x, y, w, h = cv2.boundingRect(contorno)
        circularidade = 4 * np.pi * area / (perimetro ** 2) if perimetro > 0 else 0.0
        
        caracteristicas.append({
            "area": float(area),
            "perimetro": float(perimetro),
            "bounding_box": (int(x), int(y), int(w), int(h)),
            "circularidade": float(circularidade),
        })
    return caracteristicas

def extrair_lbp(imagem, P=8, R=1):
    lbp = local_binary_pattern(imagem, P=P, R=R, method="uniform")
    lbp_norm = (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-9)
    lbp_img = (lbp_norm * 255).astype("uint8")
    
    lbp_bins = np.arange(0, lbp.max() + 2)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=lbp_bins, density=True)
    
    return lbp_img, lbp_hist

def extrair_glcm(imagem, distancias=[1, 2, 3], angulos=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    glcm = graycomatrix(imagem, distances=distancias, angles=angulos, levels=256, symmetric=True, normed=True)
    
    caracteristicas = {}
    propriedades = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
    
    for prop in propriedades:
        caracteristicas[prop] = float(graycoprops(glcm, prop).mean())
    
    return caracteristicas

def extrair_hog(imagem, orientacoes=9, pixels_por_celula=(16, 16), celulas_por_bloco=(2, 2)):
    hog_vector, hog_vis = hog(
        imagem,
        orientations=orientacoes,
        pixels_per_cell=pixels_por_celula,
        cells_per_block=celulas_por_bloco,
        visualize=True,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    
    hog_vis_norm = (hog_vis - hog_vis.min()) / (hog_vis.max() - hog_vis.min() + 1e-9)
    hog_img = (hog_vis_norm * 255).astype("uint8")
    
    return hog_vector, hog_img
