import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from db_common import select_data
from io_minio import get_single_object_img
from pathlib import Path
import sys

# Habilita import dos módulos em libs/
ROOT_DIR = Path(__file__).resolve().parents[1]
LIBS_DIR = ROOT_DIR / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.append(str(LIBS_DIR))

from texture import extrair_lbp, extrair_glcm


class KNN:
    def __init__(self):
        self.df_database_images = None
        self.__load_df_database_images__()
        self.knn = None
        # Novo: cache para features persistidas (IMAGE_FEATURES)
        self.df_features = None
        self.knn_features = None

    # TODO (LEGADO) Deve ser removido / trocado por features persistidas ou pipeline de features compactas
    def process_image_pdi_concat(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Histograma em tons de cinza
        hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_gray = cv2.normalize(hist_gray, hist_gray).flatten()

        # Histogramas RGB
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
        hist_b = cv2.normalize(hist_b, hist_b).flatten()
        hist_g = cv2.normalize(hist_g, hist_g).flatten()
        hist_r = cv2.normalize(hist_r, hist_r).flatten()
        hist_rgb = np.concatenate([hist_b, hist_g, hist_r])

        # LBP: média e desvio do histograma LBP (barato e informativo)
        _lbp_img, lbp_hist = extrair_lbp(gray, p=8, r=1)
        lbp_mean = float(np.mean(lbp_hist))
        lbp_std = float(np.std(lbp_hist))

        # GLCM: propriedades leves (usar ângulos em radianos)
        glcm_features = extrair_glcm(gray, [1, 2], [0, np.pi/4])
        glcm_vec = np.array([
            glcm_features.get('contrast', 0.0),
            glcm_features.get('homogeneity', 0.0),
            glcm_features.get('energy', 0.0)
        ], dtype=float)

        # Hu Moments: 7 invariantes de forma (log-transform para estabilizar)
        m = cv2.moments(gray)
        hu = cv2.HuMoments(m).flatten()
        hu = np.sign(hu) * np.log1p(np.abs(hu))

        # Sobel energy: energia dos gradientes em X e Y
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_energy = np.array([float((gx**2).mean()), float((gy**2).mean())], dtype=float)

        # Vetor final: hist_gray (256) + hist_rgb (768) + 2 (LBP) + 3 (GLCM) + 7 (Hu) + 2 (Sobel)
        return np.concatenate([
            hist_gray,
            hist_rgb,
            np.array([lbp_mean, lbp_std], dtype=float),
            glcm_vec,
            hu.astype(float),
            sobel_energy
        ])

    def __load_df_database_images__sql__(self, sql):
        df_database_images = select_data(sql)
        df_database_images['img'] = df_database_images.apply(lambda row: get_single_object_img(row['path_data']),axis=1)

        features = df_database_images['img'].apply(self.process_image_pdi_concat)
        feature_matrix = np.vstack(features.values)

        num_cols = feature_matrix.shape[1]
        feature_cols = [f'feat_{i}' for i in range(num_cols)]

        df_features = pd.DataFrame(feature_matrix, columns=feature_cols)
        df_database_images = pd.concat([df_database_images, df_features], axis=1)

        knn = NearestNeighbors(n_neighbors=len(df_database_images), metric='euclidean')
        knn.fit(feature_matrix)

        return [df_database_images, knn]

    def __load_df_database_images__(self, not_is_this_products=None):
        if not_is_this_products is None:
            not_is_this_products = []

        if not_is_this_products != [] and isinstance(not_is_this_products, list):
            not_is_this_products = [n for n in not_is_this_products if isinstance(n, int)]

            return self.__load_df_database_images__sql__(f"""
            SELECT d.* FROM data d
            JOIN product_data pd ON pd.id_data = d.id_data
            JOIN product p ON p.id_product = pd.id_product
            WHERE p.id_product NOT IN ({','.join((map(str, not_is_this_products)))})
            """)

        if self.df_database_images is None or self.knn is None:
            self.df_database_images, self.knn = self.__load_df_database_images__sql__("""
            SELECT d.* FROM data d
            JOIN product_data pd ON pd.id_data = d.id_data
            JOIN product p ON p.id_product = pd.id_product
            """)

        return [self.df_database_images, self.knn]

    def knn_process_image(self, query_img, not_is_this_products):
        df_database_images, knn = self.__load_df_database_images__(not_is_this_products)

        query_vec = self.process_image_pdi_concat(query_img).reshape(1, -1)
        distances, indices = knn.kneighbors(query_vec)

        results = []
        for rank, idx in enumerate(indices[0]):
            image_name = df_database_images.iloc[idx]['path_data']
            distance = round(float(distances[0][rank]), 5)

            results.append({'image_path': image_name, 'distance': distance})

        df_results = (pd.DataFrame(results).sort_values(by='distance', ascending=True).reset_index(drop=True))

        return df_results.iloc[0]['image_path']
