import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from db_common import select_data
from io_minio import get_single_object_img
from libs.knn_process import knn_process_df_image


class KNN:
    def __init__(self):
        self.df_database_images = None
        self.__load_df_database_images__()
        self.knn = None

    def process_image_pdi_concat(self, image):
        return knn_process_df_image(image_process=image)

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
