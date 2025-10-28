import os

import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from common import allowed_file
from pdi_methods_grouped import process_image_pdi_concat

all_images_paths = []


# Carrega todas as images do banco
def __load_files__():
    path = "/home/gbrl/Área de trabalho/Developer/Study/PDI/pdi-reconhecimento-produtos/dataset/"
    all_images_paths = []
    __find_files__(path)


def __find_files__(path):
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path):
            __find_files__(full_path)
        elif os.path.isfile(full_path) and allowed_file(name):
            all_images_paths.append(full_path)


def knn_process_image(query_img):
    __load_files__()
    all_images = [cv2.imread(name) for name in all_images_paths]
    X = np.array([process_image_pdi_concat(img) for img in all_images])

    query = process_image_pdi_concat(query_img).reshape(1, -1)

    knn = NearestNeighbors(n_neighbors=len(all_images_paths), metric="euclidean")
    knn.fit(X)

    distances, indices = knn.kneighbors(query)

    results = []
    for rank, idx in enumerate(indices[0]):
        image_name = all_images_paths[idx]
        distance = round(float(distances[0][rank]), 5)
        similarity = round(float(1 / (1 + distance)), 2)
        results.append(
            {"Rank": rank + 1, "Nome da Imagem": image_name, "Distância": distance, 'Similaridade': similarity, })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="Distância", ascending=True).reset_index(drop=True)
    pd.set_option('display.max_colwidth', None)

    return df_results[df_results['Rank'] == 1]['Nome da Imagem'].iloc[0]
