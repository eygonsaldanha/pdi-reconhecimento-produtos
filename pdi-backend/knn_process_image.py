import cv2
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from db_common import select_data
from io_minio import get_single_object_img
from libs.knn_process import extrair_features_otimizado


class KNN:
    def __init__(self, cache_file='knn_model_cache.pkl', use_optimized_features=True):
        """
        Inicializa o modelo KNN com normalização e cache.
        
        Args:
            cache_file: Caminho para o arquivo de cache
            use_optimized_features: Se True, usa features otimizadas (~4.4k valores).
                                   Se False, usa features antigas (~100k valores)
        """
        self.df_database_images = None
        self.knn = None
        self.scaler = StandardScaler()
        self.cache_file = cache_file
        self.use_optimized_features = use_optimized_features
        
        # Tentar carregar do cache
        if os.path.exists(cache_file):
            print(f"🔄 Carregando modelo do cache: {cache_file}")
            self.__load_from_cache__()
        else:
            print(f"📊 Cache não encontrado. Carregando dataset...")
            self.__load_df_database_images__()

    def process_image_pdi_concat(self, image):
        """Extrai features da imagem"""
        if self.use_optimized_features:
            return extrair_features_otimizado(image_process=image)
        else:
            # Fallback para função antiga (deprecated)
            from libs.knn_process import knn_process_df_image
            return knn_process_df_image(image_process=image)

    def __load_df_database_images__sql__(self, sql):
        """Carrega dataset do banco e treina o modelo"""
        print("📥 Carregando imagens do banco de dados...")
        df_database_images = select_data(sql)
        
        print(f"🖼️  Processando {len(df_database_images)} imagens...")
        df_database_images['img'] = df_database_images.apply(
            lambda row: get_single_object_img(row['path_data']), axis=1
        )

        # Extrair features
        print("🔍 Extraindo features...")
        features_list = []
        for idx, row in df_database_images.iterrows():
            try:
                features = self.process_image_pdi_concat(row['img'])
                features_list.append(features)
            except Exception as e:
                print(f"⚠️  Erro ao processar {row['path_data']}: {e}")
                # Criar vetor de features vazio
                if self.use_optimized_features:
                    features_list.append(np.zeros(4413))
                else:
                    features_list.append(np.zeros(100000))
        
        feature_matrix = np.vstack(features_list)
        print(f"📐 Shape das features: {feature_matrix.shape}")

        # Normalizar features
        print("📊 Normalizando features com StandardScaler...")
        feature_matrix_normalized = self.scaler.fit_transform(feature_matrix)

        # Criar DataFrame com features
        num_cols = feature_matrix_normalized.shape[1]
        feature_cols = [f'feat_{i}' for i in range(num_cols)]
        df_features = pd.DataFrame(feature_matrix_normalized, columns=feature_cols)
        df_database_images = pd.concat([df_database_images.reset_index(drop=True), df_features], axis=1)

        # Treinar KNN
        n_neighbors = min(10, len(df_database_images))
        print(f"🤖 Treinando KNN com k={n_neighbors}...")
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(feature_matrix_normalized)

        print("✅ Modelo treinado com sucesso!")
        return [df_database_images, knn]

    def __load_df_database_images__(self, not_is_this_products=None):
        """Carrega dataset com opção de filtrar produtos"""
        if not_is_this_products is None:
            not_is_this_products = []

        # Se há produtos para filtrar, recarrega o dataset
        if not_is_this_products != [] and isinstance(not_is_this_products, list):
            not_is_this_products = [n for n in not_is_this_products if isinstance(n, int)]
            
            print(f"🚫 Filtrando produtos: {not_is_this_products}")
            return self.__load_df_database_images__sql__(f"""
            SELECT d.* FROM data d
            JOIN product_data pd ON pd.id_data = d.id_data
            JOIN product p ON p.id_product = pd.id_product
            WHERE p.id_product NOT IN ({','.join((map(str, not_is_this_products)))})
            """)

        # Se já tem o modelo carregado, retorna
        if self.df_database_images is not None and self.knn is not None:
            return [self.df_database_images, self.knn]
        
        # Carrega dataset completo
        self.df_database_images, self.knn = self.__load_df_database_images__sql__("""
        SELECT d.* FROM data d
        JOIN product_data pd ON pd.id_data = d.id_data
        JOIN product p ON p.id_product = pd.id_product
        """)
        
        # Salvar cache
        self.__save_to_cache__()

        return [self.df_database_images, self.knn]

    def knn_process_image(self, query_img, not_is_this_products):
        """Processa uma imagem query e retorna o vizinho mais próximo"""
        df_database_images, knn = self.__load_df_database_images__(not_is_this_products)

        # Extrair e normalizar features da query
        query_vec = self.process_image_pdi_concat(query_img)
        query_vec_normalized = self.scaler.transform(query_vec.reshape(1, -1))
        
        # Buscar vizinhos mais próximos
        distances, indices = knn.kneighbors(query_vec_normalized)

        # Preparar resultados
        results = []
        for rank, idx in enumerate(indices[0]):
            image_name = df_database_images.iloc[idx]['path_data']
            distance = round(float(distances[0][rank]), 5)
            results.append({'image_path': image_name, 'distance': distance})

        df_results = pd.DataFrame(results).sort_values(by='distance', ascending=True).reset_index(drop=True)
        
        return df_results.iloc[0]['image_path']

    def __save_to_cache__(self):
        """Salva modelo, scaler e dataset em cache"""
        try:
            cache_data = {
                'df_database_images': self.df_database_images,
                'knn': self.knn,
                'scaler': self.scaler,
                'use_optimized_features': self.use_optimized_features
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"💾 Modelo salvo em cache: {self.cache_file}")
        except Exception as e:
            print(f"⚠️  Erro ao salvar cache: {e}")

    def __load_from_cache__(self):
        """Carrega modelo, scaler e dataset do cache"""
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            self.df_database_images = cache_data['df_database_images']
            self.knn = cache_data['knn']
            self.scaler = cache_data['scaler']
            self.use_optimized_features = cache_data.get('use_optimized_features', True)
            print(f"✅ Modelo carregado do cache com sucesso!")
            print(f"📊 Dataset: {len(self.df_database_images)} imagens")
        except Exception as e:
            print(f"⚠️  Erro ao carregar cache: {e}")
            print(f"📊 Carregando dataset do banco de dados...")
            self.__load_df_database_images__()
    
    def clear_cache(self):
        """Remove o arquivo de cache"""
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
            print(f"🗑️  Cache removido: {self.cache_file}")
        else:
            print(f"ℹ️  Cache não existe: {self.cache_file}")
