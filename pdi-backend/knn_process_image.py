import pandas as pd

from libs.knn_process import knn_process_df_image


class KNN:
    def __init__(self):
        self.df_all_results = pd.read_parquet('df_all_results.parquet')
        self.df_database_images = None
        self.knn = None

    def knn_process_image(self, image_process, not_is_this_products):
        df_results = self.df_all_results[~self.df_all_results['id_product'].isin(not_is_this_products)]

        df_img_test = pd.DataFrame([knn_process_df_image(image_process=image_process)])

        return 'fb099acbd20b66ddd426b1d92'
