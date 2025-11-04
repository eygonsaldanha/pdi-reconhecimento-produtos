from db_common import insert_data, select_data
from io_minio import upload_img
from common import generate_hash
import os
import random
import datetime
import cv2

def insert_data_in_product(full_path_file, id_product):
    path_data = generate_hash(full_path_file)
    id_data = select_data('SELECT COUNT(*) FROM DATA')['count'].iloc[0] + 2
    
    insert_data('data', [{'id_data': id_data, 'path_data': path_data, 'tp_data': 'IMG', 'dt_inclusion': datetime.datetime.now()}])
    insert_data('product_data', [{'id_product': id_product, 'id_data': id_data}])
    
    img = cv2.imread(full_path_file)
    extension = os.path.splitext(full_path_file)[1].replace('.', '').lower()
    
    upload_img(img, f'image/{extension}', key=path_data)

path = 'dataset'
for name in os.listdir(path):
    full_path = os.path.join(path, name)
    if os.path.isdir(full_path):
        id_product = select_data("SELECT COUNT(*) FROM PRODUCT")['count'].iloc[0] + 2
        insert_data('product', [{'id_product': id_product, 'nm_product': name, 'vl_product': round(5 + (random.random() * 15), 2)}])
        for full_path_file in os.listdir(full_path):
            full_path_file = os.path.join(full_path, full_path_file)
            
            insert_data_in_product(full_path_file, id_product)