import io

import numpy as np
import boto3
import cv2
from botocore.exceptions import ClientError
import io
import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError
from common import generate_hash
import mimetypes
import pandas as pd
from botocore.exceptions import ClientError
from io import BytesIO

minio_client = boto3.client(service_name="s3",  #
                            endpoint_url="http://localhost:9090",  #
                            aws_access_key_id="admin",  #
                            aws_secret_access_key="admin123"  #
                            )

def ensure_bucket(bucket_name: str):
    try:
        minio_client.head_bucket(Bucket=bucket_name)
    except ClientError:
        minio_client.create_bucket(Bucket=bucket_name)


def upload_img_path(image_path, key=None):
    image = cv2.imread(image_path)
    content_type, _ = mimetypes.guess_type(image_path)
    return upload_img(image, content_type, key)

def upload_img(image, content_type, key=None):
    bucket_name = 'dataset'
    ensure_bucket(bucket_name)
    
    if not key:
        key = generate_hash(content_type)
    
    _, extension = content_type.split('/')
    _, buffer = cv2.imencode(f".{extension}", image)
    image_bytes = io.BytesIO(buffer)

    minio_client.put_object(Bucket=bucket_name,  #
                            Key=key,  #
                            Body=image_bytes.getvalue(),  #
                            ContentType=content_type  #
                            )

def upload_parquet(df, key):
    bucket_name = 'dataset-parquet'
    ensure_bucket(bucket_name)
    
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df), buffer)
    buffer.seek(0)
    
    minio_client.put_object(
        Bucket=bucket_name, #
        Key=key, #
        Body=buffer.getvalue(), #
        ContentType='application/octet-stream' 
    )

def get_image_minio(object_name, bucket_name='dataset'):
    try:
        response = minio_client.get_object(Bucket=bucket_name, Key=object_name)

        file_data = response['Body'].read()
        response['Body'].close()

        return file_data

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            print(f"Erro: Objeto '{object_name}' não encontrado no bucket '{bucket_name}'.")
        else:
            print(f"Erro ao baixar objeto: {e}")
        return None

def get_parquet_minio(object_name, bucket_name='dataset-parquet'):
    try:
        response = minio_client.get_object(Bucket=bucket_name, Key=object_name)
        file_data = response['Body'].read()
        response['Body'].close()

        buffer = BytesIO(file_data)
        return pd.read_parquet(buffer, engine='pyarrow')
    
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            print(f"Erro: Objeto '{object_name}' não encontrado no bucket '{bucket_name}'.")
        else:
            print(f"Erro ao baixar/parquet: {e}")
        return None

    except Exception as e:
        print(f"Erro inesperado ao processar '{object_name}': {e}")
        return None