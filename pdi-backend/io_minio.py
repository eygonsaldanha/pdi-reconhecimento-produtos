import io

import boto3
import cv2
from botocore.exceptions import ClientError

from common import generate_hash

bucket_name = 'dataset'

minio_client = boto3.client(service_name="s3",  #
                            endpoint_url="http://localhost:9000",  #
                            aws_access_key_id="admin",  #
                            aws_secret_access_key="admin123"  #
                            )


def upload_img(image, content_type):
    _, extension = content_type.split('/')
    _, buffer = cv2.imencode(f".{extension}", image)
    image_bytes = io.BytesIO(buffer)

    minio_client.put_object(Bucket=bucket_name,  #
                            Key=generate_hash(content_type),  #
                            Body=image_bytes.getvalue(),  #
                            ContentType=content_type  #
                            )


def get_single_object(object_name):
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
