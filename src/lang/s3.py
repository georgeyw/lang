import os
import boto3
import pickle
from botocore.exceptions import NoCredentialsError


BUCKET = 'devinterp-language'
TEMP_FILE = '/content/temp_file.pkl'


def upload_file_to_s3(file_name, bucket, object_name=None):
    """
    Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified, file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Get AWS credentials from environment variables
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY')
    aws_secret_access_key = os.getenv('AWS_SECRET_KEY')

    # Upload the file
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        s3_client.upload_file(file_name, bucket, object_name)
    except NoCredentialsError:
        print("Credentials not available")
        return False
    except Exception as e:
        print(f"Failed to upload file: {e}")
        return False
    return True


def download_file_from_s3(object_name, bucket, file_name=None):
    """
    Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified, file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if file_name is None:
        file_name = object_name

    # Get AWS credentials from environment variables
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY')
    aws_secret_access_key = os.getenv('AWS_SECRET_KEY')

    # Upload the file
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        s3_client.download_file(bucket, object_name, file_name)
    except NoCredentialsError:
        print("Credentials not available")
        return False
    except Exception as e:
        print(f"Failed to upload file: {e}")
        return False
    return True


def push_pickle_to_s3(data, object_name, model='gpt2', bucket=BUCKET, temp_file=TEMP_FILE):
  with open(temp_file, 'wb') as file:
    pickle.dump(data, file)
  object_prefix = f'data/{model}/'
  upload_file_to_s3(temp_file, bucket,
                    object_name=f'{object_prefix}{object_name}')