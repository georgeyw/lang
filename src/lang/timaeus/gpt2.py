import os
import torch
from transformers import GPTNeoXForCausalLM
from lang.utils import S3_SESSION

S3_BUCKET = 'devinterp-language'
S3_PATH = 'checkpoints/gpt-2-small'

# def _checkpoint_s3_path(step):
#     return f's3://{S3_BUCKET}/{S3_PATH}/checkpoint-{step}'

def _checkpoint_local_path(step, local_dir='./checkpoints'):
    return os.path.join(local_dir, f'gpt-2-small/checkpoint-{step}')

def _download_checkpoint(step, bucket_name=S3_BUCKET, s3_folder=S3_PATH, local_dir='./checkpoints'):
    checkpoint_path = f'{s3_folder}/checkpoint-{step}'
    local_dir = os.path.join(local_dir, f'checkpoint-{step}')
    s3_client = S3_SESSION.client('s3',
                                  aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                                  aws_secret_access_key=os.environ['AWS_SECRET_KEY'])
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=checkpoint_path):
      for obj in page.get('Contents', []):
          # Extract relative path of the object
          filename = obj['Key'].split('/')[-1]
          local_file_path = os.path.join('./', checkpoint_path, filename)

          # Create local directory structure if it doesn't exist
          os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

          # Download file
          s3_client.download_file(bucket_name, obj['Key'], local_file_path)
          print(f"Downloaded {obj['Key']} to {local_file_path}")


def _check_if_checkpoint_is_cached(step, local_dir='./checkpoints'):
    local_dir = os.path.join(local_dir, f'checkpoint-{step}')
    if not os.path.exists(local_dir):
        return False
    return True

def _download_or_cache_checkpoint(step, local_dir='./checkpoints'):
    if not _check_if_checkpoint_is_cached(step, local_dir):
        _download_checkpoint(step, local_dir=local_dir)
    return _checkpoint_local_path(step)

def load_checkpoint(step, local_dir='./checkpoints'):
    checkpoint_path = _download_or_cache_checkpoint(step, local_dir)
    model = GPTNeoXForCausalLM.from_pretrained(
            checkpoint_path, 
            local_files_only=True,
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
        )
    return model