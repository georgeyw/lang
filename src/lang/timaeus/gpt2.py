import os
import torch
from transformers import GPTNeoXForCausalLM
from lang.utils import S3_SESSION
from lang.utils import HF_API

S3_BUCKET = 'devinterp-language'
S3_PATH = 'checkpoints/gpt-2-small'

CONFIG = [
    (10_000, 100),
    (20_000, 200),
    (50_000, 500),
    (100_000, 1000),
    (200_000, 2000),
    (800_001, 5000),
]

def get_sparse_steps_gpt2(step_config=CONFIG):
  steps = []
  curr_left = 0
  for config in step_config:
    curr_right = config[0]
    step_size = config[1]
    steps += list(range(curr_left, curr_right, step_size))
    curr_left = curr_right
  return steps

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
          # ensure checkpoint is correct
          directory = obj['Key'].split('/')[-2]
          if directory != f'checkpoint-{step}':
            continue
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

def load_aws_checkpoint(step, local_dir='./checkpoints'):
    if step == 0:
        return _load_init_checkpoint()
    checkpoint_path = _download_or_cache_checkpoint(step, local_dir)
    model = GPTNeoXForCausalLM.from_pretrained(
            checkpoint_path, 
            local_files_only=True,
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
        )
    return model

def _load_init_checkpoint():
    model = GPTNeoXForCausalLM.from_pretrained(
            'georgeyw/gpt-2-small-init-seed-5', 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
        )
    return model

def load_hf_checkpoint(step):
    if step == 0:
        return _load_init_checkpoint().to('cuda')
    repo_id = f'georgeyw/gpt-2-small-log-spacing'
    filepath = f'checkpoints/checkpoint-{step}'
    model_path = HF_API.hf_hub_download(repo_id, repo_type='model', filename=f'{filepath}/model.safetensors')
    HF_API.hf_hub_download(repo_id, repo_type='model', filename=f'{filepath}/config.json')
    local_path = os.path.dirname(model_path)
    model = GPTNeoXForCausalLM.from_pretrained(
            local_path, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
        )
    model.to('cuda')
    return model