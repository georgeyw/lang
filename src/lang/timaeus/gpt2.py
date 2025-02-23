import os
import torch
from transformers import GPTNeoXForCausalLM
from lang.utils import S3_SESSION
from lang.utils import HF_API

S3_BUCKET = 'devinterp-language'
S3_PATH = 'checkpoints/gpt-2-small'

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


from lang.timaeus.gpt2 import load_hf_checkpoint
import transformer_lens.loading_from_pretrained as loading
from transformer_lens import HookedTransformer, HookedTransformerConfig


CONFIG_DICT = {
  'act_fn': 'gelu',
  'attention_dir': 'causal',
  'attn_only': False,
  'attn_types': None,
  'checkpoint_index': None,
  'checkpoint_label_type': None,
  'checkpoint_value': None,
  'd_head': 64,
  'd_mlp': 3072,
  'd_model': 768,
  'd_vocab': 50304,
  'd_vocab_out': 50304,
  'default_prepend_bos': True,
  'dtype': torch.bfloat16,
  'eps': 1e-05,
  'final_rms': False,
  'from_checkpoint': False,
  'gated_mlp': False,
  'init_mode': 'gpt2',
  'init_weights': False,
  'initializer_range': 0.02886751345948129,
  'model_name': None,
  'n_ctx': 1024,
  'n_devices': 1,
  'n_heads': 12,
  'n_key_value_heads': None,
  'n_layers': 12,
  'n_params': 84934656,
  'normalization_type': 'LNPre',
  'original_architecture': 'GPTNeoXForCausalLM',
  'parallel_attn_mlp': True,
  'positional_embedding_type': 'rotary',
  'post_embedding_ln': False,
  'rotary_adjacent_pairs': False,
  'rotary_base': 10000,
  'rotary_dim': 16,
  'scale_attn_by_inverse_layer_idx': False,
  'seed': None,
  'tokenizer_name': 'EleutherAI/pythia-160m',
  'tokenizer_prepends_bos': False,
  'trust_remote_code': False,
  'use_attn_in': False,
  'use_attn_result': False,
  'use_attn_scale': True,
  'use_hook_mlp_in': False,
  'use_hook_tokens': False,
  'use_local_attn': False,
  'use_split_qkv_input': False,
  'window_size': None
}

def load_hooked_transformer(step, dtype=torch.bfloat16):
  config = HookedTransformerConfig(**CONFIG_DICT)
  config.dtype = dtype
  model_ht = HookedTransformer(config).to('cuda')

  model = load_hf_checkpoint(step).to('cuda')
  official_model_name = 'EleutherAI/pythia-160m'
  state_dict = loading.get_pretrained_state_dict(
      official_model_name, config, model, dtype=torch.bfloat16
  )
  # processing the state dict adds this to match the layers in HookedTransformer
  # for some reason, this is initialized on cpu, even though the same code works 
    # and initializes on cuda for pythia 160?
  # anyways this fixes it
  state_dict['unembed.b_U'] = state_dict['unembed.b_U'].to('cuda')

  model_ht.load_and_process_state_dict(
      state_dict,
      fold_ln=True,
      center_writing_weights=True,
      center_unembed=True,
      fold_value_biases=True,
      refactor_factored_attn_matrices=False,
  )
  return model_ht