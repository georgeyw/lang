import torch
from transformer_lens import HookedTransformerConfig, HookedTransformer

from lang.utils import HF_API


MODEL_CFG = HookedTransformerConfig(
    n_layers=2,
    d_model=256,
    d_head=32,
    n_heads=8,
    n_ctx=1024,
    d_vocab=5000,
    tokenizer_name='georgeyw/TinyStories-tokenizer-5k',
    normalization_type='LN',
    attn_only=True,
    seed=1,
    positional_embedding_type='shortformer',
)

def load_hf_checkpoint(step, n_layers=2):
  model_cfg = MODEL_CFG
  model_cfg.n_layers=n_layers
  model_name = f'L{n_layers}W256-3m-rr'
  repo_id = f'georgeyw/{model_name}'
  checkpoint_name = f'checkpoint_{step:0>7d}.pth'
  model_path = HF_API.hf_hub_download(repo_id, repo_type='model', filename=f'checkpoints/{checkpoint_name}')
  state_dict = torch.load(model_path)
  checkpoint = HookedTransformer(model_cfg)
  checkpoint.load_state_dict(state_dict)
  return checkpoint