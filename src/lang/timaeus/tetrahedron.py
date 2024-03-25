import os
import json

import torch
import torch.nn as nn
from transformer_lens import HookedTransformerConfig, HookedTransformer

from lang.utils import HF_API


HF_MODEL_NAME = 'model.pth'

def load_hf_model(model_name: str, checkpoint_name: str = None) -> nn.Module:
    repo_id = model_name
    if not repo_id.startswith(os.environ['HF_AUTHOR']):
        repo_id = os.environ['HF_AUTHOR'] + '/' + model_name
    if model_name.startswith(os.environ['HF_AUTHOR']):
        model_name = model_name.split('/')[1]
    assert HF_API.token is not None, "Missing HF token"
    assert _check_model_exists(model_name), f"Model repo at {repo_id} does not exist"

    if checkpoint_name is None:
        filename = HF_MODEL_NAME
    else:
        if not checkpoint_name.endswith('.pth'):
            checkpoint_name += '.pth'
        filename = f'checkpoints/{checkpoint_name}'
    model_path = HF_API.hf_hub_download(repo_id, repo_type='model', filename=filename)

    config = HookedTransformerConfig(**load_hf_config(model_name, 'model'))
    state_dict = torch.load(model_path)

    model = HookedTransformer(config)
    model.load_state_dict(state_dict)
    return model


def load_hf_config(model_name: str, config_type: str) -> dict:
    repo_id = os.environ['HF_AUTHOR'] + '/' + model_name
    assert HF_API.token is not None, "Missing HF token"
    assert _check_model_exists(model_name), f"Model repo at {repo_id} does not exist"

    filename = None
    files = HF_API.list_files_info(repo_id=repo_id, repo_type='model')
    for file in files:
        if file.path.startswith(f'configs/{config_type}/'):
            if filename is not None:
                raise ValueError(
                    f"Multiple {config_type} configs found in model repo at {repo_id}.")
            filename = file.path

    config_path = HF_API.hf_hub_download(repo_id, repo_type='model', filename=filename)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        return config
    

def _check_model_exists(model_name: str) -> bool:
    models = HF_API.list_models(author=os.environ['HF_AUTHOR'], search=model_name)
    for model in models:
        if model.id == os.environ['HF_AUTHOR'] + '/' + model_name:
            return True
    return False