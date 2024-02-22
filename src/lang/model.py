import torch.nn as nn
from prettytable import PrettyTable
from transformers.models.gpt2 import GPT2LMHeadModel

from lang.mistral.fetch import clone_and_pull_lfs_file
from lang.mistral.fetch import DIRECTORIES


def load_mistral_model(model_name: str, step: int) -> nn.Module:
    filenames = [
        "pytorch_model.bin",
        "config.json",
    ]
    clone_and_pull_lfs_file(model_name, step, filenames=filenames)
    dir_name = DIRECTORIES[model_name]
    model = GPT2LMHeadModel.from_pretrained(f"./{dir_name}")
    return model


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params} ({total_params//1e6}M)")
    return total_params