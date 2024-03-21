import datasets
import torch
import random
from tqdm import tqdm

from lang.transformer_lens import tokenize_and_concatenate

from transformers import GPT2TokenizerFast

DATASET = 'georgeyw/dsir-pile-100k'
DS_COL = 'contents'
BATCH_SIZE = 8
CONTEXT_LEN = 1024


def ed_dataset(tokenizer=None, context_length=CONTEXT_LEN, dataset=DATASET, ds_col=DS_COL, batch_size=BATCH_SIZE):
    if tokenizer is None:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    dataset = datasets.load_dataset(dataset, split='train')
    tokens_dataset = tokenize_and_concatenate(dataset,
                                              tokenizer,
                                              streaming=False,
                                              max_length=context_length,
                                              column_name=ds_col,
                                              add_bos_token=True,
                                              num_proc=12)
    data_loader = torch.utils.data.DataLoader(tokens_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    print('num batches: ', len(data_loader))
    return data_loader


def per_token_logit(model, data_loader, num_tokens=10000, seed=0, device='cuda'):
    '''Data loader must not be shuffled'''
    random.seed(seed)
    per_token_logits = []
    for batch in tqdm(data_loader):
        if len(per_token_logits) >= num_tokens:
            break
        tokens = batch['tokens'].to(device)
        batch_logits = model(tokens).detach()
        for i, logits in enumerate(batch_logits):
          if len(per_token_logits) >= num_tokens:
            break
          idx = random.randint(0, len(logits)-2)
          true_next_token = tokens[i][idx+1].cpu().item()
          per_token_logits.append(logits[idx][true_next_token].cpu().item())
    return per_token_logits