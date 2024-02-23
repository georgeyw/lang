import datasets
import torch

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