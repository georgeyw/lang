import datasets
import torch

from lang.transformer_lens import tokenize_and_concatenate

from transformers import GPT2TokenizerFast

DATASET = 'georgeyw/dsir-pile-1m-2'
DS_COL = 'contents'
BATCH_SIZE = 8
CONTEXT_LEN = 1024

def _reformat_tokens_dataset_for_learning_coeff(tokens_dataset, batch_size):
    def custom_collate(batch):
        tokens = [item['tokens'] for item in batch]
        tokens_tensor = torch.stack(tokens)
        return [tokens_tensor, tokens_tensor.clone()]

    return torch.utils.data.DataLoader(tokens_dataset,
                                        batch_size=batch_size,
                                        collate_fn=custom_collate,
                                        shuffle=True,
                                        num_workers=2,
                                        pin_memory=True)


def llc_dataset(tokenizer=None, context_length=CONTEXT_LEN, dataset=DATASET, ds_col=DS_COL, batch_size=BATCH_SIZE):
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
    llc_dataset = _reformat_tokens_dataset_for_learning_coeff(tokens_dataset,
                                                              batch_size)
    print('num batches: ', len(llc_dataset))
    return llc_dataset
