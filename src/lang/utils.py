import os
import dotenv
import wandb
import torch.nn.functional as F

import boto3
from huggingface_hub import HfApi


def local_env_setup() -> None:
    rel_path = os.path.join(os.path.dirname(__file__), '../..', '.env')
    dotenv.load_dotenv(rel_path)

    wandb.login(key=os.environ['WANDB_API_KEY'])
    HF_API.token = os.environ['HF_API_KEY']


def colab_env_setup() -> None:
    # mount Google Drive first
    env_path = '/content/drive/MyDrive/env/.env'
    dotenv.load_dotenv(env_path)

    wandb.login(key=os.environ['WANDB_API_KEY'])
    HF_API.token = os.environ['HF_API_KEY']
    S3_SESSION.aws_access_key_id = os.environ['AWS_ACCESS_KEY']
    S3_SESSION.aws_secret_access_key = os.environ['AWS_SECRET_KEY']


S3_SESSION = boto3.Session(
    aws_access_key_id=os.environ['AWS_ACCESS_KEY'] if 'AWS_ACCESS_KEY' in os.environ else None,
    aws_secret_access_key=os.environ['AWS_SECRET_KEY'] if 'AWS_SECRET_KEY' in os.environ else None,
)

HF_API = HfApi(
    endpoint="https://huggingface.co",
    token=os.environ['HF_API_KEY'] if 'HF_API_KEY' in os.environ else None,
)


# from transformer_lens
def lm_cross_entropy_loss(
    logits,
    tokens,
    per_token: bool = False,
):
    """Cross entropy loss for the language model, gives the loss for predicting the NEXT token.

    Args:
        logits (torch.Tensor): Logits. Shape [batch, pos, d_vocab]
        tokens (torch.Tensor[int64]): Input tokens. Shape [batch, pos]
        per_token (bool, optional): Whether to return the log probs predicted for the correct token, or the loss (ie mean of the predicted log probs). Note that the returned array has shape [batch, seq-1] as we cannot predict the first token (alternately, we ignore the final logit). Defaults to False.
    """
    log_probs = F.log_softmax(logits.logits, dim=-1)
    # Use torch.gather to find the log probs of the correct tokens
    # Offsets needed because we're predicting the NEXT token (this means the final logit is meaningless)
    # None and [..., 0] needed because the tensor used in gather must have the same rank.
    predicted_log_probs = log_probs[..., :-1, :].gather(
        dim=-1, index=tokens[..., 1:, None]
    )[..., 0]
    if per_token:
        return -predicted_log_probs
    else:
        return -predicted_log_probs.mean()