import os
import dotenv
import wandb
from huggingface_hub import HfApi


def local_env_setup() -> None:
    rel_path = os.path.join(os.path.dirname(__file__), '../..', '.env')
    dotenv.load_dotenv(rel_path)

    wandb.login(key=os.environ['WANDB_API_KEY'])
    HF_API.token = os.environ['HF_API_KEY']


def colab_env_setup() -> None:
    # mount Google Drive first
    env_path = '/content/drive/MyDrive/induction_heads/env/.env'
    dotenv.load_dotenv(env_path)

    wandb.login(key=os.environ['WANDB_API_KEY'])
    HF_API.token = os.environ['HF_API_KEY'] 


HF_API = HfApi(
    endpoint="https://huggingface.co",
    token=os.environ['HF_API_KEY'] if 'HF_API_KEY' in os.environ else None,
)
