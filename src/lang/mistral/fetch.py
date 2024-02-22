import subprocess
import sys


REPO_URLS = {
    'alias': 'https://huggingface.co/stanford-crfm/alias-gpt2-small-x21',
    'battlestar': 'https://huggingface.co/stanford-crfm/battlestar-gpt2-small-x49',
    'caprica': 'https://huggingface.co/stanford-crfm/caprica-gpt2-small-x81',
    'darkmatter': 'https://huggingface.co/stanford-crfm/darkmatter-gpt2-small-x343',
    'expanse': 'https://huggingface.co/stanford-crfm/expanse-gpt2-small-x777',
}

DIRECTORIES = {
    'alias': 'alias-gpt2-small-x21',
    'battlestar': 'battlestar-gpt2-small-x49',
    'caprica': 'caprica-gpt2-small-x81',
    'darkmatter': 'darkmatter-gpt2-small-x343',
    'expanse': 'expanse-gpt2-small-x777',
}


def checkpoints_list() -> list:
    output = []
    output += list(range(0, 100, 10))
    output += list(range(100, 2000, 50))
    output += list(range(2000, 20_000, 100))
    output += list(range(20_000, 400_001, 1000))
    return output


def install_git_lfs():
    try:
        print("Installing Git LFS...")
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y", "git-lfs"], check=True)
        subprocess.run(["git", "lfs", "install"], check=True)
        print("Git LFS installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install Git LFS. Error: {e}")
        sys.exit(1)


def check_git_lfs_installed():
    try:
        subprocess.run(["git-lfs", "version"], check=True, stdout=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False


def clone_and_pull_lfs_file(model_name, step, filenames=["pytorch_model.bin"]):
    repo_url = REPO_URLS[model_name]
    if not check_git_lfs_installed():
        install_git_lfs()
    assert step in checkpoints_list(), f"Invalid checkpoint: {step}"
    branch_name = f'checkpoint-{step}'
    dir_name = repo_url.split('/')[-1].replace('.git', '')
    subprocess.run(["git", "clone", repo_url, "--branch", branch_name, "--single-branch", "--no-checkout"], check=True)
    for filename in filenames:
      subprocess.run(["git", "checkout", branch_name, filename], cwd=dir_name, check=True)
      subprocess.run(["git", "lfs", "pull", "--include", filename], cwd=dir_name, check=True)
