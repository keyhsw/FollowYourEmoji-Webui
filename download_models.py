import os
import requests
import hashlib
from tqdm import tqdm

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_file(url, filepath):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(filepath, 'wb') as file, tqdm(
        desc=filepath,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def verify_file(filepath):
    if not os.path.exists(filepath):
        return False

    if filepath.endswith(('.bin', '.pth', '.ckpt', '.safetensors')):
        if os.path.getsize(filepath) < 1000000:  # Less than 1 MB
            return False
    elif filepath.endswith('.json'):
        try:
            with open(filepath, 'r') as f:
                f.read()
        except:
            return False

    return True

def download_and_verify(url, filepath):
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            if not verify_file(filepath):
                print(f"Downloading {filepath}...")
                download_file(url, filepath)

            if verify_file(filepath):
                print(f"File {filepath} successfully downloaded and verified.")
                return True
            else:
                print(f"File {filepath} failed verification. Attempt {attempt + 1} of {max_attempts}.")
        except Exception as e:
            print(f"Error downloading {filepath}: {str(e)}. Attempt {attempt + 1} of {max_attempts}.")

    print(f"Failed to download file {filepath} after {max_attempts} attempts.")
    return False

def download():
    base_dir = "ckpt_models"
    create_directory(base_dir)

    files = {
        "base/vae/config.json": "https://huggingface.co/daswer123/FollowYourEmoji_BaseModelPack/resolve/main/vae/config.json?download=true",
        "base/vae/diffusion_pytorch_model.bin": "https://huggingface.co/daswer123/FollowYourEmoji_BaseModelPack/resolve/main/vae/diffusion_pytorch_model.bin?download=true",
        "base/vae/diffusion_pytorch_model.safetensors": "https://huggingface.co/daswer123/FollowYourEmoji_BaseModelPack/resolve/main/vae/diffusion_pytorch_model.safetensors?download=true",
        "base/unet/config.json": "https://huggingface.co/daswer123/FollowYourEmoji_BaseModelPack/resolve/main/unet/config.json?download=true",
        "base/unet/diffusion_pytorch_model.bin": "https://huggingface.co/daswer123/FollowYourEmoji_BaseModelPack/resolve/main/unet/diffusion_pytorch_model.bin?download=true",
        "base/image_encoder/config.json": "https://huggingface.co/daswer123/FollowYourEmoji_BaseModelPack/resolve/main/image_encoder/config.json?download=true",
        "base/image_encoder/pytorch_model.bin": "https://huggingface.co/daswer123/FollowYourEmoji_BaseModelPack/resolve/main/image_encoder/pytorch_model.bin?download=true",
        "base/animatediff/mm_sd_v15_v2.ckpt": "https://huggingface.co/daswer123/FollowYourEmoji_BaseModelPack/resolve/main/animatediff/mm_sd_v15_v2.ckpt?download=true",
        "ckpts/lmk_guider.pth": "https://huggingface.co/YueMafighting/FollowYourEmoji/resolve/main/ckpts/lmk_guider.pth?download=true",
        "ckpts/referencenet.pth": "https://huggingface.co/YueMafighting/FollowYourEmoji/resolve/main/ckpts/referencenet.pth?download=true",
        "ckpts/unet.pth": "https://huggingface.co/YueMafighting/FollowYourEmoji/resolve/main/ckpts/unet.pth?download=true"
    }

    for file_path, url in files.items():
        full_path = os.path.join(base_dir, file_path)
        create_directory(os.path.dirname(full_path))
        download_and_verify(url, full_path)

if __name__ == "__main__":
    download()
