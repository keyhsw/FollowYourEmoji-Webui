#!/bin/bash

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies"
pip install -r requirements.txt

# Install extra dependencies
echo "Installing extra dependencies"
pip install torch==2.1.1 torchaudio==2.1.1
pip install deepspeed==0.13.1

# Download pretrained model files
echo "Downloading pretrained model files"
git lfs install
git clone https://huggingface.co/YueMafighting/FollowYourEmoji ./ckpt_models
git clone https://huggingface.co/daswer123/FollowYourEmoji_BaseModelPack ./ckpt_models/base

echo "Installation complete"
