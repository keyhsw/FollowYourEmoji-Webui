python -m venv venv 
venv/scripts/activate

Echo Install dependencies
pip install -r requirements.txt

Echo Install extra dependencies
pip install torch==2.1.1+cu121 torchaudio==2.1.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install https://github.com/daswer123/deepspeed-windows/releases/download/13.1/deepspeed-0.13.1+cu121-cp310-cp310-win_amd64.whl
@REM pip install xformers==0.0.23
@REM pip install mediapipe==0.10.13 protobuf==4.25.3

Echo Download pretrained model files
git lfs install
git clone https://huggingface.co/YueMafighting/FollowYourEmoji ./ckpt_models
git clone https://huggingface.co/daswer123/FollowYourEmoji_BaseModelPack ./ckpt_models/base