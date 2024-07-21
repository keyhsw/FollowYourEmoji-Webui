python -m venv venv 
venv/scripts/activate

pip install -r requirements.txt

pip install torch==2.1.1+cu121 torchaudio==2.1.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.23
pip install mediapipe==0.10.13 protobuf==4.25.3