NoWebUI version - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daswer123/FollowYourEmoji-colab/blob/main/colab/follow_emoji_collab_nowebui.ipynb)

WebUI version - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daswer123/FollowYourEmoji-colab/blob/main/colab/follow_emoji_collab_webui.ipynb)

# About this fork

In this fork I implemented the functionality on webui, I modified the original code a bit so that it would be possible to expose as many settings as possible.

Here you can experience the full functionality of FollowYourEmoji.

Here is a short list of what has been added to webui
1. Ability to conveniently upload a reference picture and video.
2. a tool to crop the reference picture so that it fits the video perfectly.
3. Ability to see a preview of the cropped picture, zoom in or shift the cropping.
4. The ability to upload any video without the need for additional processing (the interface itself processes everything).
5. Ability to upload .npy file, as well as choose from a folder. Each processed video is added to the folder, which allows you to select the same video without re-processing.
6. Ability to see how the animation will look like before generation.
7. Many different settings, both custom and official.
8. Ability to specify the FPS of the output video 
9. Mechanism to remove "Anomalous frames" in automatic mode
10. Possibility to get all frames in the archive in addition to video.

And many more small improvements that will allow you to work conveniently in one interface.

## About google colab

Colab has been tested on the free version, everything works. Processing time is about 5 minutes for 300 frames.

**Attention free colab is working at the limit of its capabilities, and I do not advise you to change the generation parameters, because you are likely to crash due to lack of RAM**

You can try FollowYourEmoji online by clicking one of the buttons above!

## Screenshoot

## Installation

Before you start, make sure you have: CUDA 12.1, ffmpeg, python 3.10

There are two ways to install FollowYourEmoji-Webui: Simple and Manual. Choose the method that suits you best.

### Simple Installation

1. Clone the repository:
```
git clone https://github.com/daswer123/FollowYourEmoji-Webui.git
cd FollowYourEmoji-Webui
```

2. Run the installation script:
   - For Windows: `install.bat`
   - For Linux: `./install.sh`

3. Start the application:
   - For Windows: `start.bat`
   - For Linux: `./start.sh`

### Manual Installation

#### For Linux:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install extra dependencies
pip install torch==2.1.1 torchaudio==2.1.1
pip install deepspeed==0.13.1

# Download pretrained model files
git lfs install
git clone https://huggingface.co/YueMafighting/FollowYourEmoji ./ckpt_models
git clone https://huggingface.co/daswer123/FollowYourEmoji_BaseModelPack ./ckpt_models/base
```

#### For Windows:

```batch
# Create virtual environment
python -m venv venv
venv\scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install extra dependencies
pip install torch==2.1.1+cu121 torchaudio==2.1.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install https://github.com/daswer123/deepspeed-windows/releases/download/13.1/deepspeed-0.13.1+cu121-cp310-cp310-win_amd64.whl

# Download pretrained model files
git lfs install
git clone https://huggingface.co/YueMafighting/FollowYourEmoji ./ckpt_models
git clone https://huggingface.co/daswer123/FollowYourEmoji_BaseModelPack ./ckpt_models/base
```

## Launch

After installation, you have several options to launch FollowYourEmoji-Webui:

### Using launch scripts

- For Windows: `start.bat`
- For Linux: `./start.sh`

### Manual launch

1. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux: `source venv/bin/activate`

2. Run the application:
   - Windows: `python app.py`
   - Linux: `python3 app.py`

### Sharing your instance

To create a tunnel and share your instance with others, add the `--share` flag:

- Windows: `python app.py --share`
- Linux: `python3 app.py --share`

This will generate a public URL that you can share with others, allowing them to access your FollowYourEmoji-Webui instance remotely.



# Original README 

<!-- <h1 align="center"><span>Follow-Your-Emoji: Fine-Controllable and Expressive Freestyle Portrait Animation</strong></h1>

<p id="authors" class="serif" align='center'>
    <a href="https://github.com/mayuelala">Yue Ma<sup>1*</sup></a>
    <a href="https://yingqinghe.github.io/">Hongyu Liu<sup>1*<dag></sup></a>
    <a href="https://follow-your-emoji.github.io/">Hongfa Wang<sup>2,3*</sup></a>
    <a href="https://scholar.google.com/citations?user=DIpLfK4AAAAJ">Heng Pan<sup>2*</sup></a>
    <a href="https://yingqinghe.github.io/">Yingqing He<sup>1</sup></a> <br>
    <a href="https://0-scholar-google-com.brum.beds.ac.uk/citations?user=j3iFVPsAAAAJ&hl=zh-CN">Junkun Yuan<sup>2</sup></a>
    <a href="https://ailingzeng.site/">Ailing Zeng<sup>2</sup></a>
    <a href="https://follow-your-emoji.github.io/">Chengfei Cai<sup>2</sup></a>
    <a href="https://scholar.google.com.hk/citations?user=9akH-n8AAAAJ&hl=en">Heung-Yeung Shum<sup>1,3</sup></a> 
    <a href="https://scholar.google.com/citations?user=AjxoEpIAAAAJ&hl=zh-CN">Wei Liu<sup>2✝</sup></a>
    <a href="https://cqf.io/">Qifeng Chen<sup>1✝</sup></a>
    <br>

</p>

<a href='https://arxiv.org/abs/2403.08268'><img src='https://img.shields.io/badge/ArXiv-2403.08268-red'></a> 
<a href='https://follow-your-click.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  ![visitors](https://visitor-badge.laobi.icu/badge?page_id=mayuelala.FollowYourClick&left_color=green&right_color=red)  [![GitHub](https://img.shields.io/github/stars/mayuelala/FollowYourClick?style=social)](https://github.com/mayuelala/FollowYourClick) 
</div> -->


<div align="center">
<h2><font color="red"> Follow-Your-Emoji </font></center> <br> <center>Fine-Controllable and Expressive Freestyle Portrait Animation</h2>

[Yue Ma*](https://mayuelala.github.io/), [Hongyu Liu*](https://kumapowerliu.github.io/), [Hongfa Wang*](https://github.com/mayuelala/FollowYourEmoji), [Heng Pan*](https://github.com/mayuelala/FollowYourEmoji), [Yingqing He](https://github.com/YingqingHe), [Junkun Yuan](https://0-scholar-google-com.brum.beds.ac.uk/citations?user=j3iFVPsAAAAJ&hl=zh-CN),  [Ailing Zeng](https://ailingzeng.site/), [Chengfei Cai](https://github.com/mayuelala/FollowYourEmoji), 
[Heung-Yeung Shum](https://scholar.google.com.hk/citations?user=9akH-n8AAAAJ&hl=en), [Wei Liu](https://scholar.google.com/citations?user=AjxoEpIAAAAJ&hl=zh-CN) and [Qifeng Chen](https://cqf.io)

<a href='https://arxiv.org/abs/2406.01900'><img src='https://img.shields.io/badge/ArXiv-2406.01900-red'></a> 
<a href='https://follow-your-emoji.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='assets/wechat_group.png'><img src='https://badges.aleen42.com/src/wechat.svg'></a> ![visitors](https://visitor-badge.laobi.icu/badge?page_id=mayuelala.FollowYourEmoji&left_color=green&right_color=red)  [![GitHub](https://img.shields.io/github/stars/mayuelala/FollowYourEmoji?style=social)](https://github.com/mayuelala/FollowYourEmoji,pko) 
</div>

<!-- <table class="center">
  <td><img src="https://follow-your-emoji.github.io/src/teaser/teaser.gif"></td>
  <tr>
    <td align="center" >🤪 For more results, visit our <a href="https://follow-your-emoji.github.io/"><strong>homepage</strong></td>
  <tr>
</td>

</table > -->


## 📣 Updates

- **[2024.07.18]** 🔥 Release `inference code`, `config` and `checkpoints`!
- **[2024.06.07]** 🔥 Release Paper and Project page!

## 🤪 Gallery
<img src="images/index.png" alt="Image 1">

<p>We present <span style="color: #c20557ee">Follow-Your-Emoji</span>, a diffusion-based framework for portrait animation, which animates a reference portrait with target landmark sequences.</p>

## 🤪 Getting Started

### 1. Clone the code and prepare the environment

```bash
pip install -r requirements.txt
```

### 2. Download pretrained weights

[FollowYourEmoji] We also provide our pretrained checkpoints in [Huggingface](https://huggingface.co/YueMafighting/FollowYourEmoji). you could download them and put them into checkpoints folder to inference our model.


### 3. Inference 🚀

```bash
bash infer.sh
```

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.run \
    --nnodes 1 \
    --master_addr $LOCAL_IP \
    --master_port 12345 \
    --node_rank 0 \
    --nproc_per_node 1 \
    infer.py \
    --config ./configs/infer.yaml \
    --model_path /path/to/model \
    --input_path your_own_images_path \
    --lmk_path ./inference_temple/test_temple.npy  \
    --output_path your_own_output_path
```

## 🤪 Make Your Emoji
You can make your own emoji using our model. First, you need to make your emoji temple using MediaPipe. We provide the script in ```make_temple.ipynb```. You can replace the video path with your own emoji video and generate the ```.npy``` file.


```bash
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.run \
    --nnodes 1 \
    --master_addr $LOCAL_IP \
    --master_port 12345 \
    --node_rank 0 \
    --nproc_per_node 1 \
    infer.py \
    --config ./configs/infer.yaml \
    --model_path /path/to/model \
    --input_path your_own_images_path \
    --lmk_path  your_own_temple_path \
    --output_path your_own_output_path
```


## 👨‍👩‍👧‍👦 Follow Family
[Follow-Your-Pose](https://github.com/mayuelala/FollowYourPose): Pose-Guided text-to-Video Generation.

[Follow-Your-Click](https://github.com/mayuelala/FollowYourClick): Open-domain Regional image animation via Short Prompts.

[Follow-Your-Handle](https://github.com/mayuelala/FollowYourHandle): Controllable Video Editing via Control Handle Transformations.

[Follow-Your-Emoji](https://github.com/mayuelala/FollowYourEmoji): Fine-Controllable and Expressive Freestyle Portrait Animation.
  
## Citation 💖
If you find Follow-Your-Emoji useful for your research, welcome to 🌟 this repo and cite our work using the following BibTeX:
```bibtex
@article{ma2024follow,
  title={Follow-Your-Emoji: Fine-Controllable and Expressive Freestyle Portrait Animation},
  author={Ma, Yue and Liu, Hongyu and Wang, Hongfa and Pan, Heng and He, Yingqing and Yuan, Junkun and Zeng, Ailing and Cai, Chengfei and Shum, Heung-Yeung and Liu, Wei and others},
  journal={arXiv preprint arXiv:2406.01900},
  year={2024}
}
```
