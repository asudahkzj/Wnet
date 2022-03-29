## Wnet: Audio-Guided Video Object Segmentation via Wavelet-Based Cross-Modal Denoising Networks

This is the official implementation of the Wnet paper:


## Introduction
Audio-Guided video object segmentation is a challenging problem in visual analysis and editing, which automatically separates foreground objects from the background in a video sequence according to the referring audio expressions. However, the existing referring video object segmentation works mainly focus on the guidance of text-based referring expressions, due to the lack of modeling the semantic representations of audio-video interaction contents. In this paper, we consider the problem of audio-guided video semantic segmentation from the viewpoint of end-to-end denoising encoder-decoder network learning.  The extensive experiments show the effectiveness of our method.

## Installation
We provide instructions how to install dependencies via conda.
First, clone the repository locally:
```
git clone https://github.com/asudahkzj/Wnet.git
```
Then, install PyTorch 1.8 and torchvision 0.9:
```
conda install pytorch==1.8.0 torchvision==0.9.0
```
Install pytorch_wavelets
```
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .
```
Install pycocotools
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
```
Compile DCN module(requires GCC>=5.3, cuda>=10.0)
```
cd models/dcn
python setup.py build_ext --inplace
```

## Preparation
Download and extract 2021 version of Refer-Youtube-VOS train images from [RVOS](https://youtube-vos.org/dataset/rvos/). 
Follow the instructions [here](https://kgavrilyuk.github.io/publication/actor_action/) to download A2D-Sentences and JHMDB-Sentences dataset.
The new audio dataset (AVOS) is also [open](https://drive.google.com/drive/folders/1GcM3pt9pyt7pPjHPBaGi1nybniuSgbIg?usp=sharing).

Download the pretrained DETR models [Google Drive](https://drive.google.com/drive/folders/1DlN8uWHT2WaKruarGW2_XChhpZeI9MFG?usp=sharing) [BaiduYun](https://pan.baidu.com/s/12omUNDRjhAeGZ5olqQPpHA)(passcode:alge) on COCO and save it to the pretrained path.


## Training
<!-- Training of the model requires at least 32g memory GPU, we performed the experiment on 32g V100 card. （As the training resolution is limited by the GPU memory, if you have a larger memory GPU and want to perform the experiment, please contact with me, thanks very much) -->

To train baseline Wnet on a single node with 4 gpus, run:
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --backbone resnet101/50 --dataset_file avos --ytvos_path /path/to/ytvos --masks --pretrained_weights /path/to/pretrained_path
```

## Inference
For RVOS:
```
python inference_rvos.py --masks --model_path /path/to/model_weights --save_path /path/to/results.json
```
In a similar way for A2D-Sentences and JHMDB-Sentences.
