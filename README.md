## Wnet: Audio-Guided Video Object Segmentation via Wavelet-Based Cross-Modal Denoising Networks

This is the official implementation of the Wnet paper:


## Introduction
Audio-Guided video object segmentation is a challenging problem in visual analysis and editing, which automatically separates foreground objects from the background in a video sequence according to the referring audio expressions. However, the existing referring video object segmentation works mainly focus on the guidance of text-based referring expressions, due to the lack of modeling the semantic representations of audio-video interaction contents. In this paper, we consider the problem of audio-guided video semantic segmentation from the viewpoint of end-to-end denoising encoder-decoder network learning.  The extensive experiments show the effectiveness of our method.

## Installation
First, clone the repo locally:
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
If you encounter the problem of missing ytvos.py file, you can manually download the file from [here](https://github.com/youtubevos/cocoapi/tree/master/PythonAPI/pycocotools) and put it in the installed pycocotools folder.

Compile DCN module(requires GCC>=5.3, cuda>=10.0)
```
cd models/dcn
python setup.py build_ext --inplace
```

## Preparation
Download and extract 2021 version of Refer-Youtube-VOS train images from [RVOS](https://youtube-vos.org/dataset/rvos/). 
Follow the instructions [here](https://kgavrilyuk.github.io/publication/actor_action/) to download A2D-Sentences and JHMDB-Sentences dataset.
The new audio dataset (AVOS) is also [open](https://drive.google.com/drive/folders/1GcM3pt9pyt7pPjHPBaGi1nybniuSgbIg?usp=sharing).
You need to extract MFCC features from audio files and convert video files in A2D into image frames. 
For extracting MFCC features, you can refer to [here](https://blog.csdn.net/chengtang2028/article/details/100837043).

Then, organize the files as follows: 

```text
Wnet/data
├── rvos/ 
│   ├── ann/
|   |   └── *.json (annotation files)  
│   ├── train/
|   │   ├── JPEGImages/
|   │   └── Annotations/
│   └── meta_expressions/train/meta_expressions.json
├── a2d/
|   ├── Release/
|   │   ├── videoset.csv 
|   │   ├── clips320/  
|   │   └── pngs320/  (image frames extracted from videos in clips320/)
|   ├── a2d_annotation_with_instances/
|   └── a2d_annotation_info.txt
├── jhmdb/
|   ├── Rename_Images/
|   ├── puppet_mask/
|   ├── jhmdb_annotation.txt
|   └── video_info.json
├── rvos_audio_feature/
|   └── *.npy   (mfcc features extracted from rvos audio file)
└── a2d_j_audio_feature/
    └── *.npy   (mfcc features extracted from a2d/jhmdb audio file)
```
*The files a2d_annotation_info.txt and video_info.json can be downloaded [here](https://1drv.ms/u/s!Ak4bpr3_F0KQe0kFeZMYifC7ZoA?e=Ogw3GQ).

Download the pretrained DETR models [OneDrive](https://1drv.ms/u/s!Ak4bpr3_F0KQclwyOviBLBMsT-A?e=VTfex6) on COCO and save it to the pretrained path.

## Training

For AVOS dataset (which contains the videos and audios of RVOS, A2D-Sentences and JHMDB-Sentences, and JHMDB-Sentences dataset is only for evaluation):
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --backbone resnet101/50 --dataset_file avos --ytvos_path /path/to/ytvos --masks --pretrained_weights /path/to/pretrained_path
```
For RVOS dataset:
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --backbone resnet101/50 --dataset_file ytvos --ytvos_path /path/to/ytvos --masks --pretrained_weights /path/to/pretrained_path
```
For A2D-Sentences dataset:
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --backbone resnet101/50 --dataset_file a2d --num_frames 8 --num_queries 8 --masks --pretrained_weights /path/to/pretrained_path
```

## Inference
For AVOS dataset, we need to test three datasets separately: 
```
python inference_rvos.py --masks --model_path /path/to/model_weights --save_path /path/to/results.json
python evaluate/evaluate.py /path/to/results.json
python inference_a2d.py --masks --model_path /path/to/model_weights
python inference_jh.py --masks --model_path /path/to/model_weights
```
For RVOS dataset:
```
python inference_rvos.py --masks --model_path /path/to/model_weights --save_path /path/to/results.json
python evaluate/evaluate.py /path/to/results.json
```
For A2D-Sentences dataset:
```
python inference_a2d.py --masks --model_path /path/to/model_weights --num_frames 8 --num_queries 8
```
For JHMDB-Sentences dataset (directly using the model trained on A2D-Sentences):
```
python inference_jh.py --masks --model_path /path/to/model_weights --num_frames 8 --num_queries 8
```

## Models

We provide Wnet models trained from the AVOS dataset, which contains the videos of RVOS, A2D-Sentences and JHMDB-Sentences.


|Name| Backbone      | J | F  | J&F | Chenkpoint
|:---:| :-----------: | :-----------: | :-----------: | :---: | :---: |
|Wnet|ResNet-50|43.0|45.0|44.0| [Link](https://1drv.ms/u/s!Ak4bpr3_F0KQakQ2gA_2DQ8nDhI?e=iJ3cDP)        |
