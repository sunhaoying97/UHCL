# UHCL: Unified Hierarchical Contrastive Learning for Video Caption

This repository is the official implementation of the paper **"Unified Hierarchical Contrastive Learning for Video Caption"**. 

The proposed method enhances the quality and distinctiveness of video caption generation through a unified hierarchical contrastive learning framework, without introducing additional inference overhead.

## Setup
Execute below scripts in the main folder, to avoid a download conflict when doing distributed pretraining.

```bash
mkdir modules/bert-base-uncased
cd modules/bert-base-uncased/
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
mv bert-base-uncased-vocab.txt vocab.txt
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz
tar -xvf bert-base-uncased.tar.gz
rm bert-base-uncased.tar.gz
cd ../../
```

Prepare the conda environment:
```bash
conda create -n clip4caption python=3.6.9 tqdm boto3 requests pandas
conda activate clip4caption
pip install torch==1.10.2 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install git+https://github.com/Maluuba/nlg-eval.git@master
pip install pycocoevalcap
pip install pickle5
pip install opencv-python==4.5.5.62
```

Download the pretrained weight of UniVL:
```bash
mkdir -p ./weight
wget -P ./weight https://github.com/microsoft/UniVL/releases/download/v0/univl.pretrained.bin
```

Download the pretrained weight of our UHCL. Place the files into the corresponding folders, and then you can start training and inference.
```bash
https://www.alipan.com/s/9XXq8RwiZ9S
code: 98ub
```

## Training & Evaluation
You may need to modify the corresponding scripts as per your needs.
```bash
cd scripts
# MSVD Dataset
bash train_msvd.sh
bash eval_msvd.sh

# MSRVTT Dataset
bash train_msrvtt.sh
bash eval_msrvtt.sh
```

## References
This repository is implemented based on the [CLIP4Caption reproduction](https://github.com/Sejong-VLI/V2T-CLIP4Caption-Reproduction),  [UniVL](https://github.com/microsoft/UniVL) and [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip)

<h1 style="color:red;">Original CLIP4Caption Notes</h1>
# Reproducing CLIP4Caption

![image](https://user-images.githubusercontent.com/5786636/210189414-aef876e0-38ab-4026-8ece-4a1f803a1005.png)

**Note**: The implementation is not considering the TSN sampling as in the CLIP4Caption paper. However, even without the TSN sampling, i.e., only using the original sampling method in CLIP4Clip, it is found that similar (even slightly better) performance results can be achieved as in the CLIP4Caption paper. While reproducing the results, it was observed that using the TSN sampling could not achieve the similar performance results as in the paper.

**Paper**: Mingkang Tang, Zhanyu Wang, Zhenhua LIU, Fengyun Rao, Dian Li, and Xiu Li. 2021. CLIP4Caption: CLIP for Video Caption. In Proceedings of the 29th ACM International Conference on Multimedia (MM '21). Association for Computing Machinery, New York, NY, USA, 4858–4862. > https://dl.acm.org/doi/10.1145/3474085.3479207


## Setup
Execute below scripts in the main folder, to avoid a download conflict when doing distributed pretraining.

```bash
mkdir modules/bert-base-uncased
cd modules/bert-base-uncased/
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
mv bert-base-uncased-vocab.txt vocab.txt
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz
tar -xvf bert-base-uncased.tar.gz
rm bert-base-uncased.tar.gz
cd ../../
```

Prepare the conda environment:
```bash
conda create -n clip4caption python=3.6.9 tqdm boto3 requests pandas
conda activate clip4caption
pip install torch==1.10.2 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install git+https://github.com/Maluuba/nlg-eval.git@master
pip install pycocoevalcap
pip install pickle5
pip install opencv-python==4.5.5.62
```

Download the pretrained weight of UniVL:
```bash
mkdir -p ./weight
wget -P ./weight https://github.com/microsoft/UniVL/releases/download/v0/univl.pretrained.bin
```

## Extract the Video Features
Follow the instructions written [here](https://github.com/Sejong-VLI/V2T-CLIP4Caption-Reproduction/tree/main/feature_extractor)

## Training & Evaluation
The shell scripts to train and to evaluate the model is provided [here](https://github.com/Sejong-VLI/V2T-CLIP4Caption-Reproduction/tree/main/scripts). You may need to modify the scripts as per your needs.

## References
This repository is implemented based on [UniVL](https://github.com/microsoft/UniVL) and [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip)
