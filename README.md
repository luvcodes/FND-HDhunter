# FND-HDhunter
This is the repository for CS-43-2 fake news detection using GALIP model implementation repository

# Fake News Detection Model using GLAIP

## Environments

- python 3.9
- Pytorch 1.9
- At least 1x24GB 3090 GPU (for training)
- Only CPU (for sampling) 

## Installation

### To install GALIP

Clone this repo.
```
git clone https://github.com/tobran/GALIP
pip install -r requirements.txt
```
### To install CLIP

Install [CLIP](https://github.com/openai/CLIP)

## Components
- GALIP​
- Cross model contrastive learning​
- Cross model fusion​
- Cross model aggregation​
- Classifier​
- Emotional feature

## Feature in progress
### Feature fusion - ImageAndText
The first version of feature fusion implmentation code is currently in TextImageFusion folder. 
- Current implmentation method: Cross-modal with multi-head attention mechanism
    - Currently using BERT and ResNet50 as the text-image encoder combination
- CLIPs with concatenation method

## GALIP training and optimizing in process

## Coolant training and optimizing in process

## Classifier implementation in process
- Using CNN as the technique to futher process the tensor matrix generated after the feature fusion