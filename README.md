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

#### Implementation order: 
- Start with the `ReadAndMerge.py` file, this will generate the csv file with image_id, followed by generate the 5415*1026 table with the last column of image_id merged. 
    - Merge files: `feature_with_image_id.csv` and `feature3.csv` file. 
    - Generate result: `final_features_fusion.csv`, this is the file with last column of image_id
- Use the `final_features_fusion.csv` file and the GALIP generated images to generate the final feature fusion tensors.
    - Generate result: `final01.csv` file

## GALIP training and optimizing in process

## Coolant training and optimizing in process

## Classifier implementation in process
- Using CNN as the technique to futher process the tensor matrix generated after the feature fusion