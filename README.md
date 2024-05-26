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

### To install VADER

```
pip install vaderSentiment
```

## Components
- GALIP​
- Cross model fusion​
- Classifier​
- Emotional feature

## Feature Fusion (Done with Fakeditt dataset)
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
    - Generate result: `final01.csv` file, 5415 * 1536 of tensors.

- Compiling step: 
  1. `FeatureFusionCrossModal.py` file takes the input of `train.pkl` dataset and generate the `feature3.csv` file
  2. `ReadAndMerge.py` file will generate the `feature_with_image_id.csv`
  3. `ReadAndMerge.py` file will combine the `feature_with_image_id.csv` and the `feature3.csv` file.
  4. `FeatureFusionFinal.py` file will generate the `final02.csv` file
  5. `FeatureFusionFinal.py` file also contains the code of combine the `final02.csv` file and the `feature_with_image_id.csv`, this gives us the final result: `final02_with_image_id.csv`
## GALIP training and optimizing (Done!)

## Classifier implementation in process
- Using CNN as the technique to futher process the tensor matrix generated after the feature fusion
- MLP
- Random Forest

## Sentiment Analysis (Done!)
- VADER as the pre-trained sentiment analysis model
- NLP translator for non-English inputs

## Detailed demonstration
The detailed domonstration for the project compiling result can be found in the demo video
