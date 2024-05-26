# Fake News Detection Model using GALIP

This is the repository for CS43-2 fake news detection project using GALIP model. 

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

### Feature Fusion

- Text and Image feature
- Multi-head Attention Mechanism
- Cross-modal fusion

### GALIP

- Pre-trained model, readjusted for the project

### Classifier
- Using CNN as the technique to futher process the tensor matrix generated after the feature fusion
- MLP
- Random Forest

### Sentiment Analysis (Done!)
- VADER as the pre-trained sentiment analysis model
- NLP translator for non-English inputs
