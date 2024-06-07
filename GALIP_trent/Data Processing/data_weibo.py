import pickle
import os
import jieba
import pandas as pd
import torch
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def load_datasets_from_pickle(input_folder):
    """
    Loads train, validate, and test datasets from pickle files.

    Args:
        input_folder (str): Path to the folder containing the pickle files.

    Returns:
        train (dict): Loaded training dataset.
        validate (dict): Loaded validation dataset.
        test (dict): Loaded test dataset.
    """
    # Load train dataset
    train_path = os.path.join(input_folder, 'train.pkl')
    with open(train_path, 'rb') as f:
        train = pickle.load(f)

    # Load validate dataset
    validate_path = os.path.join(input_folder, 'validate.pkl')
    with open(validate_path, 'rb') as f:
        validate = pickle.load(f)

    # Load test dataset
    test_path = os.path.join(input_folder, 'test.pkl')
    with open(test_path, 'rb') as f:
        test = pickle.load(f)

    print("Datasets loaded from pickle files in:", input_folder)

    return train, validate, test

train, validate, test = load_datasets_from_pickle('F:/DATA5703/EANN-KDD18-master/data/weibo/datasets_pickle1')

train_df = pd.DataFrame({
    'original_post': train['original_post'],
    'image_id': train['image_id'],
    'label': train['label']
})


validate_df = pd.DataFrame({
    'original_post': validate['original_post'],
    'image_id': validate['image_id'],
    'label': validate['label']
})


test_df = pd.DataFrame({
    'original_post': test['original_post'],
    'image_id': test['image_id'],
    'label': test['label']
})


data = pd.concat([train_df, validate_df, test_df], axis=0)

print(data.head())

# Segment the Chinese text into words
data['words'] = data['original_post'].apply(lambda x: list(jieba.cut(x)))

# Count the length of each row in the 'words' column
word_lengths = data['words'].apply(len)

# Plot a histogram of the lengths
word_lengths.hist(bins=50, color='blue', edgecolor='black')
plt.title('Weibo\'s Distribution of Word Lengths')
plt.xlabel('Word Length')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()

# Plot a boxplot of the lengths
plt.figure(figsize=(10, 6))
plt.boxplot(word_lengths, vert=False)
plt.title('Weibo\'s Boxplot of Word Lengths')
plt.xlabel('Word Length')
plt.show()


data['unique_words'] = data['words'].apply(lambda x: len(set(x)) / len(x))

# Plot a histogram of the lexical diversity
data['unique_words'].hist(bins=50, color='blue', edgecolor='black')
plt.title('Weibo\'s Distribution of Lexical Diversity')
plt.xlabel('Lexical Diversity')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()

# label distribution
plt.figure(figsize=(10, 6))
data['label'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Weibo\'s Distribution of Labels')
plt.ylabel('')
plt.show()
