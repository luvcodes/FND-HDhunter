import pickle
import pandas as pd
from PIL import Image
import os
import torch
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
    train_path = os.path.join(input_folder, 'export_df_train.pkl')
    with open(train_path, 'rb') as f:
        train = pickle.load(f)

    # Load validate dataset
    validate_path = os.path.join(input_folder, 'export_df_validate.pkl')
    with open(validate_path, 'rb') as f:
        validate = pickle.load(f)

    # Load test dataset
    test_path = os.path.join(input_folder, 'export_df_test.pkl')
    with open(test_path, 'rb') as f:
        test = pickle.load(f)

    print("Datasets loaded from pickle files in:", input_folder)

    return train, validate, test

# def load_datasets_from_pickle(input_folder):
#     """
#     Loads train, validate, and test datasets from pickle files.
#
#     Args:
#         input_folder (str): Path to the folder containing the pickle files.
#
#     Returns:
#         train (DataFrame): Loaded training dataset.
#         validate (DataFrame): Loaded validation dataset.
#         test (DataFrame): Loaded test dataset.
#     """
#     # Load train dataset
#     train_path = os.path.join(input_folder, 'export_df_train.pkl')
#     with open(train_path, 'rb') as f:
#         train = pd.DataFrame(pickle.load(f))
#
#     # Load validate dataset
#     validate_path = os.path.join(input_folder, 'export_df_validate.pkl')
#     with open(validate_path, 'rb') as f:
#         validate = pd.DataFrame(pickle.load(f))
#
#     # Load test dataset
#     test_path = os.path.join(input_folder, 'export_df_test.pkl')
#     with open(test_path, 'rb') as f:
#         test = pd.DataFrame(pickle.load(f))
#
#     print("Datasets loaded from pickle files in:", input_folder)
#
#     return train, validate, test
train, validate, test = load_datasets_from_pickle('F:/DATA5703/Fakeddit-master1/')

base_dir = r'F:\DATA5703\fakeddit-master1\dataset\images'


# Function to read images based on image_id
# def read_images(df):
#     images = {}
#     for image_id in df['image_id']:
#         # Construct the full path to the image file
#         image_path = os.path.join(base_dir, f"{image_id}.jpg")
#         # Open the image file
#         image = Image.open(image_path)
#         # Store the opened image
#         images[image_id] = image
#         image.close()
#     return images

# Read images from train, validate, and test datasets
# train_images = read_images(train)
# validate_images = read_images(validate)
# test_images = read_images(test)

#   将这三个变量加入到train, validate, test中
# train['images'] = train_images
# validate['images'] = validate_images
# test['images'] = test_images

# Function to read images based on image_id
def read_images(df):
    images = []
    for image_id in df['image_id']:
        # Construct the full path to the image file
        image_path = os.path.join(base_dir, f"{image_id}.jpg")
        # Open the image file
        image = Image.open(image_path)
        # Append the opened image to the list
        images.append(image)
        image.close()
    return images

# Read images from train, validate, and test datasets
train_images = read_images(train)
validate_images = read_images(validate)
test_images = read_images(test)

# Add these lists to train, validate, and test dataframes
train['original_images'] = train_images
validate['original_images'] = validate_images
test['original_images'] = test_images

print('111')
