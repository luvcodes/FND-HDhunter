import pickle
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

# Assuming the pickle files are located in '../Data/weibo/datasets_pickle'
train, validate, test = load_datasets_from_pickle('../Data/weibo/datasets_pickle1')
print(train['image'][0])
image_tensor = train['image'][0]
print("Shape of the tensor:", image_tensor.shape)
print("Data type of the tensor:", image_tensor.dtype)
print('max value in the tensor:', image_tensor.max())
print("Max value in the tensor:", torch.max(image_tensor))

print('---------training set---------')
# 遍历 train['image'] 中的所有元素
for idx, image_tensor in enumerate(train['image']):
    # 计算张量中的最大值和最小值
    max_value = torch.max(image_tensor)
    min_value = torch.min(image_tensor)

    # 打印最大值和最小值
    print(f"第 {idx} 个张量中的最大值为：{max_value}")
    print(f"第 {idx} 个张量中的最小值为：{min_value}")
    # 检查张量中是否有超出 (0, 1) 范围以外的数值
    out_of_range = torch.logical_or(image_tensor < 0, image_tensor > 1)

    # 如果有超出范围的数值，打印出它们的索引位置和具体数值
    if torch.any(out_of_range):
        indices = torch.nonzero(out_of_range)
        print(f"在第 {idx} 个张量中存在超出 (0, 1) 范围以外的数值，索引位置为：{indices}")
        print(f"具体数值为：{image_tensor[indices[:, 0], indices[:, 1], indices[:, 2]]}")
    # else:
    #     print(f"第 {idx} 个张量中没有超出 (0, 1) 范围以外的数值。")

print('---------validate set---------')
for idx, image_tensor in enumerate(validate['image']):
    # 计算张量中的最大值和最小值
    max_value = torch.max(image_tensor)
    min_value = torch.min(image_tensor)

    # 打印最大值和最小值
    print(f"第 {idx} 个张量中的最大值为：{max_value}")
    print(f"第 {idx} 个张量中的最小值为：{min_value}")
    # 检查张量中是否有超出 (0, 1) 范围以外的数值
    out_of_range = torch.logical_or(image_tensor < 0, image_tensor > 1)

    # 如果有超出范围的数值，打印出它们的索引位置和具体数值
    if torch.any(out_of_range):
        indices = torch.nonzero(out_of_range)
        print(f"在第 {idx} 个张量中存在超出 (0, 1) 范围以外的数值，索引位置为：{indices}")
        print(f"具体数值为：{image_tensor[indices[:, 0], indices[:, 1], indices[:, 2]]}")
    # else:
    #     print(f"第 {idx} 个张量中没有超出 (0, 1) 范围以外的数值。")

print('---------test set---------')
for idx, image_tensor in enumerate(test['image']):
    # 计算张量中的最大值和最小值
    max_value = torch.max(image_tensor)
    min_value = torch.min(image_tensor)

    # 打印最大值和最小值
    print(f"第 {idx} 个张量中的最大值为：{max_value}")
    print(f"第 {idx} 个张量中的最小值为：{min_value}")
    # 检查张量中是否有超出 (0, 1) 范围以外的数值
    out_of_range = torch.logical_or(image_tensor < 0, image_tensor > 1)

    # 如果有超出范围的数值，打印出它们的索引位置和具体数值
    if torch.any(out_of_range):
        indices = torch.nonzero(out_of_range)
        print(f"在第 {idx} 个张量中存在超出 (0, 1) 范围以外的数值，索引位置为：{indices}")
        print(f"具体数值为：{image_tensor[indices[:, 0], indices[:, 1], indices[:, 2]]}")
    # else:
    #     print(f"第 {idx} 个张量中没有超出 (0, 1) 范围以外的数值。")

print('111')
