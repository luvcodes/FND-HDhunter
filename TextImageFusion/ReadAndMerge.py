import pandas as pd
import pickle

# 加载train.pkl文件
with open('C:\\Users\\ryanw\\OneDrive\\Desktop\\datasets_pickle1\\train.pkl', 'rb') as f:
    data = pickle.load(f)

# 创建DataFrame
df = pd.DataFrame({
    'original_post': data['original_post'],
    'image_id': data['image_id']
})

# 将处理好的数据保存到CSV文件中, 这个就是有image_id的csv文件
df.to_csv('TextImageFusion\\csvFilesCollection\\feature_with_image_id.csv', index=False)

# ---------------
# 拼接带有image_id的csv文件和原来的feature3.csv文件，然后保存到新的文件中
# 读取特征文件和image_id文件
features_df = pd.read_csv('TextImageFusion\\csvFilesCollection\\features3.csv')
image_id_df = pd.read_csv('TextImageFusion\\csvFilesCollection\\feature_with_image_id.csv')

# 确保两个DataFrame行数一致
if len(features_df) == len(image_id_df):
    # 添加image_id列到特征DataFrame
    features_df['image_id'] = image_id_df['image_id']
    # 保存新的CSV文件
    features_df.to_csv('TextImageFusion\\csvFilesCollection\\final_features_fusion.csv', index=False)
    print("New CSV file with image_id has been saved successfully.")
else:
    print("Error: The number of rows in features3.csv does not match feature_with_image_id.csv")
