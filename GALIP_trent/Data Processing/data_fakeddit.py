import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pickle
from torchvision import transforms

import seaborn as sns
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

train = pd.read_csv('multimodal_train.tsv', sep='\t')
validate = pd.read_csv('multimodal_validate.tsv', sep='\t')
test = pd.read_csv('multimodal_test_public.tsv', sep='\t')

data = pd.concat([train, validate, test], axis=0)

print(data.head())

df = pd.DataFrame(data)

def BarPlotbyLabel(attribute):
    cross_tab = pd.crosstab(df[attribute], df['2_way_label'])
    cross_tab = cross_tab.loc[cross_tab.sum(axis=1).sort_values(ascending=False).index]
    top_10_authors = cross_tab.head(10)

    index = np.arange(len(top_10_authors))
    bar_width = 0.35  

    rects1 = plt.bar(index, top_10_authors[0], bar_width,
                 label='2_way_label = 0', color='b')
    rects2 = plt.bar(index + bar_width, top_10_authors[1], bar_width,
                 label='2_way_label = 1', color='r')

    
    plt.xlabel(attribute)
    plt.ylabel('Count')
    plt.title(f'Top 10 {attribute} by 2_way_label')
    plt.xticks(index + bar_width / 2, top_10_authors.index, rotation=90)
    plt.legend()

    
    plt.tight_layout()
    plt.show()

BarPlotbyLabel('author')
BarPlotbyLabel('domain')


df_filtered = df[df['domain'] == 'i.imgur.com']
df_filtered = df_filtered[['clean_title', 'domain', 'id', '2_way_label']]
df_filtered.columns = ['original_post', 'domain', 'image_id', 'label']
df_filtered.to_csv('filtered_data.csv', index=False)
print(df_filtered.isnull().sum())

df_filtered['word_count'] = df_filtered['original_post'].apply(lambda x: len(str(x).split()))


plt.figure(figsize=(10, 6))
plt.hist(df_filtered['word_count'], bins=50, color='blue', edgecolor='black')
plt.title('Fakeddit Competition\'s Distribution of Word Counts in "original_post"')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(10, 6))
plt.boxplot(df_filtered['word_count'], vert=False)
plt.title('Fakeddit Competition\'s Boxplot of Word Counts in "original_post"')
plt.xlabel('Word Count')
plt.show()


df_filtered['lexical_diversity'] = df_filtered['original_post'].apply(lambda x: len(set(str(x).split())) / len(str(x).split()))

print(df_filtered['lexical_diversity'])


plt.figure(figsize=(10, 6))
plt.hist(df_filtered['lexical_diversity'], bins=50, color='blue', edgecolor='black')
plt.title('Fakeddit Competition\'s Distribution of Lexical Diversity in "original_post"')
plt.xlabel('Lexical Diversity')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(10, 6))
df_filtered['label'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Fakeddit Competition\'s Distribution of Labels')
plt.ylabel('')
plt.show()

dataset_dict = df_filtered.to_dict('list')


# Images to tensor

image_dir = 'F:/DATA5703/fakeddit-master1/dataset/images'
image_list = []
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

for idx in range(len(dataset_dict['image_id'])):
    image_id = dataset_dict['image_id'][idx]
    image_file = os.path.join(image_dir, f'{image_id}.jpg')
    image = Image.open(image_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = img_transform(image)
    image_tensor = torch.from_numpy(np.array(image))
    image_list.append(image_tensor)
dataset_dict['image'] = image_list

# Save dictionaries to pickle files
with open('export_df.pkl', 'wb') as handle:
    pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Export as csv file
df = pd.DataFrame(dataset_dict)
df.to_csv('newdata_demo.csv', index=False)


# Load dictionaries from pickle files
with open('export_df.pkl', 'rb') as handle:
    export_dict = pickle.load(handle)

# Convert dictionaries to dataframes
df = pd.DataFrame(export_dict)
