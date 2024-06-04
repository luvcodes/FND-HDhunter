import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

# 定义混淆矩阵的数据

cm_data = [

    ['non_rumor', 'rumor'],

    [536, 173],

    [134, 622]

]


# 提取混淆矩阵的数值部分

confusion_matrix = np.array(cm_data[1:]).astype(int)

# 创建一个新的图形

plt.figure(figsize=(6, 4))

# 绘制热力图

sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d',

            xticklabels=cm_data[0], yticklabels=cm_data[0])

# 设置x轴和y轴的标签

plt.xlabel('Predicted')

plt.ylabel('Actual')

# 显示图形

plt.show()