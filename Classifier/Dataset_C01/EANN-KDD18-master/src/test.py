# 示例数据
import numpy as np
import torch

dataset = {
    'post_text': [
        [207, 3025, 2269, 3872],
        [566, 23, 2269, 4566]
    ]
}

text = torch.LongTensor(np.array(dataset['post_text']))