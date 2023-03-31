import numpy as np
import torch

# 定义两个张量 x 和 y，并设置 requires_grad=True 以便计算梯度
'''x = torch.tensor((2.0,4.0), requires_grad=True)
y = torch.randn(4)'''

y = torch.randn(10)
print(y)

print(y.view((1,-1)))
''''
num_inputs = 4
num_outputs = 3
print(np.random.normal(0, 0.01, (num_inputs, num_outputs)))
'''

t = torch.tensor([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# 定义一个索引张量
index = torch.tensor([[0, 1], [1, 0]])

# 使用 gather 提取元素
result = t.gather(1, index)

# 输出结果
print(result)