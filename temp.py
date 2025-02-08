import torch

# 假设已经有了定义好的长度为400的列表，其中每个元素是(40, 1)的张量
# 这里简单模拟生成一下这个列表
list_of_tensors = [torch.randn(40, 1) for _ in range(400)]

# 将列表中的张量沿着新的维度拼接起来
new_tensor = torch.cat([tensor.unsqueeze(1) for tensor in list_of_tensors], dim=1).squeeze(-1)

print(new_tensor.shape)  # 输出 (40, 400, 1)