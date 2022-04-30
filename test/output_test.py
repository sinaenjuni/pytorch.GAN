import torch
import torch.nn as nn
import torch.nn.functional as F



tensor = torch.tensor([[0.6, 0.3]])
label = torch.tensor([[0]])

F.cross_entropy(tensor, label)
nn.CrossEntropyLoss(tensor, label)


input = torch.randn(3, 2)
target = torch.empty(3, dtype=torch.long).random_(3)
target = (torch.rand(3) * 3).long()
# target = torch.empty(3, dtype=torch.long).random_(5)

F.cross_entropy(input, target)

fake_label = (torch.rand(100, 1) * 10).long().cuda()
