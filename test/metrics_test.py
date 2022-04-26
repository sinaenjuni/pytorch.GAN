import torch
from torchmetrics.functional import confusion_matrix, accuracy


label = torch.tensor([0, 1, 2 ,3])
pred = torch.tensor([0, 1, 1 ,2])

cm = confusion_matrix(pred, label, num_classes=5).numpy()
acc = cm.trace()/cm.sum()
acc_per_cls = cm.diagonal()/cm.sum(0)
print(acc)
print(accuracy(pred, label))
print(acc_per_cls)
print(cm)
