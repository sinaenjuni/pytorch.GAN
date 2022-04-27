import torch
from torchmetrics.functional import confusion_matrix as tcm, accuracy
from sklearn.metrics import confusion_matrix as skcm


label = torch.tensor([0, 2, 2, 2, 1, 4]).cuda()
pred = torch.tensor([0, 1, 2, 1, 0, 0]).cuda()

cm = torch.nan_to_num(tcm(pred, label, num_classes=5))
acc = torch.nan_to_num(cm.trace()/cm.sum())
acc_per_cls = torch.nan_to_num(cm.diagonal()/cm.sum(0))
print("sum", cm.sum(0))

print(acc)
print(acc_per_cls)
print(cm)

dict = {"test":2}
print(dict)

dict.update({ f"cls_{idx}" : f"{acc:.4}" for idx, acc in enumerate(acc_per_cls)})
print(dict)

# cm = skcm(label, pred)
# acc = cm.trace()/cm.sum()
# acc_per_cls = cm.diagonal()/cm.sum(0)
# print("sum", cm.sum(0))
# print(acc)
# print(acc_per_cls)
# print(cm)


# cm = tcm(label, pred, num_classes=3).numpy()
# acc = cm.trace()/cm.sum()
# acc_per_cls = cm.diagonal()/cm.sum(0)
# print(acc)
# print(accuracy(pred, label))
# print(acc_per_cls)
# print(cm)