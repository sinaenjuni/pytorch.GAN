import torch
import torch.nn.functional as F

#
# x = [[9, 5, 1],
#      [7, 9, 1],
#      [5, 4, 9]]
#
# y = [2,
#      1,
#      2]
#
# x = torch.FloatTensor(x)
# y = torch.LongTensor(y)
#
# y_ = torch.zeros(len(y),len(y))
# y_.scatter_(1, y.unsqueeze(1), 1)
# print(y_)
# print((y_*-torch.log(F.softmax(x, dim=1))).sum(1).mean())
# # F.softmax() + torch.log() = F.log_softmax()
# print((y_*-F.log_softmax(x, dim=1)).sum(1).mean())
# # F.log_softmax() + F.nll_loss() = F.cross_entropy()
# print(F.nll_loss(F.log_softmax(x, dim=1), y))
# print(F.cross_entropy(x, y))
#
#
#
#
# weight = torch.tensor([0.,
#                        100.,
#                        100.])
#
# weighted_ce = F.cross_entropy(x, y, weight=weight)
# print("weighted_ce", weighted_ce)
#
#
# img_size = 32
# fill = torch.zeros([10, 10, img_size, img_size])
# for i in range(10):
#     fill[i, i, :, :] = 1
#
# print(fill[1].mean())



# fixed noise & label
temp_z_ = torch.randn(10, 100)
fixed_z_ = temp_z_
fixed_y_ = torch.zeros(10, 1)
for i in range(9):
    fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
    temp = torch.ones(10, 1) + i
    fixed_y_ = torch.cat([fixed_y_, temp], 0)

print(fixed_z_.size())
print(fixed_y_.size())
print(fixed_y_[:11])

fixed_z_ = fixed_z_.view(-1, 100, 1, 1)
fixed_y_label_ = torch.zeros(100, 10)
fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
fixed_y_label_ = fixed_y_label_.view(-1, 10, 1, 1)


img_size = 32
# label preprocess
onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)
print(onehot[[1,2]].size())
fill = torch.zeros([10, 10, img_size, img_size])
for i in range(10):
    fill[i, i, :, :] = 1

print(fill[1].mean((1,2)))


mini_batch = 32
y_ = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor).squeeze()
print(y_)