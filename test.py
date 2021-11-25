import torch

# # fixed noise & label
# temp_z_ = torch.randn(10, 100)
# fixed_z_ = temp_z_
# fixed_y_ = torch.zeros(10, 1)
#
#
# for i in range(9):
#     fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
#     temp = torch.ones(10, 1) + i
#     fixed_y_ = torch.cat([fixed_y_, temp], 0)
#
# print(fixed_z_.size())
# fixed_z_ = fixed_z_.view(-1, 100, 1, 1)
# print(fixed_z_.size())
#
#
# print(fixed_y_)
# print(fixed_y_.size())
#
# fixed_y_label_ = torch.zeros(100, 10)
# fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
#
# print(fixed_y_label_)
# print(fixed_y_label_.size())

# for i in range(10):
#     print(fixed_z_[i])
#     print(fixed_y_[:,i])
# print(fixed_z_.size())
# print(fixed_y_.size())
# print(fixed_y_)

img_size=32
# label preprocess
onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10, 1), 1).view(10, 10, 1, 1)
# print(onehot)
fill = torch.zeros([10, 10, img_size, img_size])
for i in range(10):
    fill[i, i, :, :] = 1

# print(fill[0, 0, ...])
# print(fill[0, 0, ...].size())
print(fill[1][1][:][:])
print(fill[0].size())

mini_batch=64
print((torch.rand(mini_batch, 1) *10 ))