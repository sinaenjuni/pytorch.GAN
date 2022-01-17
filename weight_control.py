import torch
import torch.nn as nn

resnet_s_PATH = f'weights/experiments2/Resnet_s/GAN/D_10.pth'
resnet_s_weights = torch.load(resnet_s_PATH)

resnet_tade_PATH = "/home/sin/git/pytorch.GAN/weights/experiments3/Resnet_tade/classifier/model.pth"
resnet_tade_weight = torch.load(resnet_tade_PATH)

# model.load_state_dict(torch.load(SAVE_PATH), strict=False)
if __name__ == "__main__":
    for i in resnet_tade_weight.keys():
        print(i)
