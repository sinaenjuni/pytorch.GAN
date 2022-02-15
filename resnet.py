import torch
import torch.nn.functional as F
import torch.nn as nn
import ops

def linear(in_features, out_features, bias=True):
    return nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, g_cond_mtd, g_info_injection, affine_input_dim):
        super(GenBlock, self).__init__()
        self.g_cond_mtd = g_cond_mtd
        self.g_info_injection = g_info_injection

        if self.g_cond_mtd == "W/O" and self.g_info_injection in ["N/A", "concat"]:
            self.bn1 = ops.batchnorm_2d(in_features=in_channels)
            self.bn2 = ops.batchnorm_2d(in_features=out_channels)
        elif self.g_cond_mtd == "cBN" or self.g_info_injection == "cBN":
            self.bn1 = ops.ConditionalBatchNorm2d(affine_input_dim, in_channels)
            self.bn2 = ops.ConditionalBatchNorm2d(affine_input_dim, out_channels)
        else:
            raise NotImplementedError

        self.activation = nn.ReLU(inplace=True)
        self.conv2d0 = ops.conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2d1 = ops.conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = ops.conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, affine):
        x0 = x
        if self.g_cond_mtd == "W/O" and self.g_info_injection in ["N/A", "concat"]:
            x = self.bn1(x)
        elif self.g_cond_mtd == "cBN" or self.g_info_injection == "cBN":
            x = self.bn1(x, affine)
        else:
            raise NotImplementedError
        x = self.activation(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv2d1(x)

        if self.g_cond_mtd == "W/O" and self.g_info_injection in ["N/A", "concat"]:
            x = self.bn2(x)
        elif self.g_cond_mtd == "cBN" or self.g_info_injection == "cBN":
            x = self.bn2(x, affine)
        else:
            raise NotImplementedError
        x = self.activation(x)
        x = self.conv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode="nearest")
        x0 = self.conv2d0(x0)
        out = x + x0
        return out



class Generator(nn.Module):
    def __init__(self, z_dim, g_shared_dim, img_size, g_conv_dim, apply_attn, attn_g_loc, g_cond_mtd, num_classes, g_init, g_depth,
                 mixed_precision):
        super(Generator, self).__init__()
        g_in_dims_collection = {
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "128": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "256": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "512": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim]
        }

        g_out_dims_collection = {
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "128": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "256": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "512": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim, g_conv_dim]
        }

        bottom_collection = {"32": 4, "64": 4, "128": 4, "256": 4, "512": 4}

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.g_cond_mtd = g_cond_mtd
        self.mixed_precision = mixed_precision
        self.in_dims = g_in_dims_collection[str(img_size)]
        self.out_dims = g_out_dims_collection[str(img_size)]
        self.bottom = bottom_collection[str(img_size)]
        self.num_blocks = len(self.in_dims)
        self.affine_input_dim = 0
        self.info_type = "N/A"

        info_dim = 0
        # if self.MODEL.info_type in ["discrete", "both"]:
        #     info_dim += self.MODEL.info_num_discrete_c*self.MODEL.info_dim_discrete_c
        # if self.MODEL.info_type in ["continuous", "both"]:
        #     info_dim += self.MODEL.info_num_conti_c

        self.g_info_injection = "N/A"
        # if self.MODEL.info_type != "N/A":
        #     if self.g_info_injection == "concat":
        #         self.info_mix_linear = ops.linear(in_features=self.z_dim + info_dim, out_features=self.z_dim, bias=True)
        #     elif self.g_info_injection == "cBN":
        #         self.affine_input_dim += self.z_dim
        #         self.info_proj_linear = ops.linear(in_features=info_dim, out_features=self.z_dim, bias=True)

        # self.linear0 = MODULES.g_linear(in_features=self.z_dim, out_features=self.in_dims[0] * self.bottom * self.bottom, bias=True)
        self.linear0 = linear(in_features=self.z_dim, out_features=self.in_dims[0] * self.bottom * self.bottom, bias=True)

        if self.g_cond_mtd != "W/O":
            self.affine_input_dim += self.z_dim
            self.shared = ops.embedding(num_embeddings=self.num_classes, embedding_dim=self.z_dim)

        self.blocks = []
        for index in range(self.num_blocks):
            self.blocks += [[
                GenBlock(in_channels=self.in_dims[index],
                         out_channels=self.out_dims[index],
                         g_cond_mtd=self.g_cond_mtd,
                         g_info_injection=self.g_info_injection,
                         affine_input_dim=self.affine_input_dim)
            ]]

            if index + 1 in attn_g_loc and apply_attn:
                self.blocks += [[ops.SelfAttention(self.out_dims[index], is_generator=True)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.bn4 = ops.batchnorm_2d(in_features=self.out_dims[-1])
        self.activation = nn.ReLU(inplace=True)
        self.conv2d5 = ops.conv2d(in_channels=self.out_dims[-1], out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        ops.init_weights(self.modules, g_init)

    def forward(self, z, label, shared_label=None, eval=False):
        affine_list = []
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else misc.dummy_context_mgr() as mp:
            if self.info_type != "N/A":
                if self.g_info_injection == "concat":
                    z = self.info_mix_linear(z)
                elif self.g_info_injection == "cBN":
                    z, z_info = z[:, :self.z_dim], z[:, self.z_dim:]
                    affine_list.append(self.info_proj_linear(z_info))

            if self.g_cond_mtd != "W/O":
                if shared_label is None:
                    shared_label = self.shared(label)
                affine_list.append(shared_label)
            if len(affine_list) > 0:
                affines = torch.cat(affine_list, 1)
            else:
                affines = None

            act = self.linear0(z)
            act = act.view(-1, self.in_dims[0], self.bottom, self.bottom)
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    if isinstance(block, ops.SelfAttention):
                        act = block(act)
                    else:
                        act = block(act, affines)

            act = self.bn4(act)
            act = self.activation(act)
            act = self.conv2d5(act)
            out = self.tanh(act)
        return out

class make_empty_object(object):
    pass

if __name__ == "__main__":
    z_dim = 128
    g_shared_dim = "N/A"
    img_size = 32
    g_conv_dim = 64
    d_conv_dim = 64
    apply_attn = False
    attn_g_loc = ["N/A"]
    g_cond_mtd = "W/O"
    num_classes = 10
    g_init = "ortho"
    g_depth = "N/A"

    Gen = Generator(z_dim=z_dim,
                    g_shared_dim=g_shared_dim,
                    img_size=img_size,
                    g_conv_dim=g_conv_dim,
                    apply_attn=apply_attn,
                    attn_g_loc=attn_g_loc,
                    g_cond_mtd=g_cond_mtd,
                    num_classes=num_classes,
                    g_init=g_init,
                    g_depth=g_depth,
                    mixed_precision=make_empty_object)

    print(Gen)

    input_tensor = torch.rand((64, 128))
    label = torch.rand((64, 10))
    output = Gen(input_tensor, label)
    print(output)
    # Gen_mapping, Gen_synthesis = None, None