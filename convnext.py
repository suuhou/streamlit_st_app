import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class ConvNeXtGenerator(nn.Module):

    def __init__(self, input_nc=9):
        super(ConvNeXtGenerator, self).__init__()


        self.first = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, 7, padding=0), nn.ReLU(True))
        ### downsample
        self.down = downsampling_layers(input_nc=64, dims=[128,256,512,1024],
                                        drop_path_rate=0., depths=[1,1,1,1], layer_scale_init_value=1e-6) #0,0,0,0   1,1,1,1  3,3,4,3

        ### map blocks
        self.map = mapping_layers(1024, 3)

        ### upsample
        self.up = upsampling_layers(dims=[512,256,128,64], drop_path_rate=0.,
                                    depths=[1,1,1,1], layer_scale_init_value=1e-6)

        self.last = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 3, (7, 7), padding=0), nn.Tanh())


    def forward(self, input):

        input = self.first(input)
        real_low = self.down(input)
        fake_low = self.map(real_low[-1])
        fake = self.up(fake_low, real_low)
        out = self.last(fake)

        return out

class downsampling_layers(nn.Module):

    def __init__(self, input_nc, dims, drop_path_rate, depths, layer_scale_init_value):
        super(downsampling_layers, self).__init__()

        self.bulid_layers(input_nc, dims, drop_path_rate, depths, layer_scale_init_value)

    def bulid_layers(self, input_nc, dims, drop_path_rate, depths, layer_scale_init_value):

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(nn.Conv2d(input_nc, dims[0], (3,3), (2,2),1), nn.ReLU(True))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                #LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1), nn.ReLU(True))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value,
                        remove_norm = i==0 ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

    def forward(self, x):

        downsampled_x = []
        for down_layer, convnext_blocks in zip(self.downsample_layers, self.stages):
            x = convnext_blocks(down_layer(x))
            downsampled_x.append(x)

        return downsampled_x

class mapping_layers(nn.Module):

    def __init__(self, input_nc, num_layers):
        super(mapping_layers, self).__init__()

        self.mapping_layers = nn.ModuleList()
        self.input_nc = input_nc
        self.num_layers = num_layers

        self.build_layers()

    def build_layers(self):

        for _ in range(self.num_layers):
            self.mapping_layers.append(
                nn.Sequential(nn.Conv2d(self.input_nc, self.input_nc, (1, 1), (1, 1)), nn.LeakyReLU(0.02, True)))

    def forward(self, x):

        for layer in self.mapping_layers:
            x = layer(x)

        return x


class upsampling_layers(nn.Module):

    def __init__(self, depths, drop_path_rate, dims, layer_scale_init_value):
        super(upsampling_layers, self).__init__()

        self.up = []
        self.up += [DeConvBlock(1024,512, inner=True)]
        self.up += [DeConvBlock(1024, 256)]
        self.up += [DeConvBlock(512, 128)]
        self.up += [DeConvBlock(256, 64, outer=True)]

        self.up = nn.Sequential(*self.up)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value, remove_norm = i==0 ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

    def forward(self, x, downsampled_xs):

        index = [i for i in range(4)]
        for i, layer, convnext_block in zip(index, self.up, self.stages):
            x = convnext_block(layer(x, downsampled_xs[3-i]))

        return x

class DeConvBlock(nn.Module):
    def __init__(self, in_nc, out_nc, inner=False, outer=False):
        super(DeConvBlock, self).__init__()

        self.inner = inner
        self.outer = outer
        self.deconv_layer = self.bulid_layer(in_nc, out_nc)

    def bulid_layer(self, in_nc, out_nc):

        deconv = []
        deconv += [nn.Upsample(scale_factor=2)] #if not self.outer else [nn.Upsample([512,512])]
        deconv += [nn.ReflectionPad2d(1)]
        deconv += [nn.Conv2d(in_nc, out_nc, (3,3), (1,1), padding=0)]
        #deconv += [nn.InstanceNorm2d(out_nc)]
        #deconv += [LayerNorm(out_nc, eps=1e-6, data_format="channels_first")] if not self.inner else []
        deconv += [nn.ReLU(True)]
        #deconv += [nn.LeakyReLU(0.02, True)]

        return nn.Sequential(*deconv)

    def forward(self, x, downsampled_x):

        if self.inner:
            out = self.deconv_layer(x)
        else:
            out = torch.cat([x, downsampled_x], dim=1)
            out = self.deconv_layer(out)

        return out

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, remove_norm = False):
        super().__init__()
        self.dwconv = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(dim, dim, kernel_size=7, padding=0, groups=dim)) # depthwise conv
        self.norm = nn.Identity() if remove_norm else LayerNorm(dim, eps=1e-6)  ########
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):

        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)

        return x

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


if __name__ == '__main__':

    net = ConvNeXtGenerator().cuda()

    x = torch.rand([1, 9, 1520, 992]).cuda()
    y = net(x)

    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    print('[Network %s] Total number of parameters : %.3f M' % ('convnext', num_params / 1e6))
    print(y.shape)

    # net = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
    # x = torch.rand([1,3,512,512])
    # y = net(x)
    # print(y.shape)
    #
    # num_params = 0
    # for param in net.parameters():
    #     num_params += param.numel()
    # print('[Network %s] Total number of parameters : %.3f M' % ('convnext', num_params / 1e6))

    # d = downsampling_layers(input_nc=3, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024],
    #                         drop_path_rate=0., layer_scale_init_value=1e-6)
    # x = torch.rand([1,3,512,512])
    # out = d(x)
    # for y in out:
    #     print(y.shape)

