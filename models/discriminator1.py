import jittor as jt
from jittor import nn
from utils.spectral_norm_new import SpectralNorm as spectral_norm


class Discriminator(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = spectral_norm
        output_channel = (opt.semantic_nc + 1)  # for N+1 loss
        self.channels = [3, 128, 128, 256, 256, 512, 512]
        self.body_up = nn.ModuleList([])
        self.body_down = nn.ModuleList([])
        for i in range(opt.num_res_blocks):
            self.body_down.append(
                residual_block_D(self.channels[i], self.channels[(i + 1)], opt, -1, first=(i == 0)))
        self.body_up.append(residual_block_D(self.channels[-1], self.channels[-2], opt, 1))
        for i in range(1, opt.num_res_blocks - 1):
            self.body_up.append(residual_block_D((2 * self.channels[- 1 - i]), self.channels[- 2 - i], opt, 1))
        self.body_up.append(residual_block_D((2 * self.channels[1]), 64, opt, 1))
        self.layer_up_last = nn.Conv(64, output_channel, 1, stride=1, padding=0)

    def execute(self, input):
        x = input
        # encoder
        encoder_res = list()
        for i in range(len(self.body_down)):
            x = self.body_down[i](x)
            encoder_res.append(x)
        # decoder
        x = self.body_up[0](x)
        for i in range(1, len(self.body_down)):
            x = self.body_up[i](jt.contrib.concat((encoder_res[- i - 1], x), dim=1))
        ans = self.layer_up_last(x)
        return ans


class residual_block_D(nn.Module):

    def __init__(self, fin, fout, opt, up_or_down, first=False):
        super().__init__()
        # Attributes
        self.up_or_down = up_or_down
        self.first = first
        self.learned_shortcut = (fin != fout)
        fmiddle = fout
        norm_layer = spectral_norm
        if first:
            self.conv1 = nn.Sequential(norm_layer(nn.Conv(fin, fmiddle, 3, stride=1, padding=1)))
        elif self.up_or_down > 0:
            self.conv1 = nn.Sequential(nn.LeakyReLU(scale=0.2), nn.Upsample(scale_factor=2),
                                       norm_layer(nn.Conv(fin, fmiddle, 3, stride=1, padding=1)))
        else:
            self.conv1 = nn.Sequential(nn.LeakyReLU(scale=0.2),
                                       norm_layer(nn.Conv(fin, fmiddle, 3, stride=1, padding=1)))
        self.conv2 = nn.Sequential(nn.LeakyReLU(scale=0.2), norm_layer(nn.Conv(fmiddle, fout, 3, stride=1, padding=1)))
        if self.learned_shortcut:
            self.conv_s = norm_layer(nn.Conv(fin, fout, 1, stride=1, padding=0))
        if up_or_down > 0:
            self.sampling = nn.Upsample(scale_factor=2)
        elif up_or_down < 0:
            self.sampling = nn.AvgPool2d(2)
        else:
            self.sampling = nn.Sequential()

    def execute(self, x):
        x_s = self.shortcut(x)
        dx = self.conv1(x)
        dx = self.conv2(dx)
        if self.up_or_down < 0:
            dx = self.sampling(dx)
        out = x_s + dx
        return out

    def shortcut(self, x):
        if self.first:
            if self.up_or_down < 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            x_s = x
        else:
            if self.up_or_down > 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            if self.up_or_down < 0:
                x = self.sampling(x)
            x_s = x
        return x_s
