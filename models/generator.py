import jittor as jt
from jittor import nn
# from utils.spectral_norm import spectral_norm
from utils.spectral_norm_new import SpectralNorm as spectral_norm
from jittor.nn import InstanceNorm


class SPADE(nn.Module):

    def __init__(self, opt, norm_nc, label_nc):
        super().__init__()
        self.first_norm = get_norm_layer(opt, norm_nc)
        ks = opt.spade_ks
        nhidden = 128
        pw = (ks // 2)
        self.mlp_shared = nn.Sequential(nn.Conv(label_nc, nhidden, ks, padding=pw), nn.ReLU())
        self.mlp_gamma = nn.Conv(nhidden, norm_nc, ks, padding=pw)
        self.mlp_beta = nn.Conv(nhidden, norm_nc, ks, padding=pw)

    def execute(self, x, segmap):
        normalized = self.first_norm(x)
        segmap = nn.interpolate(segmap, size=x.shape[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = (normalized * (1 + gamma)) + beta
        return out


class OASIS_model(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = InstanceNorm
        ch = opt.channels_G
        self.channels = [(16 * ch), (16 * ch), (16 * ch), (8 * ch), (4 * ch), (2 * ch), (1 * ch)]
        (self.init_W, self.init_H) = self.compute_latent_vector_size(opt)
        self.conv_img = nn.Conv(self.channels[-1], 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        self.body = nn.ModuleList([])
        for i in range((len(self.channels) - 1)):
            self.body.append(ResnetBlock_with_SPADE(self.channels[i], self.channels[(i + 1)], opt))
        if (not self.opt.no_3dnoise):
            self.fc = nn.Conv((self.opt.semantic_nc + self.opt.z_dim), (16 * ch), 3, padding=1)
        else:
            self.fc = nn.Conv(self.opt.semantic_nc, (16 * ch), 3, padding=1)

    def compute_latent_vector_size(self, opt):
        w = (opt.crop_size // (2 ** (opt.num_res_blocks - 1)))
        h = round((w / opt.aspect_ratio))
        return (h, w)

    def execute(self, input, z=None):
        seg = input
        #print("input",input.shape)
        if (not self.opt.no_3dnoise):
            z = jt.randn(seg.shape[0], self.opt.z_dim)
            z = z.view((z.shape[0], self.opt.z_dim, 1, 1))
            z = z.expand(z.shape[0], self.opt.z_dim, seg.shape[2], seg.shape[3])
            seg = jt.contrib.concat((z, seg), dim=1)
        x = nn.interpolate(seg, size=(self.init_W, self.init_H))
        x = self.fc(x)
        for i in range(self.opt.num_res_blocks):
            x = self.body[i](x, seg)
            if i < (self.opt.num_res_blocks - 1):
                x = self.up(x)
        x = self.conv_img(nn.leaky_relu(x, 0.2))
        x = jt.tanh(x)
        return x


class ResnetBlock_with_SPADE(nn.Module):

    def __init__(self, fin, fout, opt):
        super().__init__()
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = spectral_norm
        self.conv_0 = sp_norm(nn.Conv(fin, fmiddle, 3, padding=1))
        self.conv_1 = sp_norm(nn.Conv(fmiddle, fout, 3, padding=1))
        # self.conv_0 = nn.Conv(fin, fmiddle, 3, padding=1)
        # self.conv_1 = nn.Conv(fmiddle, fout, 3, padding=1)
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv(fin, fout, 1, bias=False))
            # self.conv_s = nn.Conv(fin, fout, 1, bias=False)

        spade_conditional_input_dims = opt.semantic_nc
        if (not opt.no_3dnoise):
            spade_conditional_input_dims += opt.z_dim
        self.norm_0 = SPADE(opt, fin, spade_conditional_input_dims)
        self.norm_1 = SPADE(opt, fmiddle, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = SPADE(opt, fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(scale=0.2)

    def execute(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        out = x_s + dx
        return out


def get_norm_layer(opt, norm_nc):
    if opt.param_free_norm == 'instance':
        return nn.InstanceNorm2d(norm_nc, affine=False)
    # if opt.param_free_norm == 'syncbatch':
    #     return SynchronizedBatchNorm2d(norm_nc, affine=False)
    if opt.param_free_norm == 'batch':
        return nn.BatchNorm2d(norm_nc, affine=False)
    else:
        raise ValueError('%s is not a recognized param-free norm type in SPADE'
                         % opt.param_free_norm)


if __name__ == '__main__':
    import config

    opt = config.read_arguments()
    opt.crop_size = 256
    opt.aspect_ratio = 1.0
    opt.semantic_nc = 30

    model = OASIS_model(opt)

    from PIL import Image
    import numpy as np
    import jittor.transform as transform

    root = '../datasets/train/labels/48432_b67ec6cd63_b.png'
    label = Image.open(root)
    label = Image.fromarray(np.array(label).astype("uint8"))
    label = transform.resize(label, (opt.img_height, opt.img_width), Image.NEAREST)
    label = transform.to_tensor(label)
    label = label * 255
    label_map = jt.Var(label).long().view((1, 1, 256, 256))
    bs, _, h, w = label_map.size()
    nc = opt.semantic_nc
    input_label = jt.float32(jt.zeros((bs, nc, h, w)))  # bs,30,256,256
    src = jt.float32(jt.ones((bs, nc, h, w)))
    input_semantics = input_label.scatter_(1, label_map, src)

    fake = model(input_semantics)

    print(fake)
