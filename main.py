import config
import os
import cv2
import torch
import numpy as np
import jittor as jt
from CustomDataset import CustomDataset
from models import generator, discriminator
from models import generator1, discriminator1, losses
from utils.utils import *
import jittor.nn as nn

jt.flags.use_cuda = 1

# load options
opt = config.read_arguments(train=True)

# dataloader
dataloader = CustomDataset(opt).set_attrs(
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
    )
length = len(dataloader)

# path to save models.
path = os.path.join(opt.checkpoints_dir, opt.name, "models")
os.makedirs(path, exist_ok=True)
opt.output_path = 'results'      # path to save the image
os.makedirs(opt.output_path, exist_ok=True)

# create models
# model = models.GAN_model(opt)

losses_computer = losses.losses_computer(opt)

# define the model
netG = generator1.OASIS_model(opt)
netD = discriminator1.Discriminator(opt)

# optimizerss
optimizerG = jt.optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
optimizerD = jt.optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))

# ---- multi-scale training ----
size_rates = [0.75, 1, 1.25]

for epoch in range(opt.num_epochs):
    for i, data_i in enumerate(dataloader):
        # multi-scale training.
        for rate in size_rates:
            opt.rate = rate
            image, label = preprocess_input(opt, data_i)
            trainsize = int(round(opt.img_height*rate/32)*32)
            opt.tsize = trainsize
            if opt.rate != 1:
                image = nn.upsample(image, size=(opt.tsize, opt.tsize), mode='bilinear', align_corners=True)
                label = nn.upsample(label, size=(opt.tsize, opt.tsize), mode='bilinear', align_corners=True)
        
            # Train the discriminator
            start_grad(netD)
            with jt.no_grad():
                fake = netG(label)
                if opt.rate != 1:
                    fake = nn.upsample(fake, size=(opt.tsize, opt.tsize), mode='bilinear', align_corners=True)
            output_D_fake = netD(fake)
            if opt.rate != 1:
                output_D_fake = nn.upsample(output_D_fake, size=(opt.tsize, opt.tsize), mode='bilinear', align_corners=True)
            loss_D_fake = losses_computer.loss(output_D_fake, label, for_real=False)
            output_D_real = netD(image)
            if opt.rate != 1:
                output_D_real = nn.upsample(output_D_real, size=(opt.tsize, opt.tsize), mode='bilinear', align_corners=True)
            loss_D_real = losses_computer.loss(output_D_real, label, for_real=True)
            # labelMix
            mixed_inp, mask = generate_labelmix(label, fake, image)
            output_D_mixed = netD(mixed_inp)
            if opt.rate != 1:
                output_D_mixed = nn.upsample(output_D_mixed, size=(opt.tsize, opt.tsize), mode='bilinear', align_corners=True)
            loss_D_lm = opt.lambda_labelmix * losses_computer.loss_labelmix(mask, output_D_mixed, output_D_fake,
                                                                                output_D_real)
            loss_D = loss_D_fake + loss_D_real + loss_D_lm
            losses_D_list = [loss_D_fake, loss_D_real, loss_D_lm]
            loss_D, losses_D_list = loss_D.mean(), [loss.mean() if loss is not None else None for loss in losses_D_list]
            optimizerD.step(loss_D)

            # Train the generator
            stop_grad(netD)
            start_grad(netG)
            # contrastive learning
            fake_2 = netG(label, flag=True)
            output_D2 = netD(fake_2)

            fake = netG(label)
            output_D = netD(fake)
            output_D1 = nn.interpolate(output_D, scale_factor=0.5, mode='bilinear', align_corners=True)
            loss_er = jt.mean(jt.abs(output_D1[:, 1, :, :] - output_D2[:, 1, :, :]))
            if opt.rate != 1:
                output_D = nn.upsample(output_D, size=(opt.tsize, opt.tsize), mode='bilinear', align_corners=True)
            loss_G = losses_computer.loss(output_D, label, for_real=True)
            # VGG loss remaining added.
            loss_G = loss_G.mean() + loss_er
            optimizerG.step(loss_G)

            jt.sync_all(True)
        # jt.gc()
        # output loss and other
        if i % 10 == 0:
            print("Epoch[%d:%d], Batch[%d/%d], Loss_G: %f, Loss_D: %f" % (epoch, opt.batch_size, i, length, loss_G, loss_D))

    if (epoch + 1) % 5 == 0:
        netD.save(os.path.join(f"{path}/{epoch+1}_D.pkl"))
        netG.save(os.path.join(f"{path}/{epoch+1}_G.pkl"))

    with jt.no_grad():
        image, label = preprocess_input(opt, data_i)
        generated = netG(label)
        photo_id = data_i['name']
        img_t = np.array((generated + 1) / 2 * 255).astype('uint8')
        for idx in range(image.shape[0]):
            cv2.imwrite(f"{opt.output_path}/{photo_id[idx]}.jpg", img_t[idx].transpose(1, 2, 0)[:, :, ::-1])

print("The training has successfully finished.")