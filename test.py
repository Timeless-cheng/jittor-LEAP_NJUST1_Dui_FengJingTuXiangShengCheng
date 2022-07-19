import config
import jittor as jt
import jittor.nn as nn
from CustomDataset import CustomDataset
from models import generator1
import cv2
import os
import torch
import numpy as np
from utils.utils import preprocess_input

jt.flags.use_cuda = 1

# load options
opt = config.read_arguments(train=False)
opt.phase = 'test'
# dataloader
dataloader = CustomDataset(opt).set_attrs(
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0
    )

netG = generator1.OASIS_model(opt)
netG.load('checkpoints/best_G.pkl')

os.makedirs(opt.output_path, exist_ok=True)

for i, batch in enumerate(dataloader):
    print(i)
    _, label = preprocess_input(opt, batch)
    generated = netG(label)
    photo_id = batch["name"]

    img_t = jt.Var(generated)
    img_t = nn.upsample(img_t, size=(384, 512), mode='bilinear', align_corners=False)
    img_t = np.array((img_t + 1) / 2 * 255).astype('uint8')
    for idx in range(img_t.shape[0]):
        cv2.imwrite(f"{opt.output_path}/{photo_id[idx]}.jpg", img_t[idx].transpose(1, 2, 0)[:, :, ::-1])
