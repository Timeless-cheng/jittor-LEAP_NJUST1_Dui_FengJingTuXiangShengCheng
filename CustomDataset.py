from jittor.dataset.dataset import Dataset
import jittor.transform as transform
import os
import glob
import numpy as np
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        opt.crop_size = 256
        opt.label_nc = 29
        opt.contain_dontcare_label = True
        opt.semantic_nc = 30  # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0
        self.opt = opt

        if self.opt.phase == 'train':
            self.images = sorted(glob.glob(os.path.join(self.opt.input_path, self.opt.phase, "imgs") + "/*.*"))
            self.labels = sorted(glob.glob(os.path.join(self.opt.input_path, self.opt.phase, "labels") + "/*.*"))
        elif self.opt.phase == 'test':
            self.labels = sorted(glob.glob(opt.input_path + "/*.*"))

        print(f"from {self.opt.phase} split load {len(self.labels)} images.")

    def __len__(self,):
        return len(self.labels)

    def __getitem__(self, idx):
        label = Image.open(self.labels[idx % len(self.labels)])
        id = self.labels[idx % len(self.labels)].split('/')[-1][:-4]  # 保存图片名称。

        label = Image.fromarray(np.array(label).astype("uint8"))
        label = transform.resize(label, (self.opt.img_height, self.opt.img_width), Image.NEAREST)
        label = transform.to_tensor(label)
        label = label * 255

        if self.opt.phase == 'train':
            #image transforms
            image = Image.open(self.images[idx % len(self.images)])
            image = transform.resize(image, (self.opt.img_height, self.opt.img_width), Image.BICUBIC)
            image = transform.to_tensor(image)
            image = transform.image_normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            image = label.copy()

        return {"image": image, "label": label, "name": id}