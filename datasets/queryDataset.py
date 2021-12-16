import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image

class Dataset_query(Dataset):
    def __init__(self,filename,transformer,basedir):
        super(Dataset_query, self).__init__()
        self.filename = filename
        self.transformer = transformer
        self.basedir = basedir

    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self)


class Query_transforms(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, pad=20,size=256):
        self.pad=pad
        self.size = size

    def __call__(self, img):
        if self.pad<=0:
            return img
        img_=np.array(img).copy()
        img_part = img_[:,0:self.pad,:]
        img_pad = np.zeros_like(img_part,dtype=np.uint8)
        image = np.concatenate((img_pad, img_),axis=1)
        image = image[:,0:self.size,:]
        image = Image.fromarray(image.astype('uint8')).convert('RGB')


        # img_=np.array(img).copy()
        # img_part = img_[:,0:self.pad,:]
        # img_flip = cv2.flip(img_part, 1)  # 镜像
        # image = np.concatenate((img_flip, img_),axis=1)
        # image = image[:,0:self.size,:]
        # image = Image.fromarray(image.astype('uint8')).convert('RGB')
        return image