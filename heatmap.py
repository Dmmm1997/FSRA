import os

import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from tool.utils_server import load_network
import yaml
import argparse
import torch
from torchvision import datasets, models, transforms
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser(description='Training')
import math

parser.add_argument('--data_dir',default='/home/dmmm/University-Release/test',type=str, help='./test_data')
parser.add_argument('--name', default='from_transreid_256_4B_small_lr005_kl', type=str, help='save model path')
parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
parser.add_argument('--checkpoint',default="net_119.pth", help='weights' )
opt = parser.parse_args()

config_path = 'opts.yaml'
with open(config_path, 'r') as stream:
    config = yaml.load(stream)
opt.stride = config['stride']
opt.views = config['views']
opt.transformer = config['transformer']
opt.pool = config['pool']
opt.LPN = config['LPN']
opt.block = config['block']
opt.nclasses = config['nclasses']
opt.droprate = config['droprate']
opt.share = config['share']

if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']
if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else:
    opt.nclasses = 751


def heatmap2d(img, arr):
    # fig = plt.figure()
    # ax0 = fig.add_subplot(121, title="Image")
    # ax1 = fig.add_subplot(122, title="Heatmap")
    # fig, ax = plt.subplots(）
    # ax[0].imshow(Image.open(img))
    plt.figure()
    heatmap = plt.imshow(arr, cmap='viridis')
    plt.axis('off')
    # fig.colorbar(heatmap, fraction=0.046, pad=0.04)
    #plt.show()
    plt.savefig('heatmap_dbase')

data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

model = load_network(opt)

print(opt.data_dir)
for i in ["0009","0013","0015","0016","0018","0035","0039","0116","0130"]:
    print(i)
    imgpath = os.path.join(opt.data_dir,"gallery_drone/"+i)
    imgpath = os.path.join(imgpath, "image-28.jpeg")
    print(imgpath)
    img = Image.open(imgpath)
    img = data_transforms(img)
    img = torch.unsqueeze(img,0)


    model = model.eval().cuda()

    with torch.no_grad():
        # x = model.model_3.model.conv1(img.cuda())
        # x = model.model_3.model.bn1(x)
        # x = model.model_3.model.relu(x)
        # x = model.model_3.model.maxpool(x)
        # x = model.model_3.model.layer1(x)
        # x = model.model_3.model.layer2(x)
        # x = model.model_3.model.layer3(x)
        # output = model.model_3.model.layer4(x)
        features,_ = model.model_1.transformer(img.cuda())
        pos_embed = model.model_1.transformer.pos_embed
        part_features = features[:,1:]
        part_features = part_features.view(part_features.size(0),int(math.sqrt(part_features.size(1))),int(math.sqrt(part_features.size(1))),part_features.size(2))
        output = part_features.permute(0,3,1,2)

    heatmap = output.squeeze().sum(dim=0).cpu().numpy()
    # print(heatmap.shape)
    # print(heatmap)
    # heatmap = np.mean(heatmap, axis=0)
    #
    # heatmap = np.maximum(heatmap, 0)
    heatmap = normalization(heatmap)
    img = cv2.imread(imgpath)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, 2)  # 将热力图应用于原始图像model.py
    superimposed_img = heatmap * 0.8 + img  # 这里的0.4是热力图强度因子
    if not os.path.exists("heatout"):
        os.mkdir("./heatout")
    cv2.imwrite("./heatout/"+i+".jpg", superimposed_img)