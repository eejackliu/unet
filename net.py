import torch
import torch.nn as nn
import numpy as np
import torchvision as tv
from torchvision import transforms
from torch.utils.data import DataLoader as dataloader
import nonechucks as nc
from voc_seg import my_data
model=tv.models.vgg19_bn(pretrained=True)

image_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
train_data=my_data((320,240),root='/home/local/SPREADTRUM/gary.liu/Documents/cs231/seg_1224/data',transform=image_transform)
test_data=my_data((320,240),root='/home/local/SPREADTRUM/gary.liu/Documents/cs231/seg_1224/data',image_set='val',transform=image_transform)
trainset=nc.SafeDataset(train_data)
testset=nc.SafeDataset(test_data)
trainload=nc.SafeDataLoader(trainset,batch_size=4,shuffle=True)
testload=nc.SafeDataLoader(testset,batch_size=4)

class unet(nn.Module):
    def __init__(self):
        super(unet,self).__init__()
        self.conv1=nn.Sequential(model.features[0],
                                 model.features[1],
                                 nn.ReLU(inplace=True),
                                 model.features[3],
                                 model.features[4],
                                 nn.ReLU(inplace=True),
                                )
        self.pool1=model.features[6]
        self.conv2=nn.Sequential(model.features[7],
                                 model.features[8],
                                 nn.ReLU(inplace=True),
                                 model.features[10],
                                 model.features[11],
                                 nn.ReLU(inplace=True),
                                )
        self.pool2=model.features[13]
        self.conv3=nn.Sequential(model.features[14],
                                 model.features[15],
                                 nn.ReLU(inplace=True),
                                 model.features[17],
                                 model.features[18],
                                 nn.ReLU(inplace=True),
                                )
        self.pool3=model.features[26]
        self.conv4=nn.Sequential(model.features[27],
                                 model.features[28],
                                 nn.ReLU(inplace=True),
                                 model.features[30],
                                 model.features[31],
                                 nn.ReLU(inplace=True),)
        self.pool4=model.features[39]
        self.conv5=nn.Sequential(nn.Conv2d(512,1024,3,padding=1),
                                 nn.BatchNorm2d(1024),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(1024,1024,3,padding=1),
                                 nn.BatchNorm2d(1024),
                                 nn.ReLU(inplace=True))

