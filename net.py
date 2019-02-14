import torch
import torch.nn as nn
import numpy as np
import torchvision as tv
from torchvision import transforms
from torch.utils.data import DataLoader as dataloader
import nonechucks as nc
from voc_seg import my_data,label_acc_score
model=tv.models.vgg19_bn(pretrained=True)

image_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
train_data=my_data((320,240),root='/home/local/SPREADTRUM/gary.liu/Documents/cs231/seg_1224/data',transform=image_transform)
test_data=my_data((320,240),root='/home/local/SPREADTRUM/gary.liu/Documents/cs231/seg_1224/data',image_set='val',transform=image_transform)
trainset=nc.SafeDataset(train_data)
testset=nc.SafeDataset(test_data)
trainload=nc.SafeDataLoader(trainset,batch_size=4,shuffle=True)
testload=nc.SafeDataLoader(testset,batch_size=4)
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')
dtype=torch.float32
class unet(nn.Module):

    def __init__(self):
        super(unet,self).__init__()
        num_class = 21
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
                                 # nn.Dropout2d(),
                                 nn.Conv2d(1024,1024,3,padding=1),
                                 nn.BatchNorm2d(1024),
                                 nn.ReLU(inplace=True),
                                 # nn.Dropout2d(),
                                 nn.Conv2d(1024,512,1),
                                 nn.UpsamplingBilinear2d(scale_factor=2)
                                 )

        self.deconv4=nn.Sequential(nn.Conv2d(1024,512,3,padding=1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512,512,3,padding=1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512,256,1),         #since unsampling cann't change the channel num ,have to change channel num before next block
                                   nn.UpsamplingBilinear2d(scale_factor=2)

                                   ) # try to use upsampling instead of transpose conv
        self.deconv3=nn.Sequential(nn.Conv2d(512,256,3,padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256,256,3,padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256,128,1),
                                   nn.UpsamplingBilinear2d(scale_factor=2))
        self.deconv2=nn.Sequential(nn.Conv2d(256,128,3,padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128,128,3,padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128,64,1),
                                   nn.UpsamplingBilinear2d(scale_factor=2))
        self.deconv1=nn.Sequential(nn.Conv2d(128,64,3,padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64,64,3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64,num_class,1))
    def forward(self, input):
        down1=self.conv1(input)
        down2=self.conv2(down1)
        down3=self.conv3(down2)
        down4=self.conv4(down3)
        down5=self.conv5(down4)
        up1=self.deconv4(torch.cat((down4,down5),dim=1))
        up2=self.deconv3(torch.cat((up1,down3),dim=1))
        up3=self.deconv2(torch.cat((up2,down2),dim=1))
        up4=self.deconv1(torch.cat((up3,down1),dim=1))
        return up4

def train(model,epoch):
    model.train()
    model.to(device)
    criterion=nn.CrossEntropyLoss()
    optimize=torch.optim.Adam(model.parameters(),lr=0.001)
    for i in range(epoch):
        tmp=0
        for image,mask in trainload:
            image,mask=image.to(device,dtype=dtype),mask.to(device,dtype=dtype)
            optimize.zero_grad()
            pred=model(image)
            loss=criterion(pred,mask)
            loss.backward()
            optimize.step()
            tmp=loss.data
            print "loss ",tmp
        print "{0} epoch ,loss is{1}".format(i,tmp)
    return model

def test(model):

    with torch.no_grad():
        model.eval()
        model.to(device)
        for image,mask in testload:
            image=image.to(device,dtype=dtype)



model=unet()
model.train()