import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torchvision as tv
import math
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader as dataloader
import nonechucks as nc
from voc_seg import my_data,label_acc_score,voc_colormap,seg_target
vgg=tv.models.vgg19_bn(pretrained=True)

image_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
train_data=my_data((240,320),transform=image_transform)
test_data=my_data((240,320),image_set='val',transform=image_transform)
trainset=nc.SafeDataset(train_data)
testset=nc.SafeDataset(test_data)
# trainload=nc.SafeDataLoader(trainset,batch_size=8)
testload=nc.SafeDataLoader(testset,batch_size=8)


mask_transform=transforms.Compose([seg_target()])# to_tensor will make it from nhwc to nchw

train_voc=tv.datasets.VOCSegmentation('/home/llm/PycharmProjects/seg_1224/data/',image_set='train',transform=image_transform,target_transform=mask_transform)
trainload=torch.utils.data.DataLoader(train_voc,shuffle=True)


device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device=torch.device('cpu')
dtype=torch.float32
num_class = 21
class deconv(nn.Module):
    def __init__(self,inchannel,middlechannel,outchannel,transpose=False):
        super(deconv,self).__init__()
        if transpose:
            self.block=nn.Sequential(nn.Conv2d(inchannel,middlechannel,3,padding=1),
                                   nn.BatchNorm2d(middlechannel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(middlechannel,middlechannel,3,padding=1),
                                   nn.BatchNorm2d(middlechannel),
                                   nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(middlechannel,outchannel,2,2)
                                     )
        else:
            self.block=nn.Sequential(nn.Conv2d(inchannel,middlechannel,3,padding=0),
                                   nn.BatchNorm2d(middlechannel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(middlechannel,middlechannel,3,padding=0),
                                   nn.BatchNorm2d(middlechannel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(middlechannel,outchannel,1),         #since unsampling cann't change the channel num ,have to change channel num before next block
                                   nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
                                     )
    def forward(self, input):
        return self.block(input)

class UNET(nn.Module):
# should crop at low level to reduce the pixels need to be cropped in the final layer
    def __init__(self):
        super(UNET,self).__init__()

        self.conv1=nn.Sequential(vgg.features[0],
                                 vgg.features[1],
                                 nn.ReLU(inplace=True),
                                 vgg.features[3],
                                 vgg.features[4],
                                 nn.ReLU(inplace=True),
                                )
        self.pool=vgg.features[6]
        self.conv2=nn.Sequential(vgg.features[7],
                                 vgg.features[8],
                                 nn.ReLU(inplace=True),
                                 vgg.features[10],
                                 vgg.features[11],
                                 nn.ReLU(inplace=True),
                                )
        self.conv3=nn.Sequential(vgg.features[14],
                                 vgg.features[15],
                                 nn.ReLU(inplace=True),
                                 vgg.features[17],
                                 vgg.features[18],
                                 nn.ReLU(inplace=True),
                                )
        self.conv4=nn.Sequential(vgg.features[27],
                                 vgg.features[28],
                                 nn.ReLU(inplace=True),
                                 vgg.features[30],
                                 vgg.features[31],
                                 nn.ReLU(inplace=True),)
        self.centre=deconv(512,1024,512)
        self.deconv4=deconv(1024,512,256)
        self.deconv3=deconv(512,256,128)
        self.deconv2=deconv(256,128,64)
        self.deconv1=nn.Sequential(nn.Conv2d(128,64,3,padding=0),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64,64,3,padding=0),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64,num_class,1))
        # self.deconv4=nn.Sequential(nn.Conv2d(1024,512,3,padding=1),
        #                            nn.BatchNorm2d(512),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(512,512,3,padding=1),
        #                            nn.BatchNorm2d(512),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(512,256,1),         #since unsampling cann't change the channel num ,have to change channel num before next block
        #                            nn.UpsamplingBilinear2d(scale_factor=2)
        #
        #                            ) # try to use upsampling instead of transpose conv
        # self.deconv3=nn.Sequential(nn.Conv2d(512,256,3,padding=1),
        #                            nn.BatchNorm2d(256),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(256,256,3,padding=1),
        #                            nn.BatchNorm2d(256),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(256,128,1),
        #                            nn.UpsamplingBilinear2d(scale_factor=2))
        # self.deconv2=nn.Sequential(nn.Conv2d(256,128,3,padding=1),
        #                            nn.BatchNorm2d(128),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(128,128,3,padding=1),
        #                            nn.BatchNorm2d(128),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(128,64,1),
        #                            nn.UpsamplingBilinear2d(scale_factor=2))
        # self.deconv1=nn.Sequential(nn.Conv2d(128,64,3,padding=1),
        #                            nn.BatchNorm2d(64),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(64,64,3),
        #                            nn.BatchNorm2d(64),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(64,num_class,1,padding=1))
    def forward(self, input):
        down1=self.conv1(input)
        down2=self.conv2(self.pool(down1))
        down3=self.conv3(self.pool(down2))
        down4=self.conv4(self.pool(down3))
        down5=self.centre(self.pool(down4))
        # print ('down1',down1.shape)
        # print ('down2',down2.shape)
        # print ('down3',down3.shape)
        # print ('down4',down4.shape)
        # print ('down5',down5.shape)

        down5=self.pad(down4,down5)
        up1=self.deconv4(torch.cat((down4,down5),dim=1))
        # print ('up1', up1.shape)
        up1=self.pad(down3,up1)
        up2=self.deconv3(torch.cat((up1,down3),dim=1))
        # print ('up2', up2.shape)
        up2=self.pad(down2,up2)
        up3=self.deconv2(torch.cat((up2,down2),dim=1))
        # print ('up3', up3.shape)
        up3=self.pad(down1,up3)
        up4=self.deconv1(torch.cat((up3,down1),dim=1))
        up4=self.pad(input,up4)
        # print ('up4',up4.shape)
        return up4
    def center_crop(self,img,target):
        h,w = img.shape[-2:]
        th, tw = target.shape[-2:]
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return img[...,i:i+th,j:j+tw]

    def pad(self,target,img): # the img & target is reverse as center_crop
        h,w = img.shape[-2:]
        th, tw = target.shape[-2:]
        i=int(math.ceil((th-h)/2.))
        j=int(math.ceil((tw-w)/2.))
        return torch.nn.functional.pad(img,(j,tw-w-j,i,th-h-i),mode='reflect')
def train(epoch):
    model=UNET()
    model.train()
    model.to(device)
    criterion=nn.CrossEntropyLoss()
    optimize=torch.optim.Adam(model.parameters(),lr=0.001)
    for i in range(epoch):
        tmp=0
        for image,mask in trainload:
            image,mask=image.to(device,dtype=dtype),mask.to(device,dtype=torch.long)
            optimize.zero_grad()
            pred=model(image)
            loss=criterion(pred,mask)
            loss.backward()
            optimize.step()
            tmp=loss.data
            # print ("loss ",tmp)
        print ("{0} epoch ,loss is {1}".format(i,tmp))
    return model
def label2image(pred):
    colormap=np.array(voc_colormap)
    return colormap[pred]
def test(model):
    img=[]
    pred=[]
    mask=[]
    with torch.no_grad():
        model.eval()
        model.to(device)
        for image,mask_img in testload:
            image=image.to(device,dtype=dtype)
            output=model(image)
            label=output.argmax(dim=1)
            pred.append(label)
            img.append(image)
            mask.append(mask_img)
            break
    return torch.cat(img),torch.cat(pred),torch.cat(mask)
def picture(img,pred,mask):
    # all must bu numpy object
    plt.figure()
    mean,std=np.array((0.485, 0.456, 0.406)),np.array((0.229, 0.224, 0.225))
    num=len(img)
    tmp=img.transpose(0,2,3,1)
    tmp=tmp*std+mean
    tmp=np.concatenate((tmp,pred,mask),axis=0)
    for i,j in enumerate(tmp,1):
        plt.subplot(3,num,i)
        plt.imshow(j)
    plt.show()

model=train(100)
torch.save(model.state_dict(),'unet_up_no_pad')
# model=UNET()
# model.load_state_dict(torch.load('unet_up_pad'))

# img=[]
# pred=[]
# mask=[]
# with torch.no_grad():
#     model.eval()
#     model.to(device)
#     for image,mask_img in trainload:
#         image=image.to(device,dtype=dtype)
#         output=model(image)
#         label=output.argmax(dim=1)
#         pred.append(label)
#         img.append(image)
#         mask.append(mask_img)
#     # pred=torch.cat(pred,dim=0)
#     # img=torch.cat(img,dim=0)
#     # mask=torch.cat(mask,dim=0)
# ap,iou,hist=label_acc_score(mask,pred,num_class)
# print(ap,iou)
# print(hist)
# picture(img[0].cpu().numpy()[:1],label2image(pred[0].cpu().numpy())/255.0,label2image(mask[0].cpu().numpy().astype(np.int))/255.0)



def torch_pic(img,mask,pred):
    voc_map=torch.from_numpy(np.array(voc_colormap))
    mask=voc_map[mask]
    pred=voc_map[pred]
    tmp=tv.utils.make_grid(torch.cat(voc_map))


# img,pred,mask=test(model)
#
#
# score=label_acc_score(mask.numpy().astype(np.int),pred.numpy().astype(np.int),num_class)
#
# pred_image=label2image(pred.numpy().astype(np.int))
# mask_image=label2image(mask.numpy().astype(np.int))
# picture(img.numpy(),pred_image,mask_image)
