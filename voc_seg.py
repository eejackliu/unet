from PIL import Image as image
import torch
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import os
import torchvision.datasets as dset
voc_colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
class my_data(torch.utils.data.Dataset):
    #if target_transform=mask_transform is
    def __init__(self,data_size,root='/home/llm/PycharmProjects/seg_1224/data/',image_set='train',transform=None,target_transform=None):
        self.shape=data_size
        self.root=os.path.expanduser(root)
        self.transform=transform
        self.target_transform=target_transform
        self.image_set=image_set
        voc_dir=os.path.join(self.root,'VOCdevkit/VOC2012')
        image_dir=os.path.join(voc_dir,'JPEGImages')
        mask_dir=os.path.join(voc_dir,'SegmentationClass')
        splits_dir=os.path.join(voc_dir,'ImageSets/Segmentation')
        splits_f=os.path.join(splits_dir, self.image_set + '.txt')
        with open(os.path.join(splits_f),'r') as f:
            file_name=[x.strip() for x in f.readlines()]
        self.image=[os.path.join(image_dir,x+'.jpg') for x in file_name]
        self.mask=[os.path.join(mask_dir,x+'.png') for x in file_name]
        assert (len(self.image)==len(self.mask))

        self.class_index=np.zeros(256**3)
        for i,j in enumerate(voc_colormap):
            tmp=(j[0]*256+j[1])*256+j[2]
            self.class_index[tmp]=i
    def __getitem__(self, index):
        img=image.open(self.image[index]).convert('RGB')
        target=image.open(self.mask[index]).convert('RGB')
        # if img.size[0]< self.shape[1] or img.size[1] <self.shape[0]:
        #     return None,None
        i,j,h,w=transforms.RandomCrop.get_params(img,self.shape)
        # if i<0 or j<0 or h <0 or w<0:
        #     return None,None
        img=TF.crop(img,i,j,h,w)
        target=TF.crop(target,i,j,h,w)
        if  self.target_transform is not None:
            return self.transform(img),self.target_transform(target)
        target=np.array(target).transpose(2,0,1).astype(np.int32)
        target=(target[0]*256+target[1])*256+target[2]
        target=self.class_index[target]
        return self.transform(img),target

    def __len__(self):
        return len(self.image)
class seg_target(object):
    def __init__(self):
        voc_colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
        self.class_index=np.zeros(256**3)
        for i,j in enumerate(voc_colormap):
            tmp=(j[0]*256+j[1])*256+j[2]
            self.class_index[tmp]=i
    def __call__(self, target ):
        target=target.convert('RGB')
        target=np.array(target).transpose(2,0,1).astype(np.int32)
        target=(target[0]*256+target[1])*256+target[2]
        target=self.class_index[target]
        return target
def hist(label_true,label_pred,num_cls):
    # mask=(label_true>=0)&(label_true<num_cls)
    hist=np.bincount(label_pred.astype(int)*num_cls+label_true.astype(int),minlength=num_cls**2).reshape(num_cls,num_cls)
    return hist
def label_acc_score(label_true,label_pred,num_cls):
    hist_matrix=np.zeros((num_cls,num_cls))
    for i,j in zip(label_true,label_pred):
        hist_matrix+=hist(i.cpu().numpy().flatten(),j.cpu().numpy().flatten(),num_cls)
    diag=np.diag(hist_matrix)
    # acc=diag/hist_matrix.sum()
    acc_cls=diag/hist_matrix.sum(axis=0)
    m_iou=diag/(hist_matrix.sum(axis=1)+hist_matrix.sum(axis=0)-diag)
    return acc_cls,m_iou,hist_matrix