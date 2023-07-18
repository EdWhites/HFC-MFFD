import os
from re import A
from PIL import Image
import torchvision.models as models
import torch.distributed as dist
import logging
import torch.nn as nn
import torch
import time
import torch.cuda
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils.get_filename import get_filepath
import utils.function as fn
from PIL import ImageFile
import torch_dct as dct
import cv2
import dlib
import model_arch
import utils.function as fn
import config as cfg
import itertools
ImageFile.LOAD_TRUNCATED_IMAGES=True
filename_mean=cfg.mean_path
filename_var=cfg.var_path
img_to_tensor=transforms.ToTensor()
mean=np.load(filename_mean)
variance=np.load(filename_var)
variance=np.sqrt(variance)
batch_size = 128
transform = transforms.Compose([
    transforms.ToTensor(),
]
)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(cfg.predictor_path)   
sift = cv2.SIFT_create()

def compute_feature(path):
    image = cv2.imread(path, 0)
    startx = 0
    starty = 0
    endx, endy = image.shape
    rect = dlib.rectangle(startx, starty, endx, endy)
    shape = predictor(image, rect)
    kp = fn.get_keypoint(shape)
    kp, des = sift.compute(image, kp)
    x = list(itertools.chain(*des))
    return x

    
class PatchDataset(Dataset):
    def __init__(self, path, transform=None,origin=True,dct=False,sift=False):
        super(PatchDataset).__init__()
        self.path = path
        self.files = fn.get_filepath(path)
        self.origin=origin
        self.dct=dct
        self.sift=sift
        self.files = sorted(self.files, key=lambda x: (x.split('/')[-3]))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        im=0
        im_dct=0
        im_sift=0
        fname = self.files[index]
        if self.dct:
            im = fn.load_image(fname, grayscale=True)
            im_dct = torch.Tensor(im.copy())
        if self.sift:
            im_sift = compute_feature(path=fname)
            im_sift = torch.Tensor(im_sift.copy())
        if self.origin:
            im = Image.open(fname)
            im = self.transform(im)

        if fname.split('/')[-3] == 'real':
            label = 1
        if fname.split('/')[-3] == 'fake':
            label = 0

        return label,im,im_dct,im_sift



def unfold_img(images,channel):
    unfold=nn.Unfold(kernel_size=(128,128),stride=(48,48))
    images = images.type(torch.FloatTensor)
    images = images.view(-1, channel, 224, 224)
    images = unfold(images)
    images = images.transpose(2, 1)
    images = images.contiguous().reshape(-1, 3,3,channel, 128, 128).transpose(2,1).reshape(-1,channel,128,128)
    return images



def get_results(dataset,origin=True, sift=True, dct=True, extract_feature_model=None,attention_model=None,attention_model_sift=None,extract_feature_model_dct=None,attention_model_dct=None):
    acc=0.0
    final_loss=0.0
    t=0
    pred_resnet=None
    pred_sift=None
    pred_dct=None
    loader= DataLoader(dataset,batch_size=batch_size,num_workers=16,pin_memory=True,shuffle=False)
    loss = nn.CrossEntropyLoss()
    device = torch.device("cuda", int(os.environ['LOCAL_RANK']))
    
    with torch.no_grad():
        result_resnet=np.zeros((dataset.__len__(),2))
        result_resnet_dct=np.zeros((dataset.__len__(),2))
        result_sift=np.zeros((dataset.__len__(),2))
        label=np.zeros((dataset.__len__(),1))
        for j,data in enumerate(loader):
            if sift:
                pred_sift=attention_model_sift(data[3].to(device))
            if dct:
                data[2]=unfold_img(data[2],channel=1)
                data[2]=fn.turn_to_dct(data[2],mean,variance)
                feature = extract_feature_model_dct(data[2].to(device))
                pred_dct = attention_model_dct(feature)
            if origin:
                data[1]=unfold_img(data[1],channel=3)
                feature = extract_feature_model(data[1].to(device))
                pred_resnet = attention_model(feature)
            for i in range(pred_resnet.shape[0]):
                result_resnet[t]=pred_resnet.view(-1,2).data.cpu().numpy()[i]
                result_sift[t]=pred_sift.view(-1,2).data.cpu().numpy()[i]
                result_resnet_dct[t]=pred_dct.view(-1,2).data.cpu().numpy()[i]
                label[t]=data[0].data.cpu().numpy()[i]
                t+=1
            pred = (pred_resnet+pred_dct+pred_sift)/3
            batch_loss = loss(pred, data[0].to(device))
            acc1 = torch.eq(torch.max(pred, 1)[1], data[0].to(device)).sum()
            dist.all_reduce(acc1, op=dist.ReduceOp.SUM)
            acc += acc1.item()
            batch_loss = fn.reduce_mean(batch_loss, dist.get_world_size())
            final_loss += batch_loss.item()
            print("\rProcess progress: {}% ".format((j + 1) / loader.__len__() * 100), end="")
        print(
            'Acc: %3.6f loss: %3.6f' % \
            ( acc / dataset.__len__(),
             final_loss / dataset.__len__()))
    return result_resnet,result_resnet_dct,result_sift,label

def test_model(usage, origin=True,dct=False,sift=False):
    print(os.environ['LOCAL_RANK'])
    if int(os.environ['LOCAL_RANK']) != -1:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", int(os.environ['LOCAL_RANK']))
    if dct:
        extract_feature_model_dct=model_arch.ResNet_50_dct().to(device)
        attention_model_dct=model_arch.AttentionResNet(n_classes=classes,patch_num=9).to(device)
        extract_feature_model_dct.load_state_dict(torch.load('./WildDeepFake/model/extract_feature_model_resnet_dct.pth'))
        attention_model_dct.load_state_dict(torch.load('./WildDeepFake/model/attention_model_resnet_dct.pth'))
        extract_feature_model_dct.eval()
        attention_model_dct.eval()
    if sift:
        attention_model_sift=model_arch.SIFTMLP(n_classes=classes).to(device)
        attention_model_sift.load_state_dict(torch.load('./WildDeepFake/model/attention_model_sift.pth'))
        attention_model_sift.eval()
    if origin:
        extract_feature_model=model_arch.ResNet_50().to(device)
        attention_model=model_arch.AttentionResNet(n_classes=classes,patch_num=9).to(device)
        extract_feature_model.load_state_dict(torch.load('./WildDeepFake/model/extract_feature_model_resnet.pth'))
        attention_model.load_state_dict(torch.load('./WildDeepFake/model/attention_model_resnet.pth'))
        extract_feature_model.eval()
        attention_model.eval()
    if usage=='binary_classification':
        classes=2
    if usage=='gan_classification':
        classes=4
    num_gpus = torch.cuda.device_count()
    train_set = PatchDataset(cfg.train_datasets_path, transform=transform,origin=origin,dct=dct,sift=sift)
    val_set = PatchDataset(cfg.test_datasets_path, transform=transform,origin=origin, dct=dct,sift=sift)

    
    result_resnet,result_resnet_dct,result_sift,label=get_results(train_set,origin=origin,dct=dct,sift=sift, extract_feature_model=extract_feature_model,attention_model= attention_model,extract_feature_model_dct=extract_feature_model_dct,attention_model_dct=attention_model_dct,attention_model_sift=attention_model_sift)
    np.save('./result/score_train_resnet.npy',result_resnet)
    np.save('./result/score_train_sift.npy',result_sift)
    np.save('./result/score_train_resnet_dct.npy',result_resnet_dct)
    np.save('./result/label_train.npy',label)
       
    result_resnet,result_resnet_dct,result_sift,label=get_results(val_set,origin=origin,dct=dct,sift=sift,extract_feature_model= extract_feature_model, attention_model=attention_model,extract_feature_model_dct=extract_feature_model_dct,attention_model_dct=attention_model_dct,attention_model_sift=attention_model_sift)
    np.save('./result/score_test_resnet.npy',result_resnet)
    np.save('./result/score_test_sift.npy',result_sift)
    np.save('./result/score_test_resnet_dct.npy',result_resnet_dct)
    np.save('./result/label_test.npy',label)



if __name__ == "__main__":
    os.makedirs("./result",exist_ok=True)
    test_model(usage='binary_classification',dct=True,sift=True)

