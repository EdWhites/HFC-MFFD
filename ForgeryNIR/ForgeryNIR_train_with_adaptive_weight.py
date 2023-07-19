import os
import logging
import torch.nn as nn
import torch
import time
import torch.cuda
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import ImageFile
import torch_dct as dct
ImageFile.LOAD_TRUNCATED_IMAGES=True
filename = './mean.npy'
filename2 = './var.npy'
img_to_tensor=transforms.ToTensor()
mean=np.load(filename)
variance=np.load(filename2)
variance=np.sqrt(variance)
batch_size = 256

transform = transforms.Compose([
    transforms.ToTensor(),
]
)

def get_logger(filename,verbosity=1,name=None):
    level_dict={0:logging.DEBUG,1:logging.INFO,2:logging.WARNING}
    formatter=logging.Formatter('[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s')
    logger=logging.getLogger(name)
    logger.setLevel((level_dict[verbosity]))
    fh=logging.FileHandler(filename,'w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh=logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger




    


class ScoreDataset(Dataset):
    def __init__(self, resnet_path=None,resnet_dct_path=None,sift_path=None, label=None,transform=None,gan_classification_and_test=False,dct=False):
        super(ScoreDataset).__init__()
        self.resnet_path=resnet_path
        self.resnet_dct_path=resnet_dct_path
        self.sift_path=sift_path
        if resnet_path:
            self.resnet = np.load(self.resnet_path)
            if gan_classification_and_test:
                self.resnet = self.resnet[2000:]
        if resnet_dct_path:
            
            self.resnet_dct = np.load(self.resnet_dct_path)
            
        if sift_path:
            
            self.sift=np.load(self.sift_path)
            if gan_classification_and_test:
                self.sift = self.sift[2000:]
        self.transform = transform
        self.label=label


    def __len__(self):
        return len(self.resnet)

    def __getitem__(self, index):
        score_resnet=0
        score_resnet_dct=0
        score_sift=0
        if self.resnet_path:
            score_resnet = torch.Tensor(self.resnet[index])
        if self.resnet_dct_path:
            score_resnet_dct=torch.Tensor(self.resnet_dct[index])
        if self.sift_path:
            score_sift=torch.Tensor(self.sift[index])

        tmplabel=self.label[index]
        return dict(score_resnet=score_resnet,score_resnet_dct=score_resnet_dct,score_sift=score_sift,label=tmplabel)
    


def prepare_optimizer(usage):
    if usage == 'binary_classification':
        w11 = torch.tensor([1.]).requires_grad_()
        w12 = torch.tensor([1.]).requires_grad_()
        w13 = torch.tensor([1.]).requires_grad_()
        weight=dict(w11=w11,w12=w12,w13=w13)
        optimizer= torch.optim.Adam([w11,w12,w13], lr=1e-4)
        return weight,optimizer
    if usage=='gan_classification':
        w21=torch.tensor([1.]).requires_grad_()
        w22=torch.tensor([1.]).requires_grad_()
        weight=dict(w21=w21,w22=w22)
        optimizer= torch.optim.Adam([w21,w22], lr=1e-4)
        return weight,optimizer
    
def pred_with_weight(usage,data,weight):
    if usage=='binary_classification':
        train_pred=(torch.exp(weight['w11'].cuda())*data['score_resnet'].cuda()+torch.exp(weight['w12'].cuda())*data['score_resnet_dct'].cuda()+torch.exp(weight['w13'].cuda())*data['score_sift'].cuda())/(torch.exp(weight['w11'].cuda())+torch.exp(weight['w12'].cuda())+torch.exp(weight['w13'].cuda()))
    if usage=='gan_classification':
        train_pred=(torch.exp(weight['w21'].cuda())*data['score_resnet'].cuda()+torch.exp(weight['w22'].cuda())*data['score_sift'].cuda())/(torch.exp(weight['w21'].cuda())+torch.exp(weight['w22'].cuda()))
    return train_pred
def train_model(usage,traindir):
    score_path_resnet_train='./NIR_result/'+usage+'/resnet/'+traindir+'/resnet_result_'+traindir+'_train.npy'
    score_path_resnet_dct_train='./NIR_result/'+usage+'/resnet/'+traindir+'/resnet_result_'+traindir+'_train.npy'
    score_path_sift_train='./NIR_result/'+usage+'/sift/'+traindir+'/sift_result_'+traindir+'_train.npy'
    
    logger=get_logger('./NIR_weight/'+usage+'/train_'+traindir+'.log')
    logger.info(('start training!'))
    loss = nn.CrossEntropyLoss()
    weight,optimizer=prepare_optimizer(usage=usage)
    num_epoch = 150
    if traindir == 'std_single':
        filedir = ['std_single', 'rand_single', 'mix_single']
    if traindir == 'std_multi':
        filedir = [ 'std_multi','rand_multi','mix_multi']
    if traindir == 'rand_single':
        filedir = ['rand_single', 'mix_single']
    if traindir == 'rand_multi':
        filedir = ['rand_multi', 'mix_multi']
    if traindir == 'mix_single':
        filedir = ['mix_single']
    if traindir == 'mix_multi':
        filedir = ['mix_multi']
    
    if usage=='binary_classification':
        label=[0]*7000+[1]*28000
        label=np.array(label)
        train_set = ScoreDataset(resnet_path=score_path_resnet_train,resnet_dct_path=score_path_resnet_dct_train,sift_path=score_path_sift_train,label=label, transform=transform,dct=dct)
    if usage=='gan_classification':
        label=[0]*7000+[1]*7000+[2]*7000+[3]*7000
        label=np.array(label)
        train_set = ScoreDataset(resnet_path=score_path_resnet_train,sift_path=score_path_sift_train,label=label, transform=transform,dct=dct)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=16,pin_memory=True,shuffle=True)
    
    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
    
        batch_start_time = time.time()
        load_time=0
        length=train_loader.__len__()
        for j, data in enumerate(train_loader):
            optimizer.zero_grad()
            train_pred=pred_with_weight(data=data,usage=usage,weight=weight)
	
            batch_loss = loss(train_pred, data['label'].cuda())
            batch_loss.backward()
            optimizer.step()
            train_acc_1=torch.eq(torch.max(train_pred,1)[1],data['label'].cuda()).sum()
            train_acc+=train_acc_1.item()
            train_loss += batch_loss.item()  # 计算所有的loss
            batch_load_time=time.time()
            if batch_load_time-batch_start_time>10:
                load_time=batch_load_time-batch_start_time
            print("\rBatch_load time: {}, Process progress: {}% ".format(load_time,(j + 1) / length * 100), end="")
            batch_start_time=time.time()
        logger.info(weight)
        for valdir in filedir:
            score_path_resnet_test='./NIR_result/'+usage+'/resnet/'+traindir+'/resnet_result_'+valdir+'_val.npy'
            score_path_resnet_dct_test='./NIR_result/'+usage+'/resnet/'+traindir+'/resnet_result_'+valdir+'_val.npy'
            score_path_sift_test='./NIR_result/'+usage+'/sift/'+traindir+'/sift_result_'+valdir+'_val.npy'
            if usage=='binary_classification':
                label=[0]*2000+[1]*8000
                label=np.array(label)
                val_set = ScoreDataset(resnet_path=score_path_resnet_test,resnet_dct_path=score_path_resnet_dct_test,sift_path=score_path_sift_test,label=label, transform=transform,dct=dct)
            if usage=='gan_classification':
                label=[0]*2000+[1]*2000+[2]*2000+[3]*2000
                label=np.array(label)
                val_set = ScoreDataset(resnet_path=score_path_resnet_test,sift_path=score_path_sift_test,label=label, transform=transform,dct=dct,gan_classification_and_test=True)
            val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16,pin_memory=True,shuffle=False)
            val_acc = 0.0
            val_loss = 0.0
            with torch.no_grad():
                for j, data in enumerate(val_loader):
                    val_pred = pred_with_weight(data=data,usage=usage,weight=weight)
                    batch_loss = loss(val_pred, data['label'].cuda())
                    val_acc1 = torch.eq(torch.max(val_pred, 1)[1], data['label'].cuda()).sum()
                    val_acc += val_acc1.item()
                    val_loss += batch_loss.item()
                
                logger.info('[%03d/%03d] Train with {}, Val with {}, Acc is {}'.format(epoch+1, num_epoch,traindir,valdir,val_acc/val_set.__len__()))

    logger.info('finish training')    


if __name__ == "__main__":
    usage='binary_classification'
    os.makedirs("./NIR_weight/"+usage,exist_ok=True)
    print('The num of GPU:',torch.cuda.device_count())

    dir=['std_single', 'std_multi', 'rand_single', 'rand_multi', 'mix_single','mix_multi']
    for traindir in dir:
        train_model(usage=usage,traindir=traindir)

