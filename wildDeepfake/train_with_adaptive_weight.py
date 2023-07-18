import os
import torch
import time
import torch.cuda
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import ImageFile
import utils.function as fn
import torch_dct as dct
import config as cfg
ImageFile.LOAD_TRUNCATED_IMAGES=True
filename_mean=cfg.mean_path
filename_var=cfg.var_path
img_to_tensor=transforms.ToTensor()
mean=np.load(filename_mean)
variance=np.load(filename_var)
variance=np.sqrt(variance)
batch_size = 256

transform = transforms.Compose([
    transforms.ToTensor(),
]
)
    


class ScoreDataset(Dataset):
    def __init__(self, resnet_path,resnet_dct_path,sift_path, label_path,transform=None,dct=False):
        super(ScoreDataset).__init__()
        self.resnet = np.load(resnet_path)
        self.resnet_dct = np.load(resnet_dct_path)
        self.sift=np.load(sift_path)
        self.transform = transform
        self.label=np.load(label_path)

    def __len__(self):
        return len(self.resnet)

    def __getitem__(self, index):
        score_resnet = torch.Tensor(self.resnet[index])
        score_resnet_dct=torch.Tensor(self.resnet_dct[index])
        score_sift=torch.Tensor(self.sift[index])
        label=torch.Tensor(self.label[index])
        return score_resnet,score_resnet_dct,score_sift,label


def train_model():

    logger=fn.get_logger('./train_with_adaptive_weight.log')
    logger.info(('start training!'))
    loss_fn = torch.nn.CrossEntropyLoss()

    w1 = torch.tensor([1.]).requires_grad_()
    w2 = torch.tensor([1.]).requires_grad_()
    w3 = torch.tensor([1.]).requires_grad_()
    optimizer= torch.optim.Adam([w1,w2,w3], lr=1e-4)
    
    num_epoch = 150
    train_set = ScoreDataset(resnet_path='./result/score_train_resnet.npy',resnet_dct_path='./result/score_train_resnet_dct.npy',sift_path='./result/score_train_sift.npy',label_path='./result/label_train.npy', transform=transform,dct=dct)
    val_set = ScoreDataset(resnet_path='./result/score_test_resnet.npy',resnet_dct_path='./result/score_test_resnet_dct.npy',sift_path='./result/score_test_sift.npy',label_path='./result/label_test.npy', transform=transform,dct=dct)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=16,pin_memory=True,shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16,pin_memory=True,shuffle=False)

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        batch_start_time = time.time()
        load_time=0
        length=train_loader.__len__()
        for j, data in enumerate(train_loader):
            optimizer.zero_grad()
            train_pred=(torch.exp(w1.cuda())*data[0].cuda()+torch.exp(w2.cuda())*data[1].cuda()+torch.exp(w3.cuda())*data[2].cuda())/(torch.exp(w1.cuda())+torch.exp(w2.cuda())+torch.exp(w3.cuda()))
            label=data[3].squeeze(1).long().cuda()

            loss = loss_fn(train_pred, label)
            loss.backward()
            optimizer.step()
            train_acc_1=torch.eq(torch.max(train_pred,1)[1],label).sum()
            train_acc+=train_acc_1.item()
            train_loss += loss.item()  # 计算所有的loss
            batch_load_time=time.time()
            if batch_load_time-batch_start_time>10:
                load_time=batch_load_time-batch_start_time
            print("\rBatch_load time: {}, Process progress: {}% ".format(load_time,(j + 1) / length * 100), end="")
            batch_start_time=time.time()
        with torch.no_grad():
            for j, data in enumerate(val_loader):
                val_pred = (torch.exp(w1.cuda())*data[0].cuda()+torch.exp(w2.cuda())*data[1].cuda()+torch.exp(w3.cuda())*data[2].cuda())/(torch.exp(w1.cuda())+torch.exp(w2.cuda())+torch.exp(w3.cuda()))
                label=data[3].squeeze(1).long().cuda()
                batch_loss = loss_fn(val_pred, label)
                val_acc1 = torch.eq(torch.max(val_pred, 1)[1], label).sum()
                val_acc += val_acc1.item()
                val_loss += batch_loss.item()
            logger.info('[%03d/%03d] w1: %f, w2: %f, w3: %f  %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                (epoch + 1, num_epoch, w1, w2, w3, time.time() - epoch_start_time, \
                 train_acc / train_set.__len__(), train_loss / train_set.__len__(),
                 val_acc / val_set.__len__(),
                 val_loss / val_set.__len__()))

    logger.info('finish training')    

if __name__ == "__main__":
    usage=cfg.usage
    print('The num of GPU:',torch.cuda.device_count())
    train_model()

