import os
from PIL import Image
from utils.function import load_image,cal_diff
import logging
import torch
import time
import torch.cuda
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils.get_filename import get_filepath
from PIL import ImageFile
import model_arch

ImageFile.LOAD_TRUNCATED_IMAGES=True
filename_mean='./mean.npy'
filename_var='./var.npy'
mean=np.load(filename_mean)
variance=np.load(filename_var)
variance=np.sqrt(variance)
batch_size = 32
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



class PatchDataset_b(Dataset):
    def __init__(self, real_path,fake_path, transform=None,dct=False):
        super(PatchDataset).__init__()
        self.real_path = real_path
        real_files = get_filepath(real_path)
        fake_files=get_filepath(fake_path)
        real_files.extend(fake_files)
        self.files=real_files
        print(len(real_files))
        self.dct=dct
        self.files = sorted(self.files, key=lambda x: (x.split('/')[-2],int(x.split('/')[-1][:-4])))
        self.transform = transform

    def __len__(self):
        return len(self.files)//9

    def __getitem__(self, index):
        if self.dct:
            im=np.zeros((9,1,128,128))
        else:
            im=np.zeros((9,3,128,128))
        num=0
        for i in range(index * 9, (index + 1) * 9):
            fname = self.files[i]
            if self.dct:
                patch = load_image(fname, grayscale=True)
                patch = cal_diff(patch, mean, variance)
            else:
                patch=Image.open(fname)
            patch=self.transform(patch)
            im[num]=patch
            num+=1
        del patch
        im=torch.Tensor(im)
        if fname.split('/')[-2]=='cyclegan' or fname.split('/')[-2]=='progan' or fname.split('/')[-2]=='stylegan' or fname.split('/')[-2]=='stylegan2':
            label=1
        else:
            label=0
        return im,label

class PatchDataset(Dataset):
    def __init__(self, path, transform=None,dct=False):
        super(PatchDataset).__init__()
        self.path = path
        self.files = get_filepath(path)
        self.dct=dct
        self.files = sorted(self.files, key=lambda x: (x.split('/')[-2],int(x.split('/')[-1][:-4])))
        self.transform = transform

    def __len__(self):
        return len(self.files)//9

    def __getitem__(self, index):
        if self.dct:
            im=np.zeros((9,1,128,128))
        else:
            im=np.zeros((9,3,128,128))
        num=0
        for i in range(index * 9, (index + 1) * 9):
            fname = self.files[i]
            if self.dct:
                patch = load_image(fname, grayscale=True)
                patch = cal_diff(patch, mean, variance)
            else:
                patch=Image.open(fname)
            patch=self.transform(patch)
            im[num]=patch
            num+=1
        del patch
        im=torch.Tensor(im)
        if fname.split('/')[-2]=='cyclegan':
            label=0
        if fname.split('/')[-2]=='progan':
            label=1
        if fname.split('/')[-2]=='stylegan':
            label=2
        if fname.split('/')[-2]=='stylegan2':
            label=3
        return im,label


def train_model(usage, dct=False,train_path=None,test_path=None,output_path=None):
    if dct:
        logger=get_logger(output_path+'/train_resnet_dct_'+train_path.split('/')[-1]+'.log')
    else:
        logger=get_logger(output_path+'/train_resnet_'+train_path.split('/')[-1]+'.log')
    logger.info(('start training!'))
    loss = torch.nn.CrossEntropyLoss()
    if dct:
        extract_feature_model =model_arch.ResNet_50_dct().cuda()
        channel=1
    else:
        extract_feature_model =model_arch.ResNet_50().cuda()
        channel=3
    if usage=='binary_classification':
        num_epoch = 8
        classes=2
    if usage=='gan_classification':
        num_epoch = 8
        classes=4

    mlp=model_arch.R_MLP(n_classes=classes).cuda()
    attention_model=model_arch.AttentionResNet(n_classes=classes,patch_num=9).cuda()

    extract_feature_model = torch.nn.DataParallel(extract_feature_model, device_ids=[0, 1])
    mlp = torch.nn.DataParallel(mlp, device_ids=[0, 1])
    attention_model = torch.nn.DataParallel(attention_model, device_ids=[0, 1])

    optimizer_et= torch.optim.Adam(filter(lambda p: p.requires_grad, extract_feature_model.parameters()), lr=1e-4)
    optimizer_at_resnet_dct = torch.optim.Adam(filter(lambda p: p.requires_grad, attention_model.parameters()), lr=1e-5)
    optimizer_mlp=torch.optim.Adam(filter(lambda p: p.requires_grad,mlp.parameters()), lr=1e-4)

    if usage=='gan_classification':
        if dct:
            train_set = PatchDataset(train_path, transform=transform,dct=True)
            val_set = PatchDataset(test_path, transform=transform,dct=True)
        else:
            train_set = PatchDataset(train_path, transform=transform)
            val_set = PatchDataset(test_path, transform=transform)
    if usage=='binary_classification':
        train_set = PatchDataset_b(real_path='./NIR_Face_Patch/train',fake_path=train_path, transform=transform,dct=dct)
        val_set = PatchDataset_b(real_path='./NIR_Face_Patch/val',fake_path=test_path, transform=transform,dct=dct)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=16,pin_memory=True,shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16,pin_memory=True,shuffle=False)

    arfa=0.6
    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_acc_mlp=0.0
        val_loss = 0.0
        extract_feature_model.train()
        mlp.train()
        attention_model.train()
        batch_start_time = time.time()
        load_time=0
        length=train_loader.__len__()
        for j, data in enumerate(train_loader):
            data[0] = data[0].type(torch.FloatTensor)
            data[0] = data[0].view(-1, channel, 128, 128)

            feature = extract_feature_model(data[0].cuda())
            patch_label=np.zeros((len(data[1])*9))
            for i in range(len(data[1])):
                for num in range(i * 9, (i + 1) * 9):
                    patch_label[num] = data[1][i]
            patch_label=torch.LongTensor(patch_label)
            train_pred_patch=mlp(feature)
            optimizer_et.zero_grad()
            optimizer_mlp.zero_grad()
            loss_et=loss(train_pred_patch,patch_label.cuda())
            optimizer_at_resnet_dct.zero_grad()
            train_pred=attention_model(feature)
            loss_at = loss(train_pred, data[1].cuda())
            loss_all=arfa*loss_et+(1-arfa)*loss_at
            loss_all.backward()
            optimizer_mlp.step()
            optimizer_et.step()
            optimizer_at_resnet_dct.step()
            train_acc_1=torch.eq(torch.max(train_pred,1)[1],data[1].cuda()).sum()
            train_acc+=train_acc_1.item()
            train_loss += loss_all.item()  
            batch_load_time=time.time()
            del feature,train_pred_patch,train_pred
            if batch_load_time-batch_start_time>10:
                load_time=batch_load_time-batch_start_time
            print("\rBatch_load time: {}, Process progress: {}% ".format(load_time,(j + 1) / length * 100), end="")
            batch_start_time=time.time()
        attention_model.eval()
        extract_feature_model.eval()
        mlp.eval()
        with torch.no_grad():
            for j, data in enumerate(val_loader):
                data[0] = data[0].type(torch.FloatTensor)
                data[0] = data[0].view(-1, channel, 128, 128)
                feature = extract_feature_model(data[0].cuda())
                patch_label = np.zeros((len(data[1]) * 9))
                for i in range(len(data[1])):
                    for num in range(i * 9, (i + 1) * 9):
                        patch_label[num] = data[1][i]

                patch_label = torch.LongTensor(patch_label)
                val_pred_patch = mlp(feature)
                val_pred = attention_model(feature)
                batch_loss = loss(val_pred, data[1].cuda())
                val_acc_mlp_1 = torch.eq(torch.max(val_pred_patch, 1)[1], patch_label.cuda()).sum()
                val_acc1 = torch.eq(torch.max(val_pred, 1)[1], data[1].cuda()).sum()
                val_acc_mlp+=val_acc_mlp_1.item()
                val_acc += val_acc1.item()
                val_loss += batch_loss.item()
                del feature
            logger.info('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Patch Acc: %3.6f Val Acc: %3.6f loss: %3.6f' % \
                (epoch + 1, num_epoch, time.time() - epoch_start_time, \
                 train_acc / train_set.__len__(), train_loss / train_set.__len__(),
                 val_acc_mlp / (9*val_set.__len__()), val_acc / val_set.__len__(),
                 val_loss / val_set.__len__()))
        if dct == True:
            torch.save(extract_feature_model.module.state_dict(), output_path+'/extract_feature_model_resnet_dct_'+str(epoch)+'.pth')
            torch.save(mlp.module.state_dict(),output_path+'/mlp_resnet_dct_'+str(epoch)+'.pth')
            torch.save(attention_model.module.state_dict(), output_path+'/attention_model_resnet_dct_'+str(epoch)+'.pth')
        else:
            torch.save(extract_feature_model.module.state_dict(), output_path+'/extract_feature_model_resnet_'+str(epoch)+'.pth')
            torch.save(mlp.module.state_dict(),output_path+'/mlp_resnet_'+str(epoch)+'.pth')
            torch.save(attention_model.module.state_dict(), output_path+'/attention_model_resnet_'+str(epoch)+'.pth')
    logger.info('finish training')


def train_and_test_model(dct=False,usage=None):
    dir = ['std_single', 'std_multi', 'rand_single', 'rand_multi', 'mix_single','mix_multi']
    for traindir in dir:
        train_path='./NIR_Forgery_Face_Patch/train/'+traindir
        test_path='./NIR_Forgery_Face_Patch/val/'+traindir
        os.makedirs("./NIR_model/"+traindir+'/'+usage,exist_ok=True)
        output_path="./NIR_model/"+traindir+'/'+usage
        print('The num of GPU:',torch.cuda.device_count())
        
        train_model(usage, dct=dct,train_path=train_path,test_path=test_path,output_path=output_path)

if __name__ == "__main__":

    train_and_test_model(usage='binary_classification')

