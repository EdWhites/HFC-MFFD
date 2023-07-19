import os
from PIL import Image
from utils.function import load_image, cal_diff
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

ImageFile.LOAD_TRUNCATED_IMAGES = True
filename = './mean.npy'
filename2 = './var.npy'
mean = np.load(filename)
variance = np.load(filename2)
variance = np.sqrt(variance)
batch_size =128
transform = transforms.Compose([
    transforms.ToTensor(),
]
)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s')
    logger = logging.getLogger(name)
    logger.setLevel((level_dict[verbosity]))
    fh = logging.FileHandler(filename, 'w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger






class PatchDataset_b(Dataset):
    def __init__(self, real_path, fake_path, transform=None, dct=False):
        super(PatchDataset_b).__init__()
        self.real_path = real_path
        real_files = get_filepath(real_path)
        fake_files = get_filepath(fake_path)
        real_files.extend(fake_files)
        self.files = real_files
        print(len(real_files))
        self.dct = dct
        self.files = sorted(self.files, key=lambda x: (x.split('/')[1],x.split('/')[-2], int(x.split('/')[-1][:-4])))
        self.transform = transform

    def __len__(self):
        return len(self.files) // 9

    def __getitem__(self, index):
        if self.dct:
            im = np.zeros((9, 1, 128, 128))
        else:
            im = np.zeros((9, 3, 128, 128))
        num = 0
        for i in range(index * 9, (index + 1) * 9):
            fname = self.files[i]
            if self.dct:
                patch = load_image(fname, grayscale=True)
                patch = cal_diff(patch, mean, variance)
            else:
                patch = Image.open(fname)
            patch = self.transform(patch)
            im[num] = patch
            num += 1

        del patch
        im = torch.Tensor(im)
        if fname.split('/')[-2] == 'cyclegan' or fname.split('/')[-2] == 'progan' or fname.split('/')[
            -2] == 'stylegan' or fname.split('/')[-2] == 'stylegan2':
            label = 1
        else:
            label = 0
        return im, label


class PatchDataset(Dataset):
    def __init__(self, real_path, fake_path, transform=None, dct=False):
        super(PatchDataset).__init__()
        self.real_path = real_path
        real_files = get_filepath(real_path)
        fake_files = get_filepath(fake_path)
        real_files.extend(fake_files)
        self.files = real_files
        self.dct = dct
        self.files = sorted(self.files, key=lambda x: (x.split('/')[-2], int(x.split('/')[-1][:-4])))
        self.transform = transform

    def __len__(self):
        return len(self.files) // 9

    def __getitem__(self, index):
        if self.dct:
            im = np.zeros((9, 1, 128, 128))
        else:
            im = np.zeros((9, 3, 128, 128))
        num = 0
        for i in range(index * 9, (index + 1) * 9):
            fname = self.files[i]
            if self.dct:
                patch = load_image(fname, grayscale=True)
                patch = cal_diff(patch, mean, variance)
            else:
                patch = Image.open(fname)
            patch = self.transform(patch)
            im[num] = patch
            num += 1
        del patch
        im = torch.Tensor(im)
        if fname.split('/')[-2] == 'cyclegan':
            label = 0
        if fname.split('/')[-2] == 'progan':
            label = 1
        if fname.split('/')[-2] == 'stylegan':
            label = 2
        if fname.split('/')[-2] == 'stylegan2':
            label = 3
        if fname.split('/')[-2] == 'test':
            label = 0                # a pseudo label for test
        return im, label

class PatchDataset_g_train(Dataset):
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
    
def get_train_score(usage, dct=False, traindir=None, model_path=None):
    loss = torch.nn.CrossEntropyLoss()
    if dct:
        model_name='resnet_dct'
        channel = 1
        extract_feature_model=model_arch.ResNet_50_dct().cuda()
    else:
        model_name='resnet'
        channel = 3
        extract_feature_model=model_arch.ResNet_50().cuda()
    if usage == 'binary_classification':
        classes = 2
        train_num=35000
        # t could make sure that the real image is placed in the front position, which ensures that the image order is consistent with SIFT's computational features
        # t should be changed if you want to extract feature from val subset.
        t=7000              
      
    if usage == 'gan_classification':
        classes = 4
        train_num=28000
        t=0
    attention_model=model_arch.AttentionResNet(n_classes=classes,patch_num=9).cuda()
    if dct==True:
        extract_feature_model.load_state_dict(torch.load(model_path+'/extract_feature_model_resnet_dct.pth'))
        attention_model.load_state_dict(torch.load(model_path+'/attention_model_resnet_dct.pth'))
    else:
        extract_feature_model.load_state_dict(torch.load(model_path + '/extract_feature_model_resnet.pth'))
        attention_model.load_state_dict(torch.load(model_path + '/attention_model_resnet.pth'))
    os.makedirs('./NIR_result/'+usage+'/'+model_name+'/'+traindir,exist_ok=True)
    train_path='./NIR_Forgery_Face_Patch/train/'+traindir
    if usage=='gan_classification':
        train_set = PatchDataset_g_train(path=train_path, transform=transform,dct=dct)
    if usage=='binary_classification':
        train_set = PatchDataset_b(real_path='./NIR_Face_Patch/train',fake_path=train_path, transform=transform,dct=dct)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=16, pin_memory=True, shuffle=False)
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    attention_model.eval()
    extract_feature_model.eval()

    result = np.zeros((train_num, classes))

    with torch.no_grad():
        for j, data in enumerate(train_loader):
            data[0] = data[0].type(torch.FloatTensor)
            data[0] = data[0].view(-1, channel, 128, 128)
            feature = extract_feature_model(data[0].cuda())
            train_pred = attention_model(feature)
            train_pred1=train_pred.data.cpu().view(-1,classes).numpy()
            for i in train_pred1:
                result[t%train_num]=i
                t+=1
            batch_loss = loss(train_pred, data[1].cuda())
            train_acc1 = torch.eq(torch.max(train_pred, 1)[1], data[1].cuda()).sum()
            train_acc += train_acc1.item()
            train_loss += batch_loss.item()
            del feature
            print("\rProcess progress: {}% ".format( (j + 1) / train_loader.__len__() * 100), end="")
        print(
            ' %2.2f sec(s) Val Acc: %3.6f loss: %3.6f' % \
            ( time.time() - epoch_start_time, \
            train_acc / train_set.__len__(),
            train_loss / train_set.__len__()))
        np.save('./NIR_result/'+usage+'/'+model_name+'/'+traindir+'/resnet_result_'+traindir+'_train.npy',result)
            

def get_val_test_score(usage, dct=False, traindir=None, model_path=None,val_or_test=None):
    loss = torch.nn.CrossEntropyLoss()
    if usage == 'binary_classification':
        classes = 2
    if usage == 'gan_classification':
        classes = 4
    if dct:
        model_name='resnet_dct'
        channel = 1
        extract_feature_model=model_arch.ResNet_50_dct().cuda()
    else:
        model_name='resnet'
        channel = 3
        extract_feature_model =model_arch.ResNet_50().cuda()
    attention_model=model_arch.AttentionResNet(n_classes=classes,patch_num=9).cuda()
    if traindir=='std_single':
        filedir=['std_single','rand_single','mix_single']
    if traindir=='std_multi':
        filedir=['std_multi','rand_multi','mix_multi']
    if traindir=='rand_single':
        filedir=['rand_single','mix_single']
    if traindir=='rand_multi':
        filedir = ['rand_multi', 'mix_multi']
    if traindir=='mix_single':
        filedir=['mix_single']
    if traindir=='mix_multi':
        filedir = ['mix_multi']
    if dct==True:
        extract_feature_model.load_state_dict(torch.load(model_path+'/extract_feature_model_resnet_dct.pth'))
        attention_model.load_state_dict(torch.load(model_path+'/attention_model_resnet_dct.pth'))
    else:
        extract_feature_model.load_state_dict(torch.load(model_path + '/extract_feature_model_resnet.pth'))
        attention_model.load_state_dict(torch.load(model_path + '/attention_model_resnet.pth'))
    os.makedirs('./NIR_result/'+usage+'/'+model_name+'/'+traindir,exist_ok=True)

    for testdir in filedir:
        test_path = './NIR_Forgery_Face_Patch/'+val_or_test+'/' + testdir
        if usage == 'gan_classification':
            val_set = PatchDataset(real_path='./NIR_Face_Patch/'+val_or_test, fake_path=test_path, transform=transform, dct=dct)
        if usage == 'binary_classification':
            val_set = PatchDataset_b(real_path='./NIR_Face_Patch/'+val_or_test, fake_path=test_path,
                                        transform=transform, dct=dct)

        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16, pin_memory=True, shuffle=False)
        epoch_start_time = time.time()
        val_acc = 0.0
        val_loss = 0.0
        attention_model.eval()
        extract_feature_model.eval()
        result=np.zeros((10000,classes))
        t=2000
        with torch.no_grad():
            for j, data in enumerate(val_loader):

                data[0] = data[0].type(torch.FloatTensor)
                data[0] = data[0].view(-1, channel, 128, 128)
                feature = extract_feature_model(data[0].cuda())
                val_pred = attention_model(feature)
                val_pred1=val_pred.data.cpu().view(-1,classes).numpy()
                for i in val_pred1:
                    result[t%10000]=i
                    t+=1
                batch_loss = loss(val_pred, data[1].cuda())
                val_acc1 = torch.eq(torch.max(val_pred, 1)[1], data[1].cuda()).sum()
                val_acc += val_acc1.item()
                val_loss += batch_loss.item()
                del feature
                print("\rProcess progress: {}% ".format((j + 1) / val_loader.__len__() * 100), end="")
            print(
                ' %2.2f sec(s) Val Acc: %3.6f loss: %3.6f' % \
                ( time.time() - epoch_start_time, \
                val_acc / val_set.__len__(),
                val_loss / val_set.__len__()))

        np.save('./NIR_result/'+usage+'/'+model_name+'/'+traindir+'/resnet_result_'+testdir+'_'+val_or_test+'.npy',result)

def evaluate_model(dct=False, usage=None):
    dir = [ 'std_single','std_multi','rand_single','rand_multi','mix_single','mix_multi']
    for traindir in dir:
        print('The num of GPU:', torch.cuda.device_count())
        model_path = "./NIR_model/" + traindir + '/' + usage
        get_train_score(usage=usage,dct=dct,traindir=traindir,model_path=model_path)
        get_val_test_score(usage=usage,dct=dct,traindir=traindir,model_path=model_path,val_or_test='test')
        
if __name__ == "__main__":
    evaluate_model(usage='gan_classification')



