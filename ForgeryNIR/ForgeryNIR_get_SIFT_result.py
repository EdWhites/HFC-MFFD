import torch.nn as nn
import torch
import torch.cuda
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import model_arch
batch_size = 128
transform = transforms.Compose([
    transforms.ToTensor()]
)




class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        super(ImgDataset).__init__()
        self.x=x
        self.y=y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        im=self.x[index]
        im=torch.Tensor(im)
        label=self.y[index]
        return im,label


def get_sift_result_train(traindir,usage):
    os.makedirs('./NIR_result/'+usage+'/sift/'+traindir, exist_ok=True)
    if usage == 'binary_classification':
        classes = 2
    if usage == 'gan_classification':
        classes = 4
    path='./NIR_model/sift_model/'+usage+'/attention_model_sift_'+traindir+'.pth'
    model = model_arch.SIFTMLP(n_classes=classes).cuda()
    model.load_state_dict(torch.load(path))
    if usage=='binary_classification':
        train_label = [0] * 7000 + [1] * 28000
        train_label = np.array(train_label)
    if usage=='gan_classification':
        train_label = [0] * 7000 +[1] * 7000 +[2] * 7000+[3] * 7000
        train_label = np.array(train_label)
    loss=nn.CrossEntropyLoss()

    if usage=='gan_classification':
        result = np.zeros((28000, 4))
        train_path='./NIR_feature/train_des-'+traindir+'.npy'
        train_data = np.load(train_path)[7000:]
    if usage=='binary_classification':
        result=np.zeros((35000, 2))
        train_path='./NIR_feature/train_des-'+traindir+'.npy'
        train_data = np.load(train_path)
        
    train_set = ImgDataset(train_data, train_label, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    train_acc=0.0
    train_loss=0.0
    model.eval()
    t=0
    with torch.no_grad():
        for j, data in enumerate(train_loader):
            data[0] = data[0].type(torch.FloatTensor)
            train_pred = model(data[0].cuda())
            batch_loss = loss(train_pred, data[1].cuda())
            train_pred1 = train_pred.data.cpu().view(-1, classes).numpy()
            for i in train_pred1:
                result[t] = i
                t += 1
            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()
        print(
                'train Acc: %3.6f loss: %3.6f ' % \
                (train_acc / train_set.__len__(),
                 train_loss / train_set.__len__(),
                 ))
    np.save('./NIR_result/'+usage+'/sift/'+traindir+'/sift_result_' + traindir + '_train.npy', result)

def get_sift_result_test(traindir,usage,val_or_test='test'):
    os.makedirs('./NIR_result/'+usage+'/sift/'+traindir, exist_ok=True)
    if usage == 'binary_classification':
        classes = 2
    if usage == 'gan_classification':
        classes = 4
    path = './NIR_model/sift_model/'+usage+'/attention_model_sift_'+traindir+'.pth'
    model = model_arch.SIFTMLP(n_classes=classes).cuda()
    model.load_state_dict(torch.load(path))
    if usage=='binary_classification':
        test_label = [0] * 2000 + [1] * 8000
        test_label = np.array(test_label)
    if usage=='gan_classification':
        test_label = [0] * 2000+[0] * 2000 +[1] * 2000 +[2] * 2000+[3] * 2000
        test_label = np.array(test_label)
    loss=nn.CrossEntropyLoss()
    if traindir == 'std_single':
        filedir = ['std_single','rand_single','mix_single']
    if traindir == 'std_multi':
        filedir = ['std_multi', 'rand_multi', 'mix_multi']
    if traindir == 'rand_single':
        filedir = ['rand_single', 'mix_single']
    if traindir == 'rand_multi':
        filedir = ['rand_multi', 'mix_multi']
    if traindir == 'mix_single':
        filedir = ['mix_single']
    if traindir == 'mix_multi':
        filedir = ['mix_multi']

    for testdir in filedir:
        if usage=='gan_classification':
            result = np.zeros((10000, 4))
        if usage=='binary_classification':
            result=np.zeros((10000, 2))
        test_path='./NIR_feature/'+val_or_test+'_des-'+testdir+'.npy'
        test_data = np.load(test_path)
        test_set = ImgDataset(test_data, test_label, transform=transform)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        test_acc=0.0
        test_loss=0.0
        model.eval()
        t=0
        with torch.no_grad():
            for j, data in enumerate(test_loader):
                data[0] = data[0].type(torch.FloatTensor)
                test_pred = model(data[0].cuda())
                batch_loss = loss(test_pred, data[1].cuda())
                test_pred1 = test_pred.data.cpu().view(-1, classes).numpy()
                for i in test_pred1:
                    result[t] = i
                    t += 1
                test_acc += np.sum(np.argmax(test_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                test_loss += batch_loss.item()
            print(
                    'test Acc: %3.6f loss: %3.6f ' % \
                    (test_acc / test_set.__len__(),
                     test_loss / test_set.__len__(),
                     ))
        np.save('./NIR_result/'+usage+'/sift/'+traindir+'/sift_result_' + testdir + '_'+val_or_test+'.npy', result)




if __name__ == '__main__':
    function=['binary_classification','gan_classification']
    filedir = ['std_single', 'std_multi', 'rand_single', 'rand_multi', 'mix_single','mix_multi']
    for usage in function:
        for traindir in filedir:
            get_sift_result_train(traindir=traindir,usage=usage)
            get_sift_result_test(traindir=traindir,usage=usage)
