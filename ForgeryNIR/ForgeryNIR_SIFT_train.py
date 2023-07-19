import torch
import time
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


def train_model(train_dir,usage):
    os.makedirs('./NIR_model/sift_model/' + usage, exist_ok=True)
    if usage=='binary_classification':
        model = model_arch.SIFTMLP(n_classes=2)
    if usage=='gan_classification':
        model = model_arch.SIFTMLP(n_classes=4)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model = model.cuda()
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    num_epoch = 15
    train_path='./SIFT_feature/train_des-'+train_dir+'.npy'
    train_data = np.load(train_path)
    val_path='./SIFT_feature/val_des-'+train_dir+'.npy'
    val_data = np.load(val_path)
    if usage=='binary_classification':
        train_label =[0] * 7000 + [1] * 28000
        train_label = np.array(train_label)
        val_label = [0] * 1000 + [1] * 4000
        val_label = np.array(val_label)
    if usage=='gan_classification':
        train_data=train_data[7000:]
        val_data=val_data[1000:]
        train_label =[0] * 7000 + [1] * 7000 +[2] * 7000+[3] * 7000
        train_label = np.array(train_label)
        val_label = [0] * 1000 +[1] * 1000 +[2] * 1000+[3] * 1000
        val_label = np.array(val_label)

    
    train_set = ImgDataset(train_data, train_label, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)    
    val_set = ImgDataset(val_data, val_label, transform=transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        length = train_loader.__len__()
        model.train()
        batch_start_time = time.time()
        load_time = 0
        for j, data in enumerate(train_loader):
            data[0] = data[0].type(torch.FloatTensor)
            optimizer.zero_grad()
            train_pred = model(data[0].cuda())
            batch_loss = loss(train_pred, data[1].cuda())
            batch_loss.backward()
            optimizer.step()
            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy()) 
            train_loss += batch_loss.item() 
            batch_load_time = time.time()
            if batch_load_time - batch_start_time > 10:
                load_time = batch_load_time - batch_start_time
            print("\rBatch_load time: {}, Process progress: {}% ".format(load_time, (j + 1) / length * 100), end="")
            batch_start_time = time.time()

        model.eval()
        with torch.no_grad():
            for j, data in enumerate(val_loader):
                data[0] = data[0].type(torch.FloatTensor)
                val_pred = model(data[0].cuda())
                batch_loss = loss(val_pred, data[1].cuda())

                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                val_loss += batch_loss.item()

            print(
                '[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f | learning_rate: %3.6f' % \
                (epoch + 1, num_epoch, time.time() - epoch_start_time,
                 train_acc / train_set.__len__(), train_loss / train_set.__len__(),
                 val_acc / val_set.__len__(),
                 val_loss / val_set.__len__(),
                 optimizer.state_dict()['param_groups'][0]['lr']))

        torch.save(model.module.state_dict(), './NIR_model/sift_model/' + usage + '/attention_model_sift_' + train_dir + str(epoch)+ '.pth')

    del train_set, train_loader, train_data

if __name__ == '__main__':
    function=['binary_classification','gan_classification']
    filedir=['std_single','std_multi','rand_single','rand_multi','mix_single','mix_multi']
    for usage in function:
        for traindir in filedir:
            train_model(train_dir=traindir,usage=usage)
