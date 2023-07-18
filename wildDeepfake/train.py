import os
from PIL import Image
import torch
import time
import torch.cuda
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
import config as cfg
from PIL import ImageFile
import model_arch
import utils.function as fn
ImageFile.LOAD_TRUNCATED_IMAGES=True
filename_mean=cfg.mean_path
filename_var=cfg.var_path
mean=np.load(filename_mean)
variance=np.load(filename_var)
variance=np.sqrt(variance)
batch_size = cfg.batch_size

transform = transforms.Compose([
    transforms.ToTensor(),
]
)

class PatchDataset(Dataset):
    def __init__(self, path, transform=None,dct=False):
        super(PatchDataset).__init__()
        self.path = path
        self.files = fn.get_filepath(path)
        self.dct=dct
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        fname = self.files[index]
        if self.dct:
            im = fn.load_image(fname, grayscale=True)
            im = torch.Tensor(im.copy())
        else:
            im=Image.open(fname)
            im=self.transform(im)

        if fname.split('/')[-3]=='real':
            label=1
        if fname.split('/')[-3]=='fake':
            label=0
        return im,label



def load_dataset(path, transform=None, dct=False):
    dataset = PatchDataset(path, transform=transform, dct=dct)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=16, pin_memory=True, sampler=sampler)
    return dataset,loader



def initialize_model(usage, patch_num,device,dct=False):
    local_rank=int(os.environ['LOCAL_RANK'])
    if dct:
        extract_feature_model = model_arch.ResNet_50_dct().to(device)
    else:
        extract_feature_model = model_arch.ResNet_50().to(device)
    if usage == 'binary_classification':
        classes = 2
    if usage == 'gan_classification':
        classes = 4
    mlp = model_arch.R_MLP(n_classes=classes).to(device)
    attention_model = model_arch.AttentionResNet(n_classes=classes, patch_num=patch_num).to(device)

    optimizer_et = torch.optim.Adam(filter(lambda p: p.requires_grad, extract_feature_model.parameters()), lr=cfg.lr_et)
    optimizer_at = torch.optim.Adam(filter(lambda p: p.requires_grad, attention_model.parameters()), lr=cfg.lr_at)
    optimizer_mlp = torch.optim.Adam(filter(lambda p: p.requires_grad, mlp.parameters()), lr=cfg.lr_mlp)

    if torch.cuda.device_count() > 1:
        extract_feature_model = torch.nn.parallel.DistributedDataParallel(extract_feature_model, device_ids=[local_rank],
                                                                           output_device=local_rank)
        mlp = torch.nn.parallel.DistributedDataParallel(mlp, device_ids=[local_rank],
                                                         output_device=local_rank)
        attention_model = torch.nn.parallel.DistributedDataParallel(attention_model, device_ids=[local_rank],
                                                                      output_device=local_rank)

    return extract_feature_model, mlp, attention_model, optimizer_et, optimizer_mlp, optimizer_at

def train_model(usage, dct=False,patch_size=128,arfa=0.6):
    #------------------------------#
    if patch_size!=224:
        patch_num,stride=fn.cal_stride(cfg.image_size, patch_size=patch_size)   #height of the images in the WildDeepfake dataset
    #------------------------------#
    else:
        patch_num=1

    print(os.environ['LOCAL_RANK'])
    
    if int(os.environ['LOCAL_RANK']) != -1:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        device = torch.device("cuda", int(os.environ['LOCAL_RANK']))
        torch.distributed.init_process_group(backend="nccl")
    if dct:
        channel=1
    else:
        channel=3
    logger=fn.get_logger(cfg.log_path+'/'+usage+'/train.log')
    logger.info(('start training!'))
    loss = torch.nn.CrossEntropyLoss()

    
    train_set,train_loader = load_dataset(cfg.train_datasets_path, transform=transform, dct=dct)
    val_set,val_loader = load_dataset(cfg.test_datasets_path, transform=transform, dct=dct)

    extract_feature_model, mlp, attention_model, optimizer_et, optimizer_mlp, optimizer_at = initialize_model(usage, patch_num,device,dct)


    for epoch in range(cfg.num_epoch):
        train_loader.sampler.set_epoch(epoch)
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
            if patch_size!=data[0].shape[2]:
                data[0] = fn.unfold_img(data[0],channel=channel,stride=stride,patch_size=patch_size)
            if dct:
                data[0]=fn.turn_to_dct(data[0],mean=mean,variance=variance)
            patch_label = fn.prepare_patch_label(label=data[1],patch_num=patch_num)

            feature = extract_feature_model(data[0].to(device))
            train_pred_patch=mlp(feature)
            optimizer_et.zero_grad()
            optimizer_mlp.zero_grad()
            loss_et=loss(train_pred_patch,patch_label.to(device))
            optimizer_at.zero_grad()
            train_pred=attention_model(feature)
            loss_at = loss(train_pred, data[1].to(device))
            loss_all=arfa*loss_et+(1-arfa)*loss_at
            loss_all.backward()
            optimizer_mlp.step()
            optimizer_et.step()
            optimizer_at.step()
            train_acc_1=torch.eq(torch.max(train_pred,1)[1],data[1].to(device)).sum()
            loss_all=fn.reduce_mean(loss_all,torch.distributed.get_world_size())
            torch.distributed.all_reduce(train_acc_1,op=torch.distributed.ReduceOp.SUM)
            train_acc+=train_acc_1.item()
            train_loss += loss_all.item()  # 计算所有的loss
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
                if patch_size!=data[0].shape[2]:
                    data[0]=fn.unfold_img(data[0],channel=channel,stride=stride,patch_size=patch_size)
                if dct:
                    data[0] = fn.turn_to_dct(data[0], mean=mean, variance=variance)
                patch_label = fn.prepare_patch_label(label=data[1],patch_num=patch_num)

                feature = extract_feature_model(data[0].to(device))
                val_pred_patch = mlp(feature)
                val_pred = attention_model(feature)
                batch_loss = loss(val_pred, data[1].to(device))
                val_acc_mlp_1 = torch.eq(torch.max(val_pred_patch, 1)[1], patch_label.to(device)).sum()
                torch.distributed.all_reduce(val_acc_mlp_1,op=torch.distributed.ReduceOp.SUM)
                val_acc1 = torch.eq(torch.max(val_pred, 1)[1], data[1].to(device)).sum()
                torch.distributed.all_reduce(val_acc1,op=torch.distributed.ReduceOp.SUM)
                val_acc_mlp+=val_acc_mlp_1.item()
                val_acc += val_acc1.item()
                batch_loss = fn.reduce_mean(batch_loss, torch.distributed.get_world_size())
                val_loss += batch_loss.item()
                del feature
            logger.info('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Patch Acc: %3.6f Val Acc: %3.6f loss: %3.6f' % \
                (epoch + 1, cfg.num_epoch, time.time() - epoch_start_time, \
                 train_acc / train_set.__len__(), train_loss / train_set.__len__(),
                 val_acc_mlp / (patch_num * val_set.__len__()), val_acc / val_set.__len__(),
                 val_loss / val_set.__len__()))


        if dct == True:
            torch.save(extract_feature_model.state_dict(), cfg.output_model_path+'/'+usage+'/extract_feature_model_resnet_dct_epoch'+str(epoch)+'.pth')
            torch.save(mlp.state_dict(),cfg.output_model_path+'/'+usage+'/mlp_resnet_dct_epoch'+str(epoch)+'.pth')
            torch.save(attention_model.state_dict(), cfg.output_model_path+'/'+usage+'/attention_model_resnet_dct_epoch'+str(epoch)+'.pth')
        else:
            torch.save(extract_feature_model.state_dict(), cfg.output_model_path+'/'+usage+'/extract_feature_model_resnet_epoch'+str(epoch)+'.pth')
            torch.save(mlp.state_dict(),cfg.output_model_path+'/'+usage+'/mlp_resnet_epoch'+str(epoch)+'.pth')
            torch.save(attention_model.state_dict(), cfg.output_model_path+'/'+usage+'/attention_model_resnet_epoch'+str(epoch)+'.pth')
    logger.info('finish training')    

if __name__ == "__main__":
    usage=cfg.usage
    os.makedirs("./et_and_at_model/"+usage,exist_ok=True)
    print('The num of GPU:',torch.cuda.device_count())
    train_model(usage=usage,patch_size=cfg.patch_size,arfa=cfg.arfa)

