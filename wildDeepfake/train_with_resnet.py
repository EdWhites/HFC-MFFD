import os
from PIL import Image
import torch.distributed as dist
import torch.nn as nn
import torch
import time
import torch.cuda
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import ImageFile
import model_arch
import function
import config as cfg

ImageFile.LOAD_TRUNCATED_IMAGES=True

transform = transforms.Compose([
    transforms.ToTensor(),
]
)

class PatchDataset(Dataset):         # designed for wilddeepfake dataset; can be changed for other usage
    def __init__(self, path, transform=None,dct=False):
        super(PatchDataset).__init__()
        self.path = path
        self.files = function.get_filepath(path)
        self.dct = dct
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        fname = self.files[index]
        if self.dct:
            im = function.load_image(fname, grayscale=True)
            im = torch.Tensor(im.copy())
        else:
            im=Image.open(fname)
            im=self.transform(im)

        if fname.split('/')[-3]=='real':
            label=1
        if fname.split('/')[-3]=='fake':
            label=0
        return im,label


def train_model(usage, dct=False):
    local_rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['LOCAL_RANK']) != -1:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        device = torch.device("cuda", int(os.environ['LOCAL_RANK']))
        torch.distributed.init_process_group(backend="nccl")

    logger=function.get_logger(cfg.log_path)
    logger.info(('start training!'))

    if usage=='binary_classification':
        classes=2
    if usage=='gan_classification':
        classes=4

    # ------- Initialize models -------- #
    if dct:
        extract_feature_model =model_arch.ResNet_50_dct().to(device)
        channel=1
        # ------- normalization for DCT spectrum -------- #
        mean=np.load(cfg.mean_path)
        variance=np.load(cfg.var_path)
        variance=np.sqrt(variance)
    else:
        extract_feature_model =model_arch.ResNet_50().to(device)
        channel=3
    mlp=model_arch.R_MLP(n_classes=classes).to(device)
    attention_model=model_arch.AttentionResNet(n_classes=classes).to(device)

    optimizer_et= torch.optim.Adam(filter(lambda p: p.requires_grad, extract_feature_model.parameters()), lr=cfg.lr_et)
    optimizer_at = torch.optim.Adam(filter(lambda p: p.requires_grad, attention_model.parameters()), lr=cfg.lr_at)
    optimizer_mlp=torch.optim.Adam(filter(lambda p: p.requires_grad,mlp.parameters()), lr=cfg.lr_mlp)

    # ---- Prepare dataset ---- #
    train_set = PatchDataset(cfg.train_datasets_path, transform=transform,dct=dct)
    val_set = PatchDataset(cfg.test_datasets_path, transform=transform,dct=dct)
    
    num_gpus = torch.cuda.device_count()
    train_sampler=torch.utils.data.distributed.DistributedSampler(train_set)
    val_sampler=torch.utils.data.distributed.DistributedSampler(val_set)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, num_workers=16,pin_memory=True,sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, num_workers=16,pin_memory=True,sampler=val_sampler)
    loss = nn.CrossEntropyLoss()

    # ---- Prepare Training for Multi-GPUs ---- #
    if num_gpus > 1:
        logger.info('use {} gpus!'.format(num_gpus))
        extract_feature_model=torch.nn.parallel.DistributedDataParallel(extract_feature_model,device_ids=[local_rank],
                                                    output_device=local_rank)
        mlp=torch.nn.parallel.DistributedDataParallel(mlp,device_ids=[local_rank],
                                                    output_device=local_rank)
        attention_model=torch.nn.parallel.DistributedDataParallel(attention_model,device_ids=[local_rank],
                                                    output_device=local_rank)

    for epoch in range(cfg.num_epoch):
        # ---- Train the models ---- #
        train_sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_acc_mlp=0.0
        val_loss = 0.0
        extract_feature_model.train()
        mlp.train()
        attention_model.train()
        length=train_loader.__len__()
        for j, data in enumerate(train_loader):
            data[0] = function.unfold_img(data[0],channel=channel)
            if dct:
                data[0]=function.turn_to_dct(data[0],mean=mean,variance=variance)
            feature = extract_feature_model(data[0].to(device))
            patch_label=function.prepare_patch_label(batch_label=data[1])
            train_pred_patch=mlp(feature)
            optimizer_et.zero_grad()
            optimizer_mlp.zero_grad()
            loss_et=loss(train_pred_patch,patch_label.to(device))
            optimizer_at.zero_grad()
            train_pred=attention_model(feature)

            loss_at = loss(train_pred, data[1].to(device))
            loss_all=cfg.arfa*loss_et+(1-cfg.arfa)*loss_at
            loss_all.backward()

            optimizer_mlp.step()
            optimizer_et.step()
            optimizer_at.step()

            train_acc_batch=torch.eq(torch.max(train_pred,1)[1],data[1].to(device)).sum()
            loss_all=function.reduce_mean(loss_all,dist.get_world_size())
            dist.all_reduce(train_acc_batch,op=dist.ReduceOp.SUM)
            train_acc+=train_acc_batch.item()
            train_loss += loss_all.item() 
            del feature,train_pred_patch,train_pred
            print("\rProcess progress: {}% ".format((j + 1) / length * 100), end="")

        # ---- Test the models ---- #
        attention_model.eval()
        extract_feature_model.eval()
        mlp.eval()

        with torch.no_grad():
            for j, data in enumerate(val_loader):
                data[0]=function.unfold_img(data[0],channel)
                if dct:
                    data[0]=function.turn_to_dct(data[0],mean=mean,variance=variance)
                feature = extract_feature_model(data[0].to(device))
                patch_label=function.prepare_patch_label(batch_label=data[1])
                val_pred_patch = mlp(feature)
                val_pred = attention_model(feature)
                batch_loss = loss(val_pred, data[1].to(device))
                val_acc_mlp_batch = torch.eq(torch.max(val_pred_patch, 1)[1], patch_label.to(device)).sum()
                dist.all_reduce(val_acc_mlp_batch,op=dist.ReduceOp.SUM)
                val_acc1 = torch.eq(torch.max(val_pred, 1)[1], data[1].to(device)).sum()
                dist.all_reduce(val_acc1,op=dist.ReduceOp.SUM)
                val_acc_mlp+=val_acc_mlp_batch.item()
                val_acc += val_acc1.item()
                batch_loss = function.reduce_mean(batch_loss, dist.get_world_size())
                val_loss += batch_loss.item()
                del feature
            logger.info('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Patch Acc: %3.6f Val Acc: %3.6f loss: %3.6f' % \
                (epoch + 1, cfg.num_epoch, time.time() - epoch_start_time, \
                 train_acc / train_set.__len__(), train_loss / train_set.__len__(),
                 val_acc_mlp / (cfg.patch_size * val_set.__len__()), val_acc / val_set.__len__(),
                 val_loss / val_set.__len__()))
        
        # ---- Save the models ---- #
        if dct == True:
            torch.save(extract_feature_model, cfg.output_model_path+'/'+usage+'/extract_feature_model_resnet_dct_epoch'+str(epoch)+'.pkl')
            torch.save(mlp,cfg.output_model_path+'/'+usage+'/mlp_resnet_dct_epoch'+str(epoch)+'.pkl')
            torch.save(attention_model, cfg.output_model_path+'/'+usage+'/attention_model_resnet_dct_epoch'+str(epoch)+'.pkl')
        else:
            torch.save(extract_feature_model, cfg.output_model_path+'/'+usage+'/extract_feature_model_resnet_epoch'+str(epoch)+'.pkl')
            torch.save(mlp,cfg.output_model_path+'/'+usage+'/mlp_resnet_epoch'+str(epoch)+'.pkl')
            torch.save(attention_model, cfg.output_model_path+'/'+usage+'/attention_model_resnet_epoch'+str(epoch)+'.pkl')
    logger.info('finish training')    

if __name__ == "__main__":
    os.makedirs(cfg.output_model_path+'/'+cfg.usage,exist_ok=True)
    print('The num of GPU:',torch.cuda.device_count())
    train_model(usage=cfg.usage,dct=True)

