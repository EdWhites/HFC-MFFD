import torch
import time
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import dlib
import cv2
import os
import utils.function as fn
import model_arch
import config as cfg
import itertools
batch_size = cfg.batch_size_sift
transform = transforms.Compose([
    transforms.ToTensor()]
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
    def __init__(self, path, transform=None):
        super(PatchDataset).__init__()
        self.path = path
        self.files = fn.get_filepath(path)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        fname = self.files[index]
        im=compute_feature(fname)
        im=torch.Tensor(im)
        if fname.split('/')[-3]=='real':
            label=1
        if fname.split('/')[-3]=='fake':
            label=0

        return im,label

def train_model(usage):
    print(os.environ['LOCAL_RANK'])
    local_rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['LOCAL_RANK']) != -1:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        device = torch.device("cuda", int(os.environ['LOCAL_RANK']))
        torch.distributed.init_process_group(backend="nccl")

    logger=fn.get_logger(cfg.log_path_sift)
    logger.info(('start training!'))
    loss = torch.nn.CrossEntropyLoss()
    if usage=='binary_classification':
        classes=2
    if usage=='gan_classification':
        classes=4
    attention_model=model_arch.SIFTMLP(n_classes=classes).to(device)

    optimizer_at_sift= torch.optim.Adam(filter(lambda p: p.requires_grad, attention_model.parameters()), lr=cfg.lr_sift)
    num_epoch = cfg.num_epoch
    train_set = PatchDataset(cfg.train_datasets_path)
    val_set = PatchDataset(cfg.test_datasets_path)

    num_gpus = torch.cuda.device_count()
    print(device)
    train_sampler=torch.utils.data.distributed.DistributedSampler(train_set)
    val_sampler=torch.utils.data.distributed.DistributedSampler(val_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=16,pin_memory=True,sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16,pin_memory=True,sampler=val_sampler)
    if num_gpus > 1:
        logger.info('use {} gpus!'.format(num_gpus))
        attention_model=torch.nn.parallel.DistributedDataParallel(attention_model,device_ids=[local_rank],
                                                    output_device=local_rank)

    for epoch in range(num_epoch):
        train_sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        attention_model.train()
        batch_start_time = time.time()
        load_time=0
        length=train_loader.__len__()
        for j, data in enumerate(train_loader):
            data[0] = data[0].type(torch.FloatTensor)
            optimizer_at_sift.zero_grad()
            data[0]=data[0].view(-1,68,128)
            train_pred=attention_model(data[0])
            loss_at = loss(train_pred, data[1].to(device))
            loss_at.backward()
            optimizer_at_sift.step()
            train_acc_1=torch.eq(torch.max(train_pred,1)[1],data[1].to(device)).sum()
            loss_at=fn.reduce_mean(loss_at,torch.distributed.get_world_size())
            torch.distributed.all_reduce(train_acc_1,op=torch.distributed.ReduceOp.SUM)
            train_acc+=train_acc_1.item()
            train_loss += loss_at.item()  # 计算所有的loss
            batch_load_time=time.time()
            if batch_load_time-batch_start_time>10:
                load_time=batch_load_time-batch_start_time
            print("\rBatch_load time: {}, Process progress: {}% ".format(load_time,(j + 1) / length * 100), end="")
            batch_start_time=time.time()
        attention_model.eval()
        with torch.no_grad():
            for j, data in enumerate(val_loader):
                data[0] = data[0].type(torch.FloatTensor)
                val_pred = attention_model(data[0])
                batch_loss = loss(val_pred, data[1].to(device))
                val_acc1 = torch.eq(torch.max(val_pred, 1)[1], data[1].to(device)).sum()
                torch.distributed.all_reduce(val_acc1,op=torch.distributed.ReduceOp.SUM)
                val_acc += val_acc1.item()
                batch_loss = fn.reduce_mean(batch_loss, torch.distributed.get_world_size())
                val_loss += batch_loss.item()
            logger.info('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                (epoch + 1, num_epoch, time.time() - epoch_start_time, \
                 train_acc / train_set.__len__(), train_loss / train_set.__len__(),
                 val_acc / val_set.__len__(),
                 val_loss / val_set.__len__()))
        
        torch.save(attention_model.state_dict(), cfg.output_sift_model_path+'/'+usage+'/attention_model_sift_epoch'+str(epoch)+'.pth')
    logger.info('finish training')

def train_and_test_model(usage=None):
    os.makedirs(cfg.output_sift_model_path+"/"+usage,exist_ok=True)
    print('The num of GPU:',torch.cuda.device_count())
    train_model(usage)

if __name__ == "__main__":
    train_and_test_model(usage=cfg.usage)
