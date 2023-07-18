import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import ImageFile
import cv2
import dlib
import model_arch
import utils.function as fn
import itertools
ImageFile.LOAD_TRUNCATED_IMAGES=True

#-----------------should be changed according to your situation-------------------------#
filename_mean='./mean.npy'
filename_var='./var.npy'
img_to_tensor=transforms.ToTensor()
mean=np.load(filename_mean)
variance=np.load(filename_var)
variance=np.sqrt(variance)
batch_size = 128
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
sift = cv2.SIFT_create()
#----------------------------v-----------------------------------------------------------#

transform = transforms.Compose([
    transforms.ToTensor(),
]
)
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


#-------- kernal_size and stride should be changed if you have different arguments. ---#
def unfold_img(images,channel):
    unfold=torch.nn.Unfold(kernel_size=(128,128),stride=(48,48))
    images = images.type(torch.FloatTensor)
    images = images.view(-1, channel, 224, 224)
    images = unfold(images)
    images = images.transpose(2, 1)
    images = images.contiguous().reshape(-1, 3,3,channel, 128, 128).transpose(2,1).reshape(-1,channel,128,128)
    return images

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
            im_sift=compute_feature(path=fname)
            im_sift=torch.Tensor(im_sift.copy())
        if self.origin:
            im=Image.open(fname)
            im=self.transform(im)

        if fname.split('/')[-3]=='real':
            label=1
        if fname.split('/')[-3]=='fake':
            label=0

        return label,im,im_dct,im_sift



def test_model(usage, origin=True,dct=False,sift=False,weight_origin=1,weight_dct=1,weight_sift=1):
    print(os.environ['LOCAL_RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if int(os.environ['LOCAL_RANK']) != -1:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        device = torch.device("cuda", int(os.environ['LOCAL_RANK']))
        torch.distributed.init_process_group(backend="nccl")
    loss = torch.nn.CrossEntropyLoss()
    if usage=='binary_classification':
        classes=2
    if usage=='gan_classification':
        classes=4
    # --------- patch_num should be changed if you have a different one. -------#
    if dct:
        extract_feature_model_dct=model_arch.ResNet_50_dct().to(device)
        attention_model_dct=model_arch.AttentionResNet(n_classes=classes,patch_num=9).to(device)
        extract_feature_model_dct.load_state_dict(torch.load('./WildDeepFake/model/extract_feature_model_resnet_dct.pth'))
        attention_model_dct.load_state_dict(torch.load('./WildDeepFake/model/attention_model_resnet_dct.pth'))
    if sift:
        attention_model_sift=model_arch.SIFTMLP(n_classes=classes).to(device)
        attention_model_sift.load_state_dict(torch.load('./WildDeepFake/model/attention_model_sift.pth'))
    if origin:
        extract_feature_model=model_arch.ResNet_50().to(device)
        attention_model=model_arch.AttentionResNet(n_classes=classes,patch_num=9).to(device)
        extract_feature_model.load_state_dict(torch.load('./WildDeepFake/model/extract_feature_model_resnet.pth'))
        attention_model.load_state_dict(torch.load('./WildDeepFake/model/attention_model_resnet.pth'))

    num_gpus = torch.cuda.device_count()
    val_set = PatchDataset('/data/zzy/datasets/test', transform=transform, dct=dct,sift=sift)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16,pin_memory=True,shuffle=False)
    val_acc = 0.0
    val_loss = 0.0
    val_pred_sift=None
    val_pred_dct=None
    val_sampler=torch.utils.data.distributed.DistributedSampler(val_set)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16,pin_memory=True,sampler=val_sampler)
    if num_gpus > 1:
        if origin:
            extract_feature_model=torch.nn.parallel.DistributedDataParallel(extract_feature_model,device_ids=[local_rank],
                                                        output_device=local_rank)
            attention_model=torch.nn.parallel.DistributedDataParallel(attention_model,device_ids=[local_rank],
                                                        output_device=local_rank)
        if dct:
            extract_feature_model_dct=torch.nn.parallel.DistributedDataParallel(extract_feature_model_dct,device_ids=[local_rank],
                                                        output_device=local_rank)
            attention_model_dct=torch.nn.parallel.DistributedDataParallel(attention_model_dct,device_ids=[local_rank],
                                                        output_device=local_rank)
        if sift:
            attention_model_sift=torch.nn.parallel.DistributedDataParallel(attention_model_sift,device_ids=[local_rank],
                                                        output_device=local_rank)

    attention_model.eval()
    extract_feature_model.eval()
    extract_feature_model_dct.eval()
    attention_model_dct.eval()
    attention_model_sift.eval()

    with torch.no_grad():
        for j, data in enumerate(val_loader):
            if sift:
                val_pred_sift=attention_model_sift(data[3])
            if dct:
                data[2]=unfold_img(data[2],channel=1)
                data[2]=fn.turn_to_dct(data[2],mean=mean,variance=variance)
                feature = extract_feature_model_dct(data[2].to(device))
                val_pred_dct = attention_model_dct(feature)
            if origin:
                data[1]=unfold_img(data[1],channel=3)
                feature = extract_feature_model(data[1].to(device))
                val_pred = attention_model(feature)
            val_pred =weight_dct*val_pred_dct+weight_origin*val_pred+weight_sift*val_pred_sift      
            batch_loss = loss(val_pred, data[0].to(device))

            val_acc_tmp = torch.eq(torch.max(val_pred, 1)[1], data[0].to(device)).sum()

            torch.distributed.all_reduce(val_acc_tmp, op=torch.distributed.ReduceOp.SUM)

            val_acc += val_acc_tmp.item()
            batch_loss = fn.reduce_mean(batch_loss, torch.distributed.get_world_size())
            val_loss += batch_loss.item()
            del feature
        print(
            'Val Acc: %3.6f loss: %3.6f' % \
            ( val_acc / val_set.__len__(),
             val_loss / val_set.__len__()))




if __name__ == "__main__":
    test_model(usage='binary_classification',dct=True,origin=True,sift=True)

