from PIL import Image
import numpy as np
from scipy import fftpack
import os
import torch
import torch_dct
import config as cfg
import logging
import torch.distributed as dist
import cv2
import dlib
import itertools
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
sift = cv2.SIFT_create()

def _welford_update(existing_aggregate, new_value):
    (count, mean, M2) = existing_aggregate
    if count is None:
        count, mean, M2 = 0, np.zeros_like(new_value), np.zeros_like(new_value)

    count += 1
    delta = new_value - mean
    mean = mean+delta / count
    delta2 = new_value - mean
    M2 = M2+delta * delta2

    return (count, mean, M2)


def _welford_finalize(existing_aggregate):
    count, mean, M2 = existing_aggregate
    mean, variance, sample_variance = (mean, M2/count, M2/(count - 1))
    if count < 2:
        return (float("nan"), float("nan"), float("nan"))
    else:
        return (mean, variance, sample_variance)


def welford(sample):
    """Calculates the mean, variance and sample variance along the first axis of an array.
    Taken from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    existing_aggregate = (None, None, None)
    for data in sample:
        existing_aggregate = _welford_update(existing_aggregate, data)

    # sample variance only for calculation
    return _welford_finalize(existing_aggregate)[:-1]


def welford_multidimensional(sample):
    """Same as normal welford but for multidimensional data, computes along the last axis.
    """
    aggregates = {}

    for data in sample:
        # for each sample update each axis seperately
        for i, d in enumerate(data):
            existing_aggregate = aggregates.get(i, (None, None, None))
            existing_aggregate = _welford_update(existing_aggregate, d)
            aggregates[i] = existing_aggregate

    means, variances = list(), list()

    # in newer python versions dicts would keep their insert order, but legacy
    for i in range(len(aggregates)):
        aggregate = aggregates[i]
        mean, variance = _welford_finalize(aggregate)[:-1]
        means.append(mean)
        variances.append(variance)

    return np.asarray(means), np.asarray(variances)

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

def get_filepath(filepath,path_read=None):
    if path_read==None:
        path_read=[]
    temp_list=os.listdir(filepath)
    for temp_list_each in temp_list:
        if os.path.isfile(filepath+'/'+temp_list_each):
            temp_path=filepath+'/'+temp_list_each
            if os.path.splitext(temp_path)[-1]=='.png':
                path_read.append(temp_path)
            else:
                continue
        else:
            path_read=get_filepath(filepath+'/'+temp_list_each,path_read)
    return path_read


def turn_to_dct(img,mean,variance):
    img = torch_dct.dct_2d(img, norm='ortho')
    img = (img.cpu().numpy() - mean) / variance
    img = torch.Tensor(img)
    return img

def dct2(array):
    array = fftpack.dct(array, type=2, norm="ortho", axis=0)
    array = fftpack.dct(array, type=2, norm="ortho", axis=1)
    return array


def load_image(path, grayscale=True, tf=False):
    x = Image.open(path)
    if grayscale:
        x = x.convert("L")
        if tf:
            x = np.asarray(x)
            x = np.reshape(x, [*x.shape, 1])
    return np.asarray(x)

def get_mean_and_var(path):
    imgpath=get_filepath(path)  # get all imgpath end with 'png' in the given file path
    images=map(load_image,imgpath)
    images=map(dct2,imgpath)
    mean,var=welford(images)
    np.save('./mean.npy',mean)
    np.save('./var.npy',var)


def unfold_img(batch,channel):
    unfold = nn.Unfold(kernel_size=(128, 128), stride=(48, 48))
    batch = batch.type(torch.FloatTensor)
    batch = batch.view(-1, channel, 224, 224)
    batch = unfold(batch)
    batch = batch.transpose(2, 1)
    batch = batch.contiguous().view(-1, channel, 128, 128)
    return batch

def prepare_patch_label(batch_label):
    patch_label=np.zeros((cfg.batch_size * cfg.patch_size))
    for i in range(cfg.batch_size):
        for num in range(i*cfg.patch_size,(i+1)*cfg.patch_size):
            patch_label[num]=batch_label[i]
    patch_label=torch.LongTensor(patch_label)
    return patch_label

def reduce_mean(tensor,nprocs):
    rt=tensor.clone()
    dist.all_reduce(rt,op=dist.ReduceOp.SUM)
    rt/=nprocs
    return rt

def new_shuffle(x):
    for i in range(x.shape[0]):
        index = torch.randperm(9)
        x[i]=x[i][index]
    return x

def get_keypoint(shape):
    d = []
    for i in range(0, 68):
        x = shape.part(i).x
        y = shape.part(i).y
        d.append((x, y))
    keypoint = cv2.KeyPoint_convert(d)
    return keypoint

def compute_feature(path):
    image = cv2.imread(path, 0)
    startx = 0
    starty = 0
    endx, endy = image.shape
    rect = dlib.rectangle(startx, starty, endx, endy)
    shape = predictor(image, rect)
    kp = get_keypoint(shape)
    kp, des = sift.compute(image, kp)
    x = list(itertools.chain(*des))
    return x