import cv2
import os
import itertools
import dlib
import numpy as np

i = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
sift = cv2.SIFT_create()


def get_keypoint(shape):
    d = []
    for i in range(0, 68):
        x = shape.part(i).x
        y = shape.part(i).y
        d.append((x, y))
    keypoint = cv2.KeyPoint_convert(d)
    return keypoint

def compute_feature(path, des_save, num):
    image = cv2.imread(path, 0)
    startx = 0
    starty = 0
    endx, endy = image.shape
    rect = dlib.rectangle(startx, starty, endx, endy)
    shape = predictor(image, rect)
    kp = get_keypoint(shape)
    kp, des = sift.compute(image, kp)
    x = list(itertools.chain(*des))
    des_save[num] = x
    return des_save,num

def get_sift_feature(dir,real_path=None,forgery_path=None,train_val_test=None):
    os.makedirs('./NIR_feature',exist_ok=True)
    if real_path is None:
        real_path = "./NIR_VIS 2.0/"+train_val_test
    num = 0
    file_dir = sorted(os.listdir(real_path))
    label = []
    des_save = []
    for i, file in enumerate(file_dir):
        des_save, num = compute_feature(os.path.join(real_path, file), des_save=des_save, num=num)
        print("\rProcess progress: {}% ".format((i + 1) / len(file_dir) * 100), end="")
    print('Real has completed')
    if forgery_path is None:
        forgery_path = './ForgeryNIR/ForgeryNIR-' + dir + '/'+train_val_test
    file = ['cyclegan', 'progan', 'stylegan', 'stylegan2']
    for ganfile in file:
        path = os.path.join(forgery_path, ganfile)
        file_dir = sorted(os.listdir(path))
        for i, file in enumerate(file_dir):
            des_save, num = compute_feature(os.path.join(path, file), des_save=des_save,num=num)
            print("\rProcess progress: {}% ".format((i + 1) / len(file_dir) * 100), end="")
    des_name = "./NIR_feature/"+train_val_test+"_des-" + dir + ".npy"
    np.save(des_name, des_save)
    del des_save
    del label


if __name__ == '__main__':
    filedir=['std_single', 'std_multi', 'rand_single', 'rand_multi', 'mix_single','mix_multi']
    for dir in filedir:
        get_sift_feature(dir=dir,train_val_test='test')
