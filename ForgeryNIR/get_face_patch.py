import os
import cv2

filedir=['std_single','std_multi','rand_single','rand_multi','mix_single','mix_multi']
filename=['cyclegan','progan','stylegan','stylegan2']

def get_nine_patch(img,outpath,num):
    x, y, _ = img.shape
    x_half = x // 2
    y_half = y // 2
    img1 = img[0:128, 0:128]
    cv2.imwrite(outpath + '/' + str(num) + '.png', img1)
    num += 1
    img2 = img[x_half - 64:x_half + 64, 0:128]
    cv2.imwrite(outpath + '/' + str(num) + '.png', img2)
    num += 1
    img3 = img[-128:, 0:128]
    cv2.imwrite(outpath + '/' + str(num) + '.png', img3)
    num += 1
    img4 = img[:128, y_half - 64:y_half + 64]
    cv2.imwrite(outpath + '/' + str(num) + '.png', img4)
    num += 1
    img5 = img[x_half - 64:x_half + 64, y_half - 64:y_half + 64]
    cv2.imwrite(outpath + '/' + str(num) + '.png', img5)
    num += 1
    img6 = img[-128:, y_half - 64:y_half + 64]
    cv2.imwrite(outpath + '/' + str(num) + '.png', img6)
    num += 1
    img7 = img[:128, -128:]
    cv2.imwrite(outpath + '/' + str(num) + '.png', img7)
    num += 1
    img8 = img[x_half - 64:x_half + 64, -128:]
    cv2.imwrite(outpath + '/' + str(num) + '.png', img8)
    num += 1
    img9 = img[-128:, -128:]
    cv2.imwrite(outpath + '/' + str(num) + '.png', img9)
    num += 1
    return num

def get_fake_face_patch(path=None,outpath=None,train_val_test=None):
    for dir in filedir:
        for fn in filename:
            num=0
            if path is None:
                path='./ForgeryNIR/ForgeryNIR-'+dir+'/'+train_val_test+'/'+fn
            if outpath is None:
                outpath='./NIR_Forgery_Face_Patch/'+train_val_test+'/'+dir+'/'+fn
            os.makedirs(outpath, exist_ok=True)
            imgdir=sorted(os.listdir(path))
            for i,file in enumerate(imgdir):
                img=cv2.imread(os.path.join(path,file))
                num=get_nine_patch(img,outpath,num)
                print("\rProcess progress: {}% ".format((i + 1) / len(imgdir) * 100), end="")

def get_real_patch(train_val_test):
    num = 0
    path = './NIR_VIS 2.0/'+train_val_test
    outpath = './NIR_Face_Patch/' + train_val_test
    os.makedirs(outpath, exist_ok=True)
    imgdir = sorted(os.listdir(path))
    for i, file in enumerate(imgdir):
        img = cv2.imread(os.path.join(path, file))
        num=get_nine_patch(img,outpath,num)
        print("\rProcess progress: {}% ".format((i + 1) / len(imgdir) * 100), end="")

if __name__ == '__main__':
    get_fake_face_patch(train_val_test='train')
    get_fake_face_patch(train_val_test='test')
    get_fake_face_patch(train_val_test='val')