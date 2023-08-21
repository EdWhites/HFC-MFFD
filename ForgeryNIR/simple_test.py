import os

import numpy as np
import config as cfg
import argparse
import torch
import model_arch

batch_size=256
from torch.utils.data import DataLoader, Dataset
class FeatureDataset(Dataset):
    def __init__(self, x, y=None):
        super(FeatureDataset).__init__()
        self.x=x
        # label is required to be a LongTensor
        self.y=y
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        im=self.x[index]
        im=torch.Tensor(im)
        label=self.y[index]
        return im,label

def get_result(model,feature,usage):
    model.eval()
    if usage=='gan_classification':
        result = np.zeros((10000, 4))
        classes=4
        test_label = [0] * 2000+[1] * 2000 +[2] * 2000 +[3] * 2000+[4] * 2000
        test_label = np.array(test_label)
    if usage=='binary_classification':
        classes=2
        result=np.zeros((10000, 2))
        test_label = [0] * 2000 + [1] * 8000
        test_label = np.array(test_label)
    val_set = FeatureDataset(feature,test_label)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    t=0
    with torch.no_grad():
        for j, data in enumerate(val_loader):
            data[0] = data[0].type(torch.FloatTensor)
            val_pred = model(data[0].cuda())
            val_pred1 = val_pred.data.cpu().view(-1, classes).numpy()
            for i in val_pred1:
                result[t] = i
                t += 1
    return result

if __name__ == '__main__':
    label = [0] * 2000+[1] * 2000 +[2] * 2000 +[3] * 2000+[4] * 2000
    parser=argparse.ArgumentParser(description='ForgeryNIR_simple_test')
    parser.add_argument('--train_dir',type=str)
    parser.add_argument('--test_dir',type=str)
    args=parser.parse_args()
    model_path_dict,feature_path_dict=cfg.get_feature_and_model(traindir=args.train_dir,testdir=args.test_dir)

    resnet_feature_b=np.load(feature_path_dict['resnet_feature_path_b'])
    resnet_model_b=model_arch.AttentionResNet(n_classes=2,patch_num=9).cuda()
    resnet_model_b.load_state_dict(torch.load(model_path_dict['resnet_model_path_b']))
    score_resnet_b=get_result(model=resnet_model_b,feature=resnet_feature_b,usage='binary_classification')
    del resnet_feature_b

    resnet_dct_feature_b=np.load(feature_path_dict['resnet_dct_feature_path_b'])
    resnet_dct_model_b=model_arch.AttentionResNet(n_classes=2,patch_num=9).cuda()
    resnet_dct_model_b.load_state_dict(torch.load(model_path_dict['resnet_dct_model_path_b']))
    score_resnet_dct_b=get_result(model=resnet_dct_model_b,feature=resnet_dct_feature_b,usage='binary_classification')
    del resnet_dct_feature_b

    sift_feature=np.load(feature_path_dict['sift_feature_path'])
    sift_model_b=model_arch.SIFTMLP(n_classes=2).cuda()
    sift_model_b.load_state_dict(torch.load(model_path_dict['sift_model_path_b']))
    score_sift_b=get_result(model=sift_model_b,feature=sift_feature,usage='binary_classification')

    sift_model_g = model_arch.SIFTMLP(n_classes=4).cuda()
    sift_model_g.load_state_dict(torch.load(model_path_dict['sift_model_path_g']))
    score_sift_g=get_result(model=sift_model_g,feature=sift_feature,usage='gan_classification')

    resnet_model_g = model_arch.AttentionResNet(n_classes=4,patch_num=9).cuda()
    resnet_feature_g=np.load(feature_path_dict['resnet_feature_path_g'])
    resnet_model_g.load_state_dict(torch.load(model_path_dict['resnet_model_path_g']))

    score_resnet_g=get_result(model=resnet_model_g,feature=resnet_feature_g,usage='gan_classification')

    score_b= score_resnet_b+score_resnet_dct_b+score_sift_b
    pred = np.argmax(score_b,axis=1)
    acc=0
    score_g=score_resnet_g+score_sift_g
    for i in range(len(pred)):
        if pred[i]==1:
            pred[i] = np.argmax(score_g[i])+1
    for i in range(0, len(pred)):
        if pred[i] == label[i]:
            acc += 1

    print('the result for {} is {} %'.format(args.test_dir,acc))
