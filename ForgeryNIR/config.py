

def get_feature_and_model(traindir,testdir):
    # ------------- model path----------------------------- #
    resnet_model_path_b='./model/resnet_model/binary_classification/attention_model_resnet_'+traindir+'.pth'
    resnet_dct_model_path_b='./model/resnet_dct_model/binary_classification/attention_model_resnet_dct_'+traindir+'.pth'
    sift_model_path_b='./model/sift_model/binary_classification/attention_model_sift_'+traindir+'.pth'
    resnet_model_path_g='./model/resnet_model/gan_classification/attention_model_resnet_'+traindir+'.pth'
    sift_model_path_g= './model/sift_model/gan_classification/attention_model_sift_'+traindir+'.pth'
    model_path_dict=dict(resnet_model_path_b=resnet_model_path_b,resnet_model_path_g=resnet_model_path_g,resnet_dct_model_path_b=resnet_dct_model_path_b,sift_model_path_b=sift_model_path_b,sift_model_path_g=sift_model_path_g)

    # ------------- feature path----------------------------- #
    resnet_feature_path_b='./feature/ResNet/binary_classification/'+traindir+'/resnet_feature_'+testdir+'.npy'
    resnet_dct_feature_path_b='/feature/ResNet_DCT/binary_classification/'+traindir+'/resnet_feature_'+testdir+'.npy'
    resnet_feature_path_g='/feature/ResNet/gan_classification/'+traindir+'/resnet_feature_'+testdir+'.npy'
    sift_feature_path='./feature/SIFT/test_des-' + testdir + ".npy"
    feature_path_dict=dict(resnet_feature_path_b=resnet_feature_path_b,resnet_dct_feature_path_b=resnet_dct_feature_path_b,resnet_feature_path_g=resnet_feature_path_g,sift_feature_path=sift_feature_path)
    return model_path_dict,feature_path_dict
    
    

    