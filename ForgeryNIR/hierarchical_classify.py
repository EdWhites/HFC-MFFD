import numpy as np

def gan_classify(traindir,testdir,resnet_g=False,resnet_dct_g=False,sift_g=False):

    score_resnet_g = 0
    score_resnet_dct_g = 0
    score_sift_g = 0

    label = [0] * 2000 + [1] * 2000 + [2] * 2000 + [3] * 2000
    path=''

    if resnet_g:
        path_g = './NIR_result/gan_classification/resnet/' + traindir + '/resnet_result_' + testdir + '_test.npy'
        score_resnet_g = get_score(path_g)
        path += 'resnet+'


    if resnet_dct_g:
        path_g = './NIR_result/gan_classification/resnet_dct/' + traindir + '/resnet_result_' + testdir + '_test.npy'
        score_resnet_dct_g = get_score(path_g)
        path += 'resnet_dct+'


    if sift_g:
        path_g = './NIR_result/gan_classification/sift/' + traindir + '/sift_result_' + testdir + '_test.npy'
        score_sift_g = get_score(path_g)
        path += 'sift+'
    score_b= score_resnet_g+score_resnet_dct_g+score_sift_g
    score_b=score_b[2000:]
    binary_pred = np.argmax(score_b,
                            axis=1)
    acc=0

    for i in range(0, len(binary_pred)):
        if binary_pred[i] == label[i]:
            acc += 1
    print('{}: Train with {},Test with {}, Acc is {}'.format( path[:-1],traindir, testdir,
                                                             acc / len(binary_pred)))

def get_score(filepath):
    score=np.load(filepath)

    return score
def classify(traindir,testdir,resnet_b=False,resnet_dct_b=False,sift_b=False,resnet_g=False,resnet_dct_g=False,sift_g=False):
    score_resnet_b=0
    score_resnet_dct_b=0
    score_sift_b=0
    score_resnet_g = 0
    score_resnet_dct_g = 0
    score_sift_g = 0
    binary_label = [0] * 2000 + [1] * 8000
    acc = 0
    path=''
    label = [0] * 2000 + [1] * 2000 + [2] * 2000 + [3] * 2000 + [4] * 2000
    if resnet_b:
        path_b='./NIR_result/binary_classification/resnet/'+traindir+'/resnet_result_'+testdir+'_test.npy'
        score_resnet_b=get_score(path_b)
        path+='resnet+'

    if resnet_g:
        path_g = './NIR_result/gan_classification/resnet/' + traindir + '/resnet_result_' + testdir + '_test.npy'
        score_resnet_g = get_score(path_g)

    if resnet_dct_b:
        path_b = './NIR_result/binary_classification/resnet_dct/' + traindir + '/resnet_result_' + testdir + '_test.npy'
        score_resnet_dct_b = get_score(path_b)
        path+='resnet_dct+'
    if resnet_dct_g:
        path_g = './NIR_result/gan_classification/resnet_dct/' + traindir + '/resnet_result_' + testdir + '_test.npy'
        score_resnet_dct_g = get_score(path_g)
    if sift_b:
        path_b='./NIR_result/binary_classification/sift/'+traindir+'/sift_result_'+testdir+'_test.npy'
        score_sift_b=get_score(path_b)
        path+='sift+'
    if sift_g:
        path_g = './NIR_result/gan_classification/sift/' + traindir + '/sift_result_' + testdir + '_test.npy'
        score_sift_g = get_score(path_g)
    score_b= score_resnet_b+score_resnet_dct_b+score_sift_b
    binary_pred = np.argmax(score_b,
                            axis=1)

    for i in range(0, len(binary_pred)):
        if binary_pred[i] == binary_label[i]:
            acc += 1

    print('{}: Train with {},Test with {},Bin Acc is {}'.format( path[:-1],traindir, testdir,
                                                             acc / len(binary_pred)))
    if score_resnet_g or score_sift_g or score_resnet_dct_g:
        pred=binary_pred
        acc=0
        score_g=score_resnet_g+score_resnet_dct_g+score_sift_g
        for i in range(len(binary_pred)):
            if binary_pred[i]==1:
                pred[i] = np.argmax(score_g[i])+1

        for i in range(0, len(pred)):
            if pred[i] == label[i]:
                acc += 1
        print('{}: Train with {},Test with {},Acc is {}'.format( path[:-1],traindir, testdir,
                                                             acc / len(pred)))

if __name__ == '__main__':
    filedir1=['std_single','std_multi','rand_single','rand_multi','mix_single','mix_multi']
    a=[(True,False,False),(False,True,False),(False,False,True),(True,True,False),(True,False,True),(False,True,True),(True,True,True)]
    for traindir in filedir1:
        for b,c,d in a:
            if traindir == 'std_single':
                filedir = ['std_single', 'rand_single', 'mix_single']
            if traindir == 'std_multi':
                filedir = [ 'std_multi','rand_multi','mix_multi']
            if traindir == 'rand_single':
                filedir = ['rand_single', 'mix_single']
            if traindir == 'rand_multi':
                filedir = ['rand_multi', 'mix_multi']
            if traindir == 'mix_single':
                filedir = ['mix_single']
            if traindir == 'mix_multi':
                filedir = ['mix_multi']
            for testdir in filedir:
                gan_classify(traindir=traindir, testdir=testdir, resnet_g=b,sift_g=c,resnet_dct_g=d
                         )
                         
