import numpy as np

def gan_classify(traindir,testdir,resnet_g=False,resnet_dct_g=False,sift_g=False,weight=None):

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

def binary_weight_score(score_resnet_b,score_resnet_dct_b,score_sift_b,weight):
    pred=(np.exp(weight[0])*score_resnet_b+np.exp(weight[1])*score_resnet_dct_b+np.exp(weight[2])*score_sift_b)/(np.exp(weight[0])+np.exp(weight[1])+np.exp(weight[2]))
    return pred

def gan_weight_score(score_resnet_g,score_sift_g,weight):
    pred=(np.exp(weight[0])*score_resnet_g+np.exp(weight[1])*score_sift_g)/(np.exp(weight[0])+np.exp(weight[1]))
    return pred

def get_score(filepath):
    score=np.load(filepath)

    return score
def classify(traindir,testdir,resnet_b=False,resnet_dct_b=False,sift_b=False,resnet_g=False,resnet_dct_g=False,sift_g=False,weight=None):
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

    if sift_b:
        path_b='./NIR_result/binary_classification/sift/'+traindir+'/sift_result_'+testdir+'_test.npy'
        score_sift_b=get_score(path_b)
        path+='sift+'
    if sift_g:
        path_g = './NIR_result/gan_classification/sift/' + traindir + '/sift_result_' + testdir + '_test.npy'
        score_sift_g = get_score(path_g)
    
    score_b= score_resnet_b+score_resnet_dct_b+score_sift_b
    if weight:
        score_b=binary_weight_score(score_resnet_b=score_resnet_b,score_resnet_dct_b=score_resnet_dct_b,score_sift_b=score_sift_b,weight=weight['binary'])

    binary_pred = np.argmax(score_b,
                            axis=1)

    for i in range(0, len(binary_pred)):
        if binary_pred[i] == binary_label[i]:
            acc += 1
    acc=0
    
    score_g=score_resnet_g+score_resnet_dct_g+score_sift_g
    if weight:
        score_g=gan_weight_score(score_resnet_g=score_resnet_g,score_sift_g=score_sift_g,weight=weight['gan'])
    for i in range(len(binary_pred)):
        if binary_pred[i]==1:

            binary_pred[i] = np.argmax(score_g[i])+1
    acc1=0
    for i in range(0, len(binary_pred)):
        if binary_pred[i] == label[i]:
            acc += 1
            if i>=2000:
                acc1+=1
    print('{}: Train with {},Test with {}, Acc is {}'.format( path[:-1],traindir, testdir,
                                                             acc / len(binary_pred)))

def prepare_weight(resnet_b,resnet_dct_b,sift_b,resnet_g,sift_g):
    return dict(binary=[resnet_b,resnet_dct_b,sift_b],gan=[resnet_g,sift_g])

if __name__ == '__main__':
    filedir1 = [ 'std_single','std_multi','rand_single','rand_multi','mix_single','mix_multi']
    a=[(True,True,True)]
    for traindir in filedir1:
        if traindir == 'std_single':
            filedir = ['std_single', 'rand_single', 'mix_single']
            weight=prepare_weight(resnet_b=1.0462,resnet_dct_b=1.0462,sift_b=0.9523,resnet_g=1.0796,sift_g=0.9204)
        if traindir == 'std_multi':
            filedir = [ 'std_multi','rand_multi','mix_multi']
            weight=prepare_weight(resnet_b=1.2808,resnet_dct_b=1.2808,sift_b=0.6968,resnet_g=1.0084,sift_g=0.9916)
        if traindir == 'rand_single':
            filedir = ['rand_single', 'mix_single']
            weight=prepare_weight(resnet_b=0.9973,resnet_dct_b=0.9973,sift_b=1.0027,resnet_g=0.9951,sift_g=1.0049)
        if traindir == 'rand_multi':
            filedir = ['rand_multi', 'mix_multi']
            weight=prepare_weight(resnet_b=1.0095,resnet_dct_b=1.0095,sift_b=0.9899,resnet_g=0.9925,sift_g=1.0075)
        if traindir == 'mix_single':
            filedir = ['mix_single']
            weight=prepare_weight(resnet_b=1.0748,resnet_dct_b=1.0748,sift_b=0.9220,resnet_g=1.0645,sift_g=0.9355)
        if traindir == 'mix_multi':
            filedir = ['mix_multi']
            weight=prepare_weight(resnet_b=1.0546,resnet_dct_b=1.0546,sift_b=0.9437,resnet_g=1.0953,sift_g=0.9047)
        for testdir in filedir:
            classify(traindir=traindir, testdir=testdir, resnet_b=True,sift_b=True,resnet_dct_b=True,resnet_g=True,sift_g=True,weight=weight
                         )
                         
