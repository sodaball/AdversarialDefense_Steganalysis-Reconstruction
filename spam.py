import os
import numpy as np
from argument import parser
from tqdm import tqdm
args = parser()
eps = args.epsilon

path_load_adv = os.path.join(args.advs_root, 'eps={:}'.format(eps), 'adv_samples.npy')
path_load_clr = os.path.join(args.oris_root, 'clr_samples.npy')

path_save_adv = os.path.join(args.advf_root, 'eps={:}'.format(eps))
path_save_clr = args.orif_root

def GetM3(L,C,R,T):
    # marginalization into borders
    L = np.clip(L, -T, T).flatten('F')
    C = np.clip(C, -T, T).flatten('F')
    R = np.clip(R, -T, T).flatten('F')

    # get cooccurences [-T...T]
    M = np.zeros((2*T+1,2*T+1,2*T+1))
    for i in range(-T, T+1, 1):
        C2 = C[L==i];
        R2 = R[L==i];
        for j in range(-T, T+1, 1):
            R3 = R2[C2==j];
            for k in range(-T, T+1, 1):
                M[i+T,j+T,k+T] = np.sum(R3==k)

    # normalization
    M = M.flatten('F')
    M /= np.max(M)
    return M

def spam_extract_2(X, T):
    # horizontal left-right
    X = np.concatenate((X[:,:,0], X[:,:,1], X[:,:,2]), axis=1)
    D = X[:,:-1] - X[:,1:]
    L = D[:,2:]
    C = D[:,1:-1]
    R = D[:,:-2]
    Mh1 = GetM3(L.copy(),C.copy(),R.copy(),T)

    # horizontal right-left
    D = -D;
    L = D[:,:-2]
    C = D[:,1:-1]
    R = D[:,2:]
    Mh2 = GetM3(L.copy(),C.copy(),R.copy(),T)

    # vertical bottom top
    D = X[:-1,:] - X[1:,:]
    L = D[2:,:]
    C = D[1:-1,:]
    R = D[:-2,:]
    Mv1 = GetM3(L.copy(),C.copy(),R.copy(),T)

    # vertical top bottom
    D = -D
    L = D[:-2,:]
    C = D[1:-1,:]
    R = D[2:,:]
    Mv2 = GetM3(L.copy(),C.copy(),R.copy(),T)

    # diagonal left-right
    D = X[:-1,:-1] - X[1:,1:]
    L = D[2:,2:]
    C = D[1:-1,1:-1]
    R = D[:-2,:-2]
    Md1 = GetM3(L.copy(),C.copy(),R.copy(),T)

    # diagonal right-left
    D = -D
    L = D[:-2,:-2]
    C = D[1:-1,1:-1]
    R = D[2:,2:]
    Md2 = GetM3(L.copy(),C.copy(),R.copy(),T)

    # minor diagonal left-right
    D = X[1:,:-1] - X[:-1,1:]
    L = D[:-2,2:]
    C = D[1:-1,1:-1]
    R = D[2:,:-2]
    Mm1 = GetM3(L.copy(),C.copy(),R.copy(),T)

    # minor diagonal right-left
    D = -D
    L = D[2:,:-2]
    C = D[1:-1,1:-1]
    R = D[:-2,2:]
    Mm2 = GetM3(L.copy(),C.copy(),R.copy(),T);

    F1 = (Mh1+Mh2+Mv1+Mv2)/4;
    F2 = (Md1+Md2+Mm1+Mm2)/4;
    F = np.concatenate((F1, F2), axis=0)
    return F

def SPAM(path):
    data = np.load(path)
    data = data.transpose(0,2,3,1) * 255
    feature_sum = np.zeros((1,686),dtype=float)
    feature = []
    for i in tqdm(range(data.shape[0])):
        feature.append(spam_extract_2(data[i], 3))
        feature = np.array(feature)
        feature.resize(1,686)
        if i == 0:
            feature_sum = feature
        else:
            feature_sum = np.concatenate((feature_sum, feature), axis=0)
        feature = []
    return feature_sum

def gen_feature(data_path, save_path):
    output = SPAM(data_path)
    output = np.array(output)
    np.save(save_path, output)

if __name__ == '__main__':
    # loadData = np.load('D:\Python_code\OVERALL/features/adv\eps=0.01/adv_f.npy')
    # print(loadData.shape)
    gen_feature(path_load_adv, os.path.join(path_save_adv, 'adv_f.npy'))
    gen_feature(path_load_clr, os.path.join(path_save_clr, 'clr_f.npy'))
    print('fin')