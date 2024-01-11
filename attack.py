#python 3.8

import os
import cv2
import numpy as np

import torch
# import tqdm
from tqdm import tqdm

from torchattacks import FGSM

from argument import parser
from train import testloader,VGGTest

##############################################
args = parser()

#ndarray转img
def ndarray2img(in_img):
    img = in_img[0].transpose(1, 2, 0)
    img = np.asarray(img * 255, dtype = int)

    return img

def main():

    model = torch.load(os.path.join('ckpt', 'ckpt_fin.pkl'))
    vggtest = VGGTest()

    vggtest.load_state_dict(model['net'])

    vggtest = vggtest.cuda()

    eps = args.epsilon
    atk = FGSM(vggtest, eps)

    k = 1
    out_init = np.ones((2,546,3)) * 255

    clr_path = args.oris_root
    adv_path = os.path.join(args.advs_root,'eps={:}'.format(eps))

    for images, labels in tqdm(testloader):

        # 干净图像的numpy格式
        images_clr_ndarray = images.cpu().numpy()

        # 拼接干净样本的numpy
        if k == 1:
            images_clr_ndarray_sum = images_clr_ndarray
        else:
            images_clr_ndarray_sum = np.concatenate((images_clr_ndarray_sum,images_clr_ndarray),axis=0)

        # 画图 clr 并保存
        out_img0 = ndarray2img(images_clr_ndarray)
        cv2.imwrite(os.path.join(clr_path, 'clr_{:05d}.png'.format(k)), out_img0)

        # 生成攻击样本，numpy格式
        images_atk = atk(images, labels).cuda()
        images_atk_ndarray = images_atk.cpu().detach().numpy()

        # 拼接对抗样本的numpy
        if k == 1:
            images_atk_ndarray_sum = images_atk_ndarray
        else:
            images_atk_ndarray_sum = np.concatenate((images_atk_ndarray_sum,images_atk_ndarray),axis=0)

        # 画图 adv 并保存
        out_img1 = ndarray2img(images_atk_ndarray)
        cv2.imwrite(os.path.join(adv_path, 'adv_{:05d}.png'.format(k)), out_img1)
        
        # lables
        if k == 1 :
            lables_sum = labels
        else:
            lables_sum = np.concatenate((lables_sum,labels))

        # 保存clr和adv样本的numpy格式
        if k >= 10000:
            np.save(os.path.join(adv_path,'adv_samples.npy'),images_atk_ndarray_sum)
            np.save(os.path.join(clr_path,'clr_samples.npy'),images_clr_ndarray_sum)
            # (10000,3,32,32)
            # 保存label的numpy格式
            np.save(os.path.join('samples', 'label.npy'), lables_sum)

        k += 1
        out_img0 = out_init
        out_img1 = out_init

if __name__ == '__main__':
    main()