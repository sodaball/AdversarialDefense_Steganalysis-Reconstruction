from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import cv2
import numpy as np
import os
from train import VGGTest,testset
from argument import parser

if __name__ == '__main__':
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False)
    args =parser()
    # 1.加载模型
    model = VGGTest()
    ckpt = torch.load(os.path.join('ckpt', 'ckpt_fin.pkl'))['net']
    model.load_state_dict(ckpt)

    #选择目标层
    target_layer = [model.pool1,model.pool2,model.pool3,model.pool4]

    cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True)

    # ori_{:05d}.png: 原始测试图像的可视化
    # cam_gray_{:05d}.png: 使用 Grad-CAM 生成的灰度类激活图像
    # cam_binary_{:05d}.png: 将灰度类激活图像二值化得到的二值图像，用于更好地可视化类激活区域
    # cam_jet_{:05d}.png: 将灰度类激活图像应用颜色映射后得到的图像
    # stk_{:05d}.png: 将原始测试图像和彩色类激活图像叠加在一起得到的 CAM 可视化图像

    k = 0
    for image,lable in tqdm(testloader):
        k += 1
        target_category = None
        rgb_img = image.cpu().numpy()[0].transpose(1, 2, 0)
        rgb_img_c = np.uint8(255 * rgb_img )
        grayscale_cam = cam(input_tensor=image,targets=target_category)
        grayscale_cam = grayscale_cam[0]
        grayscale_cam_c = np.uint8(255 * grayscale_cam )
        binaryscale_cam = np.asarray([ [0 if k <= 224 else 255 for k in i] for i in grayscale_cam_c])
        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam ), cv2.COLORMAP_JET)
        visualization = show_cam_on_image(rgb_img, grayscale_cam,image_weight=0.7)
        cv2.imwrite('./cam_gray_binary/ori_{:05d}.png'.format(k), rgb_img_c)
        cv2.imwrite('./cam_gray_binary/cam_gray_{:05d}.png'.format(k), grayscale_cam_c)
        cv2.imwrite('./cam_gray_binary/cam_binary_{:05d}.png'.format(k), binaryscale_cam)
        cv2.imwrite('./cam_gray_binary/cam_jet_{:05d}.png'.format(k), heatmap)
        cv2.imwrite('./cam_gray_binary/stk_{:05d}.png'.format(k), visualization)