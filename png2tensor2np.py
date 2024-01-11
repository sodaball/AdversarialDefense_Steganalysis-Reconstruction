import numpy as np
from PIL import Image
import os
import torch
from tqdm import tqdm

# 将PNG格式的图像转换为形状为(3, 32, 32)的PyTorch张量
def png_to_tensor(png_path):
    img = Image.open(png_path)
    img = img.convert('RGB') # 将图像转换为RGB格式
    img_array = np.array(img)
    # 将NumPy数组转换为PyTorch的tensor对象，并将像素值归一化到[0, 1]范围内
    tensor_img = torch.from_numpy(img_array).float() / 255.0
    tensor_img = tensor_img.permute(2, 0, 1)  # 将维度顺序由(3, 32, 32)改为(32, 32, 3)
    return tensor_img

# 将目录中的所有PNG格式的图像转换为形状为(10000, 3, 32, 32)的PyTorch张量
def convert_images_to_tensor(image_dir):
    np_images = []
    filenames = ["fgsm_rebuild_{:05d}.png".format(i) for i in range(1, 10001)] # 生成文件名列表
    for filename in tqdm(filenames):
        png_path = os.path.join(image_dir, filename)
        np_img = png_to_tensor(png_path)
        np_images.append(np_img)
    np_images = torch.stack(np_images) # 将列表中的张量堆叠为一个张量
    return np_images

# 将包含10000张PNG格式的图像的目录转换为形状为(10000, 3, 32, 32)的PyTorch张量
tensor_images = convert_images_to_tensor('./fgsm_rebuild')
# 并保存为.npy文件
numpy_images = tensor_images.numpy()
np.save('./fgsm_rebuild/rebuild.npy', numpy_images)