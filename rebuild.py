import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import os

# 遍历图像文件，从1到10000
for i in tqdm(range(1, 10001)):
    # 读取对应的mask图像
    pos_path = os.path.join('cam_gray_binary', f'cam_binary_{i:05d}.png')
    pos = Image.open(pos_path).convert('1')

    # 获取缺失像素点位置列表
    position = []
    for x in range(0, 32):
        for y in range(0, 32):
            if pos.getpixel((x, y)) == 255:
                position.append((y, x))  # OpenCV使用的坐标顺序是(x,y)

    # 转换掩膜图像为灰度图像
    mask = np.array(pos.convert('L'))

    # 读取要修改的图像
    img_path = os.path.join('fgsm_remove', f'fgsm_removed_{i:05d}.png')
    img = cv2.imread(img_path)

    # 检查像素点是否为NaN
    is_nan = np.isnan(img)

    # 选择插值方法（基于邻域像素的插值）
    method = cv2.INPAINT_NS

    # 进行插值
    for coord in position:
        # 使用OpenCV的函数进行插值
        inpaint_img = cv2.inpaint(inpaint_img, mask.astype(np.uint8), 3, method)

    # 保存修改后的图像，使用新的文件名
    new_image_path = os.path.join('fgsm_rebuild', f'fgsm_rebuild_{i:05d}.png')
    # 保存结果
    cv2.imwrite(new_image_path, inpaint_img)


# # Load the npy file containing the modified images with removed pixels
# remove_path = './fgsm_remove/remove.npy'
# remove_images = np.load(remove_path)

# # Create an empty numpy array to store the rebuilt images
# rebuild_images = np.empty((10000, 3, 32, 32), dtype=np.float32)

# # Iterate over each image in the npy file
# for i in tqdm(range(1, 10001)):
#     # Load the corresponding binary mask image
#     mask_path = os.path.join('cam_gray_binary', f'cam_binary_{i:05d}.png')
#     mask = Image.open(mask_path).convert('1')

#     # Get a list of positions for the missing pixels
#     position = []
#     for x in range(0, 32):
#         for y in range(0, 32):
#             if mask.getpixel((x, y)) == 255:
#                 position.append((y, x))  # OpenCV uses (x,y) coordinate order

#     # Convert the mask image to a grayscale numpy array
#     gray_mask = np.array(mask.convert('L'))

#     # Load the corresponding modified image with removed pixels
#     img = remove_images[i - 1]
#     # print(img.shape)

#     # Check for NaN pixels in the image
#     is_nan = np.isnan(img)

#     # Select the interpolation method (cubic spline-based interpolation)
#     method = cv2.INPAINT_CUBIC

#     # Perform the interpolation
#     inpaint_img = (img.transpose((1, 2, 0)) * 255).astype(np.uint8)

#     for coord in position:
#         # Use OpenCV's function to perform the inpainting
#         inpaint_img = cv2.inpaint(inpaint_img, gray_mask.astype(np.uint8), 3, method)

#     # Convert the interpolated image back to the original shape
#     inpaint_img = (inpaint_img.astype(np.float32) / 255).transpose((2, 0, 1))

#     # Save the rebuilt image in the array
#     rebuild_images[i - 1] = inpaint_img

# # Save the rebuilt images as a npy file
# rebuild_path = './fgsm_rebuild/rebuild.npy'
# np.save(rebuild_path, rebuild_images)