from PIL import Image
import os
from tqdm import tqdm
import numpy as np

# # 遍历图像文件，从1到10000
# for i in tqdm(range(1, 10001)):
#     # 读取对应的mask图像
#     mask_path = os.path.join('cam_gray_binary', f'cam_binary_{i:05d}.png')
#     # mask = Image.open(mask_path)
#     mask = Image.open(mask_path).convert('1')
#     # print(type(mask))
#     width, height = mask.size
#     # print(f"The image has {width} columns and {height} rows.")

#     # 读取要修改的图像
#     image_path = os.path.join('samples/adv/eps=0.01/png', f'adv_{i:05d}.png')
#     image = Image.open(image_path).convert('RGB')
#     width, height = mask.size
#     # print(f"The adv image has {width} columns and {height} rows.")

#     # 遍历所有像素，如果对应的mask像素为白色则修改为白色
#     for x in range(0, 32):
#         for y in range(0, 32):
#             # print(x, y)
#             if mask.getpixel((x, y)) == 255:
#                 image.putpixel((x, y), (255, 255, 255))
#     # sys.exit()
#     # 保存修改后的图像，使用新的文件名
#     new_image_path = os.path.join('fgsm_remove', f'fgsm_removed_{i:05d}.png')
#     image.save(new_image_path)



# Load the adv samples
adv_samples = np.load('./samples/adv/eps=0.01/adv_samples.npy')

# Create an array to store the modified images
modified_images = np.zeros_like(adv_samples)

# Iterate over each image/mask pair
for i in tqdm(range(1, 10001)):
    # Load the binary mask image
    mask_path = os.path.join('cam_gray_binary', f'cam_binary_{i:05d}.png')
    mask = Image.open(mask_path).convert('1')

    # Create a copy of the adv sample
    image = adv_samples[i - 1].copy()

    # Iterate over each pixel in the mask and modify the corresponding pixel in the image
    for x in range(0, 32):
        for y in range(0, 32):
            if mask.getpixel((x, y)) == 255:
                image[:, x, y] = np.array([1., 1., 1.])

    # Add the modified image to the array
    modified_images[i - 1] = image

# Save the modified images as a single npy file
new_image_path = os.path.join('fgsm_remove', 'remove.npy')
np.save(new_image_path, modified_images)