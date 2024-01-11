import os
import torch
import torchvision
import torchvision.transforms as transforms
from train import VGGTest
from argument import parser
import numpy as np
from tqdm import tqdm
import argparse

args = parser()
data_dir = args.data_path

# 加载预训练的VGG16模型
vgg16 = VGGTest()

# 加载训练好的权重
ckpt = torch.load(os.path.join('ckpt', 'ckpt_fin.pkl'))['net']

vgg16.load_state_dict(ckpt)

# 将模型设置为评估模式
vgg16.eval()

# 定义分类标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 定义损失函数
criterion = torch.nn.CrossEntropyLoss()

# 定义变量来记录分类准确率和总损失
total_loss = 0
total_correct = 0

# 加载标签
label = np.load('./samples/label.npy')

if args.image == 'clean':
    # 对干净样本进行分类
    data_path = './samples/ori/clr_samples.npy'

elif args.image == 'attack':
    # 对FGSM样本进行分类
    data_path = './samples/adv/eps=0.05/adv_samples.npy'

elif args.image == 'rebuild':
    # 对重建后的FGSM样本进行分类
    data_path = './fgsm_rebuild/rebuild.npy'

else:
    raise ValueError('Invalid image type. Please choose from clean, attack or rebuild.')

data = np.load(data_path)
print(data.shape)
print(data.dtype)

# 对每个样本进行分类
for i in tqdm(range(len(data))):

    # 转换图像和标签为 PyTorch 张量
    tensor_data = torch.from_numpy(data[i])
    tensor_label = torch.from_numpy(np.array(label[i]))

    # 将转换后的 Tensor 作为输入传递给模型进行分类
    with torch.no_grad():
        output = vgg16(tensor_data.unsqueeze(0))
        loss = criterion(output, tensor_label.unsqueeze(0))
        total_loss += loss.item()
        predicted = torch.argmax(output).item()
        total_correct += int(predicted == tensor_label.item())

# 计算分类准确率和平均损失
accuracy = 100 * total_correct / len(data)
average_loss = total_loss / len(data)

print('分类准确率：%.2f%%' % accuracy)
print('平均损失：%.3f' % average_loss)
