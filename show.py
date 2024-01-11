
import torch
from torchvision.transforms import ToPILImage, ToTensor, Resize
from train import VGGTest
import os
import model
from fisher import FisherClassifier
from torchattacks import FGSM
import pickle
import cv2
from spam import spam_extract_2
import gradio as gr
import torch
from torchvision.transforms import ToPILImage, ToTensor, Resize
from train import VGGTest
import os
import model
from fisher import FisherClassifier
from torchattacks import FGSM
import pickle
import cv2
from spam import spam_extract_2

F1 = 'triple_net_model/data_model/data_model.pth'   # 原始分类网络
F2 = 'triple_net_model/data_model_adv/data_model_adv.pth'
F3 = 'triple_net_model/data_model_F3/data_model_F3.pth'

# 类别标签
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
is_attack = ('干净样本', '对抗样本')

filename = './fisher_model/trained_classifier.pkl'
clf = FisherClassifier()
clf = pickle.load(open(filename, 'rb'))



# 定义分类函数
def classify_image(image, choose, eps):
    
    if choose == "生成FGSM攻击样本":
        # 将图像转换为Tensor
        tensor_image = ToTensor()(image).unsqueeze(0)

        # 加载FGSM攻击并设置攻击参数
        net = model.get_classification_net(net_choose=0, num_class=10, pretrained=True, path=F1)
        fgsm = FGSM(net, eps)
        # 使用VGG16进行图像分类
        with torch.no_grad():
            output = net(tensor_image)
            _, predicted = torch.max(output, 1)
        # 预测的类别
        predicted_class = classes[predicted.item()]
        
        # 定义目标标签
        target_label = torch.tensor([predicted.item()]).cuda()  # 示例中将目标标签设置为3

        # 对图像进行FGSM攻击
        adversarial_image = fgsm(tensor_image, target_label)

        # 使用VGG16对对抗样本进行图像分类
        with torch.no_grad():
            output = net(adversarial_image)
            _, predicted = torch.max(output, 1)
        # 预测的类别
        predicted_class = classes[predicted.item()]

        # 将Tensor转换为PIL图像
        pil_image = ToPILImage()(adversarial_image.squeeze(0))

        # 调整图像大小为256*256，便于展示
        resized_image = Resize((256, 256))(pil_image)


        return 'FGSM攻击参数：' + str(eps), '类别：' + predicted_class, resized_image
    if choose == "对图像进行分类":
        spam_feature = spam_extract_2(image, 3)
        # 判断是否是对抗样本，干净样本标签为0，对抗样本标签为1
        y_pred = clf.predict(spam_feature)

        # 选择原始分类网络还是三生网络
        if y_pred == 0:
            net = model.get_classification_net(net_choose=0, num_class=10, pretrained=True, path=F1)
        else:
            net = model.get_triple_net(net_choose=0, num_class=10, pretrained=True, path1=F1, path2=F2, path3=F3)
        sample_class = is_attack[y_pred]

        # 将模型设置为评估模式
        net.eval()

        # 将图像转换为Tensor
        tensor_image = ToTensor()(image).unsqueeze(0)

        # 使用VGG16进行图像分类
        with torch.no_grad():
            output = net(tensor_image)
            _, predicted = torch.max(output, 1)

        # 返回预测的类别
        predicted_class = classes[predicted.item()]

        # 将image转为PIL图像
        image = ToPILImage()(tensor_image.squeeze(0))

        # 调整图像大小为256*256，便于展示
        resized_image = Resize((256, 256))(image)

        # 返回预测的类别
        return '检测性防御结果：' + sample_class, '预测类别: ' + predicted_class, resized_image

# 设置输入和输出界面
inputs = [
    gr.Image(label="输入需要进行预测的图像"),
    gr.Radio(["生成FGSM攻击样本", "对图像进行分类"], label="选择软件功能"),
    gr.Slider(minimum=0.0, maximum=0.1, step=0.01, default=0.0, label="eps")
]

outputs = [
    gr.Label(label=""),
    gr.Label(label=""),
    gr.Image(label="样本可视化", type='pil')
]

# 创建界面
interface = gr.Interface(fn=classify_image, inputs=inputs, outputs=outputs, capture_session=True)

# 运行界面
interface.launch()

F1 = 'triple_net_model/data_model/data_model.pth'   # 原始分类网络
F2 = 'triple_net_model/data_model_adv/data_model_adv.pth'
F3 = 'triple_net_model/data_model_F3/data_model_F3.pth'

# 类别标签
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
is_attack = ('干净样本', '对抗样本')

filename = './fisher_model/trained_classifier.pkl'
clf = FisherClassifier()
clf = pickle.load(open(filename, 'rb'))



# 定义分类函数
def classify_image(image, choose, eps):
    
    if choose == "生成FGSM攻击样本":
        # 将图像转换为Tensor
        tensor_image = ToTensor()(image).unsqueeze(0)

        # 加载FGSM攻击并设置攻击参数
        net = model.get_classification_net(net_choose=0, num_class=10, pretrained=True, path=F1)
        fgsm = FGSM(net, eps)
        # 使用VGG16进行图像分类
        with torch.no_grad():
            output = net(tensor_image)
            _, predicted = torch.max(output, 1)
        # 预测的类别
        predicted_class = classes[predicted.item()]
        
        # 定义目标标签
        target_label = torch.tensor([predicted.item()]).cuda()  # 示例中将目标标签设置为3

        # 对图像进行FGSM攻击
        adversarial_image = fgsm(tensor_image, target_label)

        # 将Tensor转换为PIL图像
        pil_image = ToPILImage()(adversarial_image.squeeze(0))

        # 调整图像大小为32x32
        resized_image = Resize((32, 32))(pil_image)

        return 'FGSM攻击参数：' + str(eps), '类别：' + predicted_class, resized_image
    if choose == "对图像进行分类":
        spam_feature = spam_extract_2(image, 3)
        # 判断是否是对抗样本，干净样本标签为0，对抗样本标签为1
        y_pred = clf.predict(spam_feature)

        # 选择原始分类网络还是三生网络
        if y_pred == 0:
            net = model.get_classification_net(net_choose=0, num_class=10, pretrained=True, path=F1)
        else:
            net = model.get_triple_net(net_choose=0, num_class=10, pretrained=True, path1=F1, path2=F2, path3=F3)
        sample_class = is_attack[y_pred]

        # 将模型设置为评估模式
        net.eval()

        # 将图像转换为Tensor
        tensor_image = ToTensor()(image).unsqueeze(0)

        # 使用VGG16进行图像分类
        with torch.no_grad():
            output = net(tensor_image)
            _, predicted = torch.max(output, 1)

        # 返回预测的类别
        predicted_class = classes[predicted.item()]
        # 返回预测的类别
        return '检测性防御结果：' + sample_class, '预测类别: ' + predicted_class, image


if __name__ == "__main__":
    image = cv2.imread("samples\ori\clr_00001.png")
    choose = "生成FGSM攻击样本"
    choose = "对图像进行分类"
    eps = 0.01
    output1, output2, img = classify_image(image, choose, eps)
    print(output1)
    print(output2)
    # 显示返回的图片img
    if choose == "生成FGSM攻击样本":
        img.show()
    else:
        # ndarray转换为PIL图像
        img = ToPILImage()(img)
        # 调整图像大小为32x32
        img = Resize((32, 32))(img)
        img.show()

