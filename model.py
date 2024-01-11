from matplotlib import container
from pip import main
import torch
import torch.nn as nn
import torchvision


class VGG(nn.Module):  # 检验见model_0.py
    def __init__(self, num_class, pretrained=True):
        super(VGG, self).__init__()

        # 100% 还原特征提取层，也就是5层共13个卷积层
        # conv1 1/2
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv2 1/4
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv3 1/8
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv4 1/16
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv5 1/32
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # load pretrained params from torchvision.models.vgg16(pretrained=True)
        # 从原始的 models.vgg16(pretrained=True) 中预设值参数值。
        if pretrained:
            pretrained_model = torchvision.models.vgg16(pretrained=pretrained)  # 从预训练模型加载VGG16网络参数
            pretrained_params = pretrained_model.state_dict()
            keys = list(pretrained_params.keys())
            new_dict = {}
            for index, key in enumerate(self.state_dict().keys()):
                new_dict[key] = pretrained_params[keys[index]]
            self.load_state_dict(new_dict)

        # 但是至于后面的全连接层，根据实际场景，就得自行定义自己的FC层了。
        if num_class == 10:  # cifar10数据集大小为32*32
            self.classifier = nn.Sequential(  # 定义自己的分类层
                # 原始模型vgg16输入image大小是224 x 224
                # 我们测试的自己模仿写的模型输入image大小是32 x 32
                # 大小是小了 7 x 7倍
                nn.Linear(in_features=512 * 1 * 1, out_features=256),  # 自定义网络输入后的大小。
                # nn.Linear(in_features=512 * 7 * 7, out_features=256),  # 原始vgg16的大小是 512 * 7 * 7 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(in_features=256, out_features=256),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(in_features=256, out_features=num_class),
            )

        elif num_class == 3:  # FLIR数据集
            self.classifier = nn.Sequential(  # 定义自己的分类层
                # 原始模型vgg16输入image大小是224 x 224
                # 我们测试的自己模仿写的模型输入image大小是32 x 32
                # 大小是小了 7 x 7倍
                nn.Linear(in_features=512 * 7 * 7, out_features=256),  # 自定义网络输入后的大小。
                # nn.Linear(in_features=512 * 7 * 7, out_features=256),  # 原始vgg16的大小是 512 * 7 * 7 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(in_features=256, out_features=256),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(in_features=256, out_features=num_class),
            )

    def forward(self, x):   # output: 32 * 32 * 3
        x = self.relu1_1(self.conv1_1(x))  # output: 32 * 32 * 64
        x = self.relu1_2(self.conv1_2(x))  # output: 32 * 32 * 64
        x = self.pool1(x)  # output: 16 * 16 * 64

        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        x = self.pool2(x)

        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        x = self.pool3(x)

        x = self.relu4_1(self.conv4_1(x))
        x = self.relu4_2(self.conv4_2(x))
        x = self.relu4_3(self.conv4_3(x))
        x = self.pool4(x)

        x = self.relu5_1(self.conv5_1(x))
        x = self.relu5_2(self.conv5_2(x))
        x = self.relu5_3(self.conv5_3(x)) 
        x = self.pool5(x)

        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output


def get_classification_net(net_choose, num_class, pretrained=False, path=None):  # 检验见model_0.py
    """
    :param pretrained: 是否加载预训练参数，默认为不加载
    :param path: 加载预训练参数时，应给出参数保存地址
    :return: 得到最基本的分类器
    """
    if net_choose == 1:  # 加载resnet18 模型
        print('加载resnet18模型')
        resnet = torchvision.models.resnet18(pretrained=True)
        net = nn.Sequential(*(list(resnet.children())[:-1]))
        net.add_module('flat', nn.Flatten())
        net.add_module('classifer', nn.Linear(512, num_class, bias=True))
        # net = torchvision.models.resnet18(num_classes=num_class)
    elif net_choose == 0:
        print('加载vgg16模型')
        net = VGG(num_class=num_class, pretrained=True)  # 这里是在classification层前的部分添加imagenet的预训练
        # net = torchvision.models.vgg16(num_classes=num_class)
    elif net_choose == 2:
        print('加载ViT模型')
    if pretrained:  # 这里是加载自己训练的参数
        net.load_state_dict(torch.load(path))
    return net

# 在forward函数中，首先对三个子网络进行了评估，并使用torch.argmax函数计算出每个子网络输出的预测结果。
# 然后，使用torch.stack函数将这三个结果沿着第二个维度（即列）进行堆叠，得到一个形状为(batch_size, 3)的输出张量，其中batch_size为批量大小。
# 最后，使用vot函数进行投票抉择，得到最终的输出结果。
# vot函数接受一个(batch_size, 3)形状的张量作为输入，该张量的每一行表示一个样本的三个并行结果。
# 函数的作用是对每个样本的三个结果进行投票，得到一个形状为(batch_size, num_class)的独热编码张量作为输出结果。
class Triple(nn.Module):
    def __init__(self, net_choose, num_class):
        self.num_class = num_class
        super(Triple, self).__init__()
        self.block1 = get_classification_net(net_choose=net_choose, num_class=num_class)
        self.block2 = get_classification_net(net_choose=net_choose, num_class=num_class)
        self.block3 = get_classification_net(net_choose=0, num_class=num_class)

    def forward(self, in_put):
        self.block1.eval()
        self.block2.eval()
        self.block3.eval()
        out_block1 = torch.argmax(self.block1(in_put), dim=1)  # (batch_size, 3/10)
        out_block2 = torch.argmax(self.block2(in_put), dim=1)
        out_block3 = torch.argmax(self.block3(in_put), dim=1)
        out_ = torch.stack((out_block1, out_block2, out_block3), dim=1)
        # print(out_)  # (batch_size, 3)  # 每行为一个样本取得的三个并行结果)
        out = vot(out_, self.num_class)
        return out  # 这里输出的是(batch_size, num_class)的独热编码  # 方便契合准确度计算函数

def get_triple_net(net_choose, num_class, pretrained=False, path1=None, path2=None, path3=None):
    """
    :param pretrained: 是否加载预训练参数，默认为不加载
    :param path1: 孪生网络block1应该预加载原始网络参数
    :param path2: 孪生网络block2应该预加载对抗训练网络参数
    :return:获得孪生网络
    """
    net = Triple(net_choose=net_choose, num_class=num_class)
    if pretrained:
        net.block1.load_state_dict(torch.load(path1))
        net.block2.load_state_dict(torch.load(path2))
        net.block3.load_state_dict(torch.load(path3))
    return net

# vot 函数的输入为并行结果 in_put 和类别数 num_class，输出为投票结果 out_put
# 对于每个样本，函数首先创建一个投票箱 box，其中包含 num_class 个位置，初始值为0。
# 然后，它将该样本的三个并行结果存储在一个列表 vots 中，并遍历该列表中的每个元素 j。
# 对于每个 j，函数将投票箱中与 j 对应的位置加1，并更新投票箱中的最大值 vot_。
# 最后，函数将最大值所在的位置对应的输出置为1，其他位置置为0，并将该结果存储在输出张量 out_put 的相应位置。
# 最终，函数返回 out_put，形状为 (batch_size, num_class)。
def vot(in_put, num_class):  # 输入网络的并行结果，投票抉择
    b, _ = in_put.shape
    out_put = torch.zeros((b, num_class))
    for i in range(b):
        box = [0 for _ in range(num_class)]  # 初始化投票箱
        vots = list(in_put[i])  # vots长为3的列表
        for j in vots:
            box[j] += 1
            vot_ = torch.argmax(torch.tensor(box))
        out_put[i][vot_] = 1
    return out_put



















class Twin(nn.Module):
    def __init__(self, net_choose, num_class):
        super(Twin, self).__init__()
        self.block1 = get_classification_net(net_choose=net_choose, num_class=num_class)
        self.block2 = get_classification_net(net_choose=net_choose, num_class=num_class)

    def forward(self, in_put):
        out_block1 = self.block1(in_put)  # (batch_size, 3/10)
        out_block2 = self.block2(in_put)
        out_put1 = torch.softmax(out_block1, dim=1)
        out_put2 = torch.softmax(out_block2, dim=1)
        return torch.cat((out_put1, out_put2), dim=0)


def get_twin_net(net_choose, num_class, pretrained=False, path1=None, path2=None):
    """
    :param pretrained: 是否加载预训练参数，默认为不加载
    :param path1: 孪生网络block1应该预加载原始网络参数
    :param path2: 孪生网络block2应该预加载对抗训练网络参数
    :return:获得孪生网络
    """
    net = Twin(net_choose=net_choose, num_class=num_class)
    if pretrained:
        net.block1.load_state_dict(torch.load(path1))
        net.block2.load_state_dict(torch.load(path2))
    return net


# 关于集成增强决策的设想1
class Decision(nn.Module):
    def __init__(self, num_class):
        super(Decision, self).__init__()
        self.linear1 = nn.Linear(num_class * 2, num_class * 4, bias=True)
        self.active1 = nn.ReLU()
        self.linear2 = nn.Linear(num_class * 4 , num_class * 16, bias=True)
        self.active2 = nn.ReLU()
        self.linear3 = nn.Linear(num_class * 16, num_class, bias=True)

    def forward(self, in_put):
        out_put1 = self.active1(self.linear1(in_put))
        out_put2 = self.active2(self.linear2(out_put1))
        out_put = self.linear3(out_put2)
        return out_put


def get_decision_net(num_class, pretrain=False, path=None):
    """
    :param pretrain:
    :param path:
    :return: 获得继承增强决策网络
    """
    net = Decision(num_class=num_class)
    if pretrain:
        net.load_state_dict(torch.load(path))
    return net


# 孪生网络加继承增强网络整体决策
class Twin_Decision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.twin = get_twin_net()
        self.decision = get_decision_net()

    def forward(self, in_put):
        out_6 = self.twin(in_put)
        out_3 = self.decision(out_6)
        return out_3


def get_twin_decision_net(pretrain=False, path1=None, path2=None, path3=None):
    """
    :param pretrain:
    :param path1:
    :param path2:
    :param path3:
    :return: 获得总体预测网络
    """
    net = Twin_Decision()
    if pretrain:
        net.twin.block1.load_state_dict(torch.load(path1))
        net.twin.block2.load_state_dict(torch.load(path2))
        net.decision.load_state_dict(torch.load(path3))
    return net


# 关于集成增强决策的设想2
class Addition_Twin(nn.Module):
    def __init__(self, net_choose, num_class):
        super(Addition_Twin, self).__init__()
        self.block1 = get_classification_net(net_choose=net_choose, num_class=num_class)
        self.block2 = get_classification_net(net_choose=net_choose, num_class=num_class)

    def forward(self, in_put):  # (m-min)/(max-min)
        with torch.no_grad():
            out_block1 = self.block1(in_put)  # (batch_size, 3)
            out_block2 = self.block2(in_put)
            print('原始网络logit值')
            print(out_block1)
            print('对抗网络logit值')
            print(out_block2)
            max1, _ = torch.max(out_block1, dim=1, keepdim=True)
            min1, _ = torch.min(out_block1, dim=1, keepdim=True)
            out_block1 = (out_block1 - min1) / (max1 - min1)
            max2, _ = torch.max(out_block2, dim=1, keepdim=True)
            min2, _ = torch.min(out_block2, dim=1, keepdim=True)
            out_block2 = (out_block2 - min2) / (max2 - min2)
            print('原始网络logit值归一化')
            print(out_block1)
            print('对抗网络logit值归一化')
            print(out_block2)
            print('原始网络logit值归一化后softmax')
            print(torch.softmax(out_block1, dim=1))
            print('对抗网络logit值归一化后softmax')
            print(torch.softmax(out_block2, dim=1))
            out_put = 0.5 * out_block1 + 0.5 * out_block2
            # print(out_put)
            return torch.softmax(out_put, dim=1)

    def forward1(self, in_put):  # (m-min)/(max-min)再softmax
        with torch.no_grad():
            out_block1 = self.block1(in_put)  # (batch_size, 3)
            out_block2 = self.block2(in_put)
            print('原始网络logit值')
            print(out_block1)
            print('对抗网络logit值')
            print(out_block2)
            max1, _ = torch.max(out_block1, dim=1, keepdim=True)
            min1, _ = torch.min(out_block1, dim=1, keepdim=True)
            out_block1 = (out_block1 - min1) / (max1 - min1)
            max2, _ = torch.max(out_block2, dim=1, keepdim=True)
            min2, _ = torch.min(out_block2, dim=1, keepdim=True)
            out_block2 = (out_block2 - min2) / (max2 - min2)
            print('原始网络logit值归一化')
            print(out_block1)
            print('对抗网络logit值归一化')
            print(out_block2)
            print('原始网络logit值归一化后softmax')
            print(torch.softmax(out_block1, dim=1))
            print('对抗网络logit值归一化后softmax')
            print(torch.softmax(out_block2, dim=1))
            out_put = 0.3*torch.softmax(out_block1, dim=1) + 0.7*torch.softmax(out_block2, dim=1)
            # print(out_put)
            return torch.softmax(out_put, dim=1)

    def forward2(self, in_put):  # 欧式距离决策
        with torch.no_grad():
            out_block1 = self.block1(in_put)  # (batch_size, 3)
            out_block2 = self.block2(in_put)

            max1, _ = torch.max(out_block1, dim=1, keepdim=True)
            min1, _ = torch.min(out_block1, dim=1, keepdim=True)
            out_1 = (out_block1 - min1) / (max1 - min1)
            max2, _ = torch.max(out_block2, dim=1, keepdim=True)
            min2, _ = torch.min(out_block2, dim=1, keepdim=True)
            out_2 = (out_block2 - min2) / (max2 - min2)
            out_ = 0.3 * out_1 + 0.7 * out_2

            out_block1 = torch.softmax(out_block1, dim=1)
            out_block2 = torch.softmax(out_block2, dim=1)
            fro2 = torch.norm((out_block1-out_block2), dim=1)

            out_put = torch.zeros_like(out_block1, dtype=out_block1.dtype, device=out_block1.device)
            decision = fro2 < 1
            for i in range(len(out_block1)):
                if decision[i]:  # 小于1，则认为是干净样本
                    out_put[i] = out_block1[i]  # 直接将原始输出作为结果
                else:
                    out_put[i] = out_[i]
            # print(out_put)
            return out_put

    def forward3(self, in_put):  # (m-min)/(max-min)再softmax
        with torch.no_grad():
            out_block1 = self.block1(in_put)  # (batch_size, 3)
            out_block2 = self.block2(in_put)
            
            print('原始网络logit值')
            print(out_block1)
            print('对抗网络logit值')
            print(out_block2)

            print('原始网络分类结果')
            print(torch.argmax(out_block1, dim=1))
            print('对抗网络分类结果')
            print(torch.argmax(out_block2, dim=1))
            
            print('原始网络logit值softmax')
            print(torch.softmax(out_block1, dim=1))
            print('对抗网络logit值softmax')
            print(torch.softmax(out_block2, dim=1))

            max1, _ = torch.max(out_block1, dim=1, keepdim=True)
            min1, _ = torch.min(out_block1, dim=1, keepdim=True)
            out_block1 = (out_block1 - min1) / (max1 - min1)
            max2, _ = torch.max(out_block2, dim=1, keepdim=True)
            min2, _ = torch.min(out_block2, dim=1, keepdim=True)
            out_block2 = (out_block2 - min2) / (max2 - min2)
            print('原始网络logit值归一化')
            print(out_block1)
            print('对抗网络logit值归一化')
            print(out_block2)

            out_put = 0.3*torch.softmax(out_block1, dim=1) + 0.7*torch.softmax(out_block2, dim=1)
            # print(out_put)
            return torch.softmax(out_put, dim=1)

    def forward4(self, in_put):  # 除最大绝对值再相加
        # 正确率:0.720
        # 正确率:0.220
        out_block1 = self.block1(in_put)  # (batch_size, 3)
        out_block2 = self.block2(in_put)
        max1 = torch.max(torch.abs(out_block1))
        max2 = torch.max(torch.abs(out_block2))
        out_block1 = out_block1 / max1
        out_block2 = out_block2 / max2
        # print(out_block1, out_block2)
        out_put = out_block1 + out_block2
        # print(out_put)
        return torch.softmax(out_put, dim=1)


def get_addition_decision_net(net_choose, num_class, pretrained=False, path1=None, path2=None):
    """
    :param pretrained: 是否加载预训练参数，默认为不加载
    :param path1: 孪生网络block1应该预加载原始网络参数
    :param path2: 孪生网络block2应该预加载对抗训练网络参数
    :return:获得孪生网络
    """
    net = Addition_Twin(net_choose=net_choose, num_class=num_class)
    if pretrained:
        net.block1.load_state_dict(torch.load(path1))
        net.block2.load_state_dict(torch.load(path2))
    return net
    

if __name__ == '__main__':
    net = get_classification_net(net_choose=1, num_class=3)
    print(net)