from torch import nn
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import numpy as np
import os
from argument import parser
savepath = r'features'
if not os.path.exists(savepath):
    os.mkdir(savepath)
"""
第一步：加载 CIFAR10 数据，大小是32*32*10
"""
args = parser()
batch_size = args.batch_size
ckpt_dir = args.ckpt_root
data_dir = args.data_path

transform = transforms.ToTensor()
trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=False)
testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False)
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class VGGTest(nn.Module):
    def __init__(self, pretrained=True, numClasses=10):
        super(VGGTest, self).__init__()

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
        if pretrained:
            pretrained_model = torchvision.models.vgg16(pretrained=pretrained)
            pretrained_params = pretrained_model.state_dict()
            keys = list(pretrained_params.keys())
            new_dict = {}
            for index, key in enumerate(self.state_dict().keys()):
                new_dict[key] = pretrained_params[keys[index]]
            self.load_state_dict(new_dict)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 1 * 1, out_features=256),
            # nn.Linear(in_features=512 * 7 * 7, out_features=256)
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=numClasses),
        )
    def draw_features1(self,width, height, x, savename):
        import time
        import matplotlib
        import cv2
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y
        tic = time.time()
        sub_output=x
        #plt.imshow(image, alpha=1)
        plt.axis('off')
        mask = np.zeros((width, height))
        b, c, h, w = np.shape(sub_output)
        sub_output = np.transpose(sub_output, [0, 2, 3, 1])[0]
        score = np.max(sigmoid(sub_output[..., 5:]), -1) * sigmoid(sub_output[..., 4])
        score = cv2.resize(score, (width, height))
        normed_score = (score * 255).astype('uint8')
        mask = np.maximum(mask, normed_score)
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(savename, dpi=200)
        print("Save to the " + savename)
        plt.cla()
        print("time:{}".format(time.time() - tic))
    def forward(self, x):   # output: 32 * 32 * 3
        x = self.relu1_1(self.conv1_1(x))  # output: 32 * 32 * 64
        x = self.relu1_2(self.conv1_2(x))  # output: 32 * 32 * 64
        x = self.pool1(x)  # output: 16 * 16 * 64
        #这个是随便加的一个位置,前面16 16
        #可以换到任意位置，主要直到特征层大小，这里是16 16
        # self.draw_features1(128, 128, x.cpu().detach().numpy(), "{}/f1_conv.png".format(savepath))
        # self.draw_features1(16, 16, x.cpu().detach().numpy(), "{}/f1_conv.png".format(savepath))
        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        x = self.pool2(x)
        # 可以换到任意位置，主要直到特征层大小，这里是8 8
        # self.draw_features1(8, 8, x.cpu().detach().numpy(), "{}/f2_conv.png".format(savepath))
        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        x = self.pool3(x)
        # 可以换到任意位置，主要直到特征层大小，这里是4 4 以此类推，每次进入网络，都会生成对应的，可以更具自己的意愿灵活切换
        # self.draw_features1(4, 4, x.cpu().detach().numpy(), "{}/f3_conv.png".format(savepath))
        x = self.relu4_1(self.conv4_1(x))
        x = self.relu4_2(self.conv4_2(x))
        x = self.relu4_3(self.conv4_3(x))
        x = self.pool4(x)
        # self.draw_features1(2, 2, x.cpu().detach().numpy(), "{}/f4_conv.png".format(savepath))
        x = self.relu5_1(self.conv5_1(x))
        x = self.relu5_2(self.conv5_2(x))
        x = self.relu5_3(self.conv5_3(x))
        x = self.pool5(x)

        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output

def vgg_train():
    epochs = args.num_epoches
    epoch = 0
    learning_rate = 1e-4

    net = VGGTest()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    if args.resume:
        # Load checkpoint
        print('loading checkpoint')
        resume_root = os.path.join(ckpt_dir,'ckpt.pkl')
        checkpoint = torch.load(resume_root)
        net.load_state_dict(checkpoint['net'])
        epoch = checkpoint['epoch']

    for epoch in range(epoch,epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data  # labels: [batch_size, 1]
            # print(i)
            print(labels)
            optimizer.zero_grad()

            outputs = net(inputs)  # outputs: [batch_size, 10]
            # print(outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 20 == 19:  # print loss every 20 mini batch
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                state = {
                    'net': net.state_dict(),
                    'epoch': epoch,
                }
                torch.save(state,os.path.join(ckpt_dir,'ckpt.pkl'))
                running_loss = 0.0

    torch.save(state, os.path.join(ckpt_dir, 'ckpt_fin.pkl'))
    print('Finished Training')

if __name__ == '__main__':
    vgg_train()