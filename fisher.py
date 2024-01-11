import numpy as np
import os
import cv2

from argument import parser
import argparse

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pickle

args = parser()
eps = args.epsilon

path_save_adv = os.path.join(args.advf_root, 'eps={:}'.format(eps))
path_save_clr = args.orif_root

def fisher_lda(X, y):
    # 计算每个类别的均值
    class_means = np.array([np.mean(X[y == i], axis=0) for i in np.unique(y)])
    # 计算总体均值
    overall_mean = np.mean(X, axis=0)
    # 计算类内散度矩阵
    Sw = np.sum([(X[y == i] - class_means[i]).T.dot(X[y == i] - class_means[i]) for i in np.unique(y)], axis=0)
    # 计算类间散度矩阵
    Sb = np.sum([len(X[y == i]) * (class_means[i] - overall_mean).reshape(-1, 1).dot((class_means[i] - overall_mean).reshape(1, -1)) for i in np.unique(y)], axis=0)
    # 计算广义特征值和特征向量
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    # 根据广义特征值对特征向量进行排序
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, sorted_indices]
    # 返回投影向量
    return eigvecs[:, 0]


class FisherClassifier:
    def __init__(self):
        self.projection_vector = None

    def train(self, X_train, y_train):
        # 计算投影向量
        self.projection_vector = fisher_lda(X_train, y_train)

    def predict(self, X):
        # 将特征投影到一维空间
        X_projected = X.dot(self.projection_vector)
        # 使用投影特征进行二分类
        return np.where(X_projected > np.median(X_projected), 1, 0)

    def get_projection_vector(self):
        return self.projection_vector

def load_data(data_dir):
    # 读取干净样本和对抗样本特征
    clean_features = np.load(os.path.join(path_save_clr, 'clr_f.npy'))
    adv_features = np.load(os.path.join(path_save_adv, 'adv_f.npy'))
    # 将干净样本和对抗样本特征组合成一个矩阵
    X = np.concatenate((clean_features, adv_features))
    # 创建标签向量，干净样本标签为0，对抗样本标签为1
    y = np.concatenate((np.zeros(clean_features.shape[0], dtype=int), np.ones(adv_features.shape[0], dtype=int)))
    # 打乱数据的顺序
    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]
    return X, y

def visualize(X, y, y_pred):
    # 将特征矩阵降到二维空间
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # 创建子图
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # 将干净样本和对抗样本分别绘制到图上
    axs[0].scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], c='blue', label='Clean')
    axs[0].scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], c='red', label='Adversarial')
    axs[0].legend()
    axs[0].set_title('Actual')

    # 将分类结果用不同颜色的点绘制到图上
    axs[1].scatter(X_2d[y_pred == 0, 0], X_2d[y_pred == 0, 1], c='green', marker='s', label='Clean (Predicted)')
    axs[1].scatter(X_2d[y_pred == 1, 0], X_2d[y_pred == 1, 1], c='orange', marker='s', label='Adversarial (Predicted)')
    axs[1].legend()
    axs[1].set_title('Predicted')

    # 显示图像
    plt.show()


def main():
    data_dir = "./features"
    X, y = load_data(data_dir)
    # 将特征矩阵归一化
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    # 将数据分为训练集和测试集
    train_size = int(X.shape[0] * 0.8)

    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # 训练Fisher分类器
    clf = FisherClassifier()
    clf.train(X_train, y_train)
    # 保存分类器到磁盘上
    filename = './fisher_model/trained_classifier.pkl'
    pickle.dump(clf, open(filename, 'wb'))
    # 在测试集上测试分类器性能
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print("Accuracy:", accuracy)

    # 可视化分类结果
    visualize(X_test, y_test, y_pred)

if __name__ == '__main__':
    main()