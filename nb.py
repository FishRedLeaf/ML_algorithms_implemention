# coding:utf-8

# 参考https://blog.csdn.net/weixin_40504503/article/details/80475754
"""
朴素贝叶斯算法的python实现

思路：最大化后验概率P(Y=Ck|X=x)
y = f(x)
  = argmax(Ck) P(Y=Ck|X=x)  # P(Y=Ck|X=x)最大时Ck的值作为y
  = argmax(Ck) P(Y=Ck) * TT(1~num_samples) P(X_j=x_j | Y=Ck)  # 乘式第二项表示连乘(j=1~num_samples)

极大似然估计
P(Y=Ck) = sum(I(yi=Ck)) / num_samples
P(X_j=x_j | Y=Ck) = sum(I(x_j=a_jl, y_i=Ck) for i in range(1, 1+num_samples)) / 
                    sum(I(y_i=Ck) for i in range(1, 1+num_samples))
j=1~num_samples; l=1~Sj, Sj表示第j个特征取值集合的大小; k=1~K, K表示label取值集合的大小
"""


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


class NaiveBayes(object):

    def getTrainSet(self, trainset_size=140):
        self.data = load_iris()
        # 查看data所具有的属性或方法:['DESCR', 'data', 'feature_names', 'filename', 'target', 'target_names']
        # print(dir(data))
        X = self.data.data  # ndarray, dim=(150, 4)
        y = self.data.target  # ndarray, dim=(150,)
        trainData = X[:trainset_size]
        trainlabels = y[:trainset_size]
        testData = X[trainset_size:]
        testlabels = y[trainset_size:]
        return X, y, trainData, trainlabels, testData, testlabels

    def classify(self, trainData, trainlabels, features):

        # 最大化 后验概率的分子
        P = []  # P中存放不同label对应的后验概率的分子的值
        set_labels = list(set(trainlabels))
        for y in set_labels:
            y_index = [i for i, label in enumerate(trainlabels) if label == y]  # labels中出现y值的所有数值的下标索引
            prior = len(y_index) / float(len(trainlabels))
            tmp = 1  # 连乘项
            for j in range(len(features)):  # features[0] 在trainData[:,0]中出现的值的所有下标索引
                x_index = [i for i, feature in enumerate(trainData[:, j]) if feature == features[j]]
                xy_count = len(set(x_index) & set(y_index))
                tmp *= xy_count / float(len(y_index))
            P.append(prior * tmp)
        return set_labels[P.index(max(P))]

    def test(self, trainData, labels, testData, testlabels):

        correct = 0
        for i in range(len(testData)):
            # Outlook Temperature Humidity Wind
            features = testData[i]
            # 该特征应属于哪一类
            result = nb.classify(trainData, labels, features)
            print(features, '属于', result)
            if nb.classify(trainData, labels, testData[i]) == testlabels[i]:
                correct += 1
            print(correct / float(len(testlabels)))

if __name__ == '__main__':
    nb = NaiveBayes()
    # 训练数据,测试数据
    X, y, trainData, trainlabels, testData, testlabels = nb.getTrainSet()
    nb.test(trainData, trainlabels, testData, testlabels)  # 0.7
    # nb.test(X, y, testData, testlabels)  # 1.0
