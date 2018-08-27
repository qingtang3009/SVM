# -*- coding: utf-8 -*-

import re
import jieba
import pandas
import numpy
import random
from snownlp import SnowNLP
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from collections import Counter
from matplotlib import pyplot


def preprocessing(single_comment):
    """
    这是一个预处理过程，包括分词，去停词，去除数字，去除特殊符号
    :param single_comment: 一条单独的文档（注意：不是整篇大文档，类似于分析购物评论中的一条评论）
    :return: [['单词', '单词', '单词',...], ['单词', '单词', '单词',...], ['', '', '',...],......]
    """
    jieba.load_userdict('D:\Pycharm\PycharmProjects\Class/jieba_dict/dict.txt')
    jieba.load_userdict('D:\Pycharm\PycharmProjects\Class/jieba_dict/coal_dict.txt')
    jieba.load_userdict('D:\Pycharm\PycharmProjects\Class/jieba_dict/user_dictionary.txt')
    comment0 = re.sub('\u3000', '', single_comment)   # 去掉一些字符，例如\u3000
    comment1 = re.sub(r'&[a-z]*', '', comment0)
    comment2 = re.sub(r'\ufffd', '', comment1)
    comment3 = re.sub('\u3000', '', comment2)
    comment4 = re.sub(r'\d+MM|\d+mm|\d+CM|\d+cm|\d+V|\d+v|\d+A|\d+m|\d+M|\d+w|\d+W', 'param', comment3)
    comment5 = re.sub(r'\d+\.\d+|\d+', 'num', comment4)
    comment6 = SnowNLP(comment5).han
    comment7 = re.sub(r'博世|博士|Bosch|BOSCH|bosch', '博世', comment6)
    comment8 = re.sub(r'小威|WORX|威克士|worx|wx|WX', '威克士', comment7)
    comment2words = jieba.__lcut(comment8)
    stop_words = open('./stop_words.txt', 'r').readlines()
    for i in range(comment2words.__len__())[::-1]:
        if comment2words[i] in stop_words:  # 去除停用词
            comment2words.pop(i)
        elif comment2words[i].isdigit():
            comment2words.pop(i)
    return comment2words


def creat_words_count_dictionary(comments_words_list):
    """
    创建一个计数器，统计每个此在整篇大文档中出现的次数
    :param comments_words_list: 来自于上个函数preprocessing的输出，[['单词', '单词', '单词',...], ['单词', '单词', '单词',...], ['', '', '',...],......]
    :return: ['单词0':出现个数, '单词1':出现个数, '单词2':出现个数, '单词3':出现个数,......]
    """
    counter = Counter()
    for comment_words in comments_words_list:
        for word in comment_words:
            counter[word] += 1
    return counter


def creat_words_bag(words_counter, min_count):
    """
    创建一个词袋
    :param words_counter: 来自于上个函数creat_words_count_dictionary的输出，每个词以及其对应的出现次数（注意：这里的统计的出现次数是相对于整个大文档而言）
    :param min_count: 最小限制次数，低于min_count则剔除
    :return: ['单词0':0, '单词1':1, '单词2':2, '单词3':3,......]（注意：这里单词是有排序的，按出现次数从高到低）
    """
    words2id = {}
    index = 0
    for (key, values) in sorted(words_counter.items(), key=lambda d: d[1], reverse=True):
        if values >= min_count:
            words2id[key] = index
            index += 1
        else:
            break
    return words2id


def creat_numpy_for_sklearn(comments_words_list, words2id):
    """
    创建一个sklearn可以识别的numpy矩阵，矩阵中np[i,j]位置对应comments_words_list对应的第i个词，对应word2id词袋中的位置以及出现次数
    :param comments_words_list: 来自preprocessing的输出，类似于全部评论切分出来的单词列表
    :param words2id: ，来自creat_words_bag的输出，一个词袋
    :return: numpy矩阵
    """
    np = numpy.zeros((len(comments_words_list), len(words2id)))
    for i in range(len(comments_words_list)):
        for word in comments_words_list[i]:
            if word in words2id.keys():
                np[i, words2id[word]] += 1
    return np


def data_analisys(url):
    """
    查看数据属性
    :param url:输入数据url
    :return: 每类数据， 标签汇总
    """
    data1 = pandas.read_csv(url, encoding='gbk')
    all_comment = []
    all_label = []
    for i in range(len(data1)):
        if data1['confidence'][i] > 0.3:
            all_comment.append(data1['附加评论'][i])
            all_label.append(data1['sentiment'][i])
    label_count = Counter()
    for l in all_label:
        label_count[l] += 1
    print(list(label_count.most_common()))

    len_0 = []
    len_1 = []
    len_2 = []
    comment_0 = []
    comment_1 = []
    comment_2 = []
    tag_0 = []
    tag_1 = []
    tag_2 = []
    for i in range(len(all_comment)):
        if all_label[i] == 0:
            len_0.append(len(all_comment[i]))
            comment_0.append(all_comment[i])
            tag_0.append('0')
        elif all_label[i] == 1:
            len_1.append(len(all_comment[i]))
            comment_1.append(all_comment[i])
            tag_1.append('1')
        else:
            len_2.append(len(all_comment[i]))
            comment_2.append(all_comment[i])
            tag_2.append('2')
    print(len(comment_0), len(comment_1), len(comment_2))
    # 不同类别的句子的长度分布
    labels = ['0', '1', '2']
    bins = [0, 5, 10, 15, 20, 25]
    pyplot.hist(x=[len_0, len_1, len_2], bins=bins, label=labels)
    pyplot.legend()
    pyplot.show()
    # 每类句子的平均长度
    print(sum(len_0) / len(len_0))
    print(sum(len_1) / len(len_1))
    print(sum(len_2) / len(len_2))
    return all_comment, comment_0, comment_1, comment_2, tag_0, tag_1, tag_2


def data_sample(all_comment, comment_0, comment_1, comment_2):
    """
    数据转化为numpy
    :param all_comment: 全部数据
    :param comment_0: 负面数据
    :param comment_1: 中性数据
    :param comment_2: 正向数据
    :return: 三个类别的numpy
    """
    random.seed(33)
    min_data = min(len(comment_0), len(comment_1), len(comment_2))
    data_0 = random.sample(comment_0, min_data)
    data_1 = random.sample(comment_1, min_data)
    data_2 = random.sample(comment_2, min_data * 2)
    comments_words = []
    for comment in all_comment:
        comments_words.append(preprocessing(comment))
    c_counter = creat_words_count_dictionary(comments_words)  # 这个是创建的词典，每个词对应其出现的次数
    c_bag = creat_words_bag(c_counter, min_count=3)   # 这个是词袋，去除小于3次的低频词
    comment_words_0 = []
    comment_words_1 = []
    comment_words_2 = []
    for comment in data_0:
        comment_words_0.append(preprocessing(comment))
    numpy_0 = creat_numpy_for_sklearn(comment_words_0, c_bag)
    for comment in data_1:
        comment_words_1.append(preprocessing(comment))
    numpy_1 = creat_numpy_for_sklearn(comment_words_1, c_bag)
    for comment in data_2:
        comment_words_2.append(preprocessing(comment))
    numpy_2 = creat_numpy_for_sklearn(comment_words_2, c_bag)

    return numpy_0, numpy_1, numpy_2


def data_split( Numpy_0, Numpy_1, Numpy_2, N, n_split=5):
    """
    这里作为划分两层SVM数据
    :param all_comment: 全部的评论
    :param comment_0: 负面评论
    :param comment_1: 中性评论
    :param comment_2: 正面评论
    :return: 返回各类的训练数据、测试数据以及其对应的标签
    """
    # random.seed(33)
    # min_data = min(len(comment_0), len(comment_1), len(comment_2))
    # data_0 = random.sample(comment_0, min_data)
    # data_1 = random.sample(comment_1, min_data)
    # data_2 = random.sample(comment_2, min_data * 2)
    # comments_words = []
    # for comment in all_comment:
    #     comments_words.append(preprocessing(comment))
    # c_counter = creat_words_count_dictionary(comments_words)  # 这个是创建的词典，每个词对应其出现的次数
    # c_bag = creat_words_bag(c_counter, min_count=3)   # 这个是词袋，去除小于3次的低频词
    # comment_words_0 = []
    # comment_words_1 = []
    # comment_words_2 = []
    # for comment in data_0:
    #     comment_words_0.append(preprocessing(comment))
    # numpy_0 = creat_numpy_for_sklearn(comment_words_0, c_bag)
    # for comment in data_1:
    #     comment_words_1.append(preprocessing(comment))
    # numpy_1 = creat_numpy_for_sklearn(comment_words_1, c_bag)
    # for comment in data_2:
    #     comment_words_2.append(preprocessing(comment))
    # numpy_2 = creat_numpy_for_sklearn(comment_words_2, c_bag)
    # 负面评论的训练数据以及测试数据
    step=int(Numpy_0.shape[0]/float(n_split))

    data_0_trian = numpy.vstack((Numpy_0[:N*step],Numpy_0[(N+1)*step:]))
    data_0_test = Numpy_0[N*step:(N+1)*step]
    # 中性评论的训练数据以及测试数据
    data_1_trian = numpy.vstack((Numpy_1[:N*step],Numpy_1[(N+1)*step:]))
    data_1_test = Numpy_1[N*step:(N+1)*step]
    # 正面评论的训练数据以及测试数据
    data_2_trian = numpy.vstack((Numpy_2[:2*N*step],Numpy_2[2*(N+1)*step:]))
    data_2_test = Numpy_2[N*step*2:2*(N+1)*step]
    # 全部评论的训练数据以及测试数据
    all_test = numpy.vstack((data_0_test, data_1_test, data_2_test))
    all_test_label = [0 for i in range(data_0_test.shape[0])] + [1 for i in range(data_1_test.shape[0])] + [2 for i in range(data_2_test.shape[0])]
    # 第一层SVM的训练数据以及其对应标签
    svm0_trian = numpy.vstack((data_0_trian, data_1_trian, data_2_trian))
    svm0_train_label = [0 for i in range(data_0_trian.shape[0] + data_1_trian.shape[0])] + [1 for i in range(data_2_trian.shape[0])]
    # 第一层SVM的测试数据以及其对应标签
    svm0_test = numpy.vstack((data_0_test, data_1_test, data_2_test))
    svm0_test_label = [0 for i in range(data_0_test.shape[0] + data_1_test.shape[0])] + [1 for i in range(data_2_test.shape[0])]
    # 第二层SVM的训练数据以及其对应标签
    svm1_trian = numpy.vstack((data_0_trian, data_1_trian))
    svm1_train_label = [0 for i in range(data_0_trian.shape[0])] + [1 for i in range(data_1_trian.shape[0])]
    # 第二层SVM的测试数据以及其对应标签
    svm1_test = numpy.vstack((data_0_test, data_1_test))
    svm1_test_label = [0 for i in range(data_0_test.shape[0])] + [1 for i in range(data_1_test.shape[0])]

    return all_test, all_test_label, svm0_trian, svm0_train_label, svm0_test, svm0_test_label, svm1_trian, svm1_train_label, svm1_test, svm1_test_label


def SVM(all_test, all_test_label, svm0_trian, svm0_train_label, svm0_test, svm0_test_label, svm1_trian, svm1_train_label, svm1_test, svm1_test_label):
    """
    训练预测
    :param all_test: 全部测试数据
    :param all_test_label: 全部测试数据对应的标签
    :param svm0_trian: 第一层SVM的训练数据
    :param svm0_train_label: 第一层SVM的训练数据对应的标签
    :param svm0_test: 第一层SVM的测试数据
    :param svm0_test_label: 第一层SVM的测试数据对应的标签
    :param svm1_trian: 第二层SVM的训练数据
    :param svm1_train_label: 第二层SVM的训练数据对应的标签
    :param svm1_test: 第二层SVM的测试数据
    :param svm1_test_label: 第二层SVM的测试数据对应的标签
    :return: 0
    """
    clf0 = LinearSVC()
    clf0.fit(svm0_trian, svm0_train_label)
    print("第一层支持向量机训练集得分：f%", clf0.score(svm0_trian, svm0_train_label))
    print("第一层支持向量机测试集得分：f%", clf0.score(all_test, svm0_test_label))
    clf1 = LinearSVC()
    clf1.fit(svm1_trian, svm1_train_label)
    print("第二层支持向量机训练集得分：f%", clf1.score(svm1_trian, svm1_train_label))
    print("第二层支持向量机测试集得分：f%", clf1.score(svm1_test, svm1_test_label))
    result_list = []
    for i in range(all_test.shape[0]):
        one_sample = numpy.atleast_2d(all_test[i, :])
        clf0_result = clf0.predict(one_sample)
        if clf0_result == 0:
            clf1_result = clf1.predict(one_sample)
            if clf1_result == 0:
                result_list.append(0)
            else:
                result_list.append(1)
        else:
            result_list.append(2)

    print("总得分：f%", accuracy_score(result_list, all_test_label))  # 打印正确率
    average_accuracy.append(accuracy_score(result_list, all_test_label))
    return 0


if __name__ == '__main__':
    url = './7_13_short.csv'
    all_comment, type_comment0, type_comment1, type_comment2, tag_0, tag_1, tag_2 = data_analisys(url)

    numpy0, numpy1, numpy2 = data_sample(all_comment=all_comment, comment_0=type_comment0, comment_1=type_comment1
                                         , comment_2=type_comment2)
    average_accuracy = []
    N_split = 5
    ss = [1, 2, 3, 4, 5]
    for i in range(len(ss)):
        print("开始第"+str(ss[i])+"轮")
        all_test, all_test_label, svm0_trian, svm0_train_label, svm0_test, svm0_test_label, svm1_trian, svm1_train_label, \
            svm1_test, svm1_test_label = data_split(Numpy_0=numpy0, Numpy_1=numpy1, Numpy_2=numpy2,
                                                    N=i, n_split=N_split)
        SVM(all_test, all_test_label, svm0_trian, svm0_train_label, svm0_test, svm0_test_label, svm1_trian,
            svm1_train_label, svm1_test, svm1_test_label)
    print("总的正确率：", sum(average_accuracy)/5)

