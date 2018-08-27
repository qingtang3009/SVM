# -*- coding: utf-8 -*-

import re
import jieba
import numpy
import pandas
from snownlp import SnowNLP
from pprint import pprint
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


def get_data_and_tags(url):
    """
    读入源文件
    :param url: 需要处理的源数据
    :return:每条评论的列表形式，每条评论对应的标签列表形式
    """
    input_file = pandas.read_csv(url, encoding='gbk')
    comment_list = input_file.ix[:, '附加评论']
    tag_list = input_file.ix[:, 'sentiment']
    return comment_list, tag_list


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


def creat_words_bag(words_counter, min_count=5):
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


def split_data_copy(data_input, tag_input):
    """
    划分数据集
    :param data_input: 数据
    :param tag_input: 标签
    :return: 切分好的数据及标签
    """
    print(data_input.shape)
    print(tag_input.shape)
    train_data, test_data, train_tag, test_tag = train_test_split(data_input, tag_input, test_size=0.2, random_state=42)
    print(train_data.shape)
    print(train_tag.shape)
    return train_data, test_data, train_tag, test_tag


def split_data(X):
    """
    将数据集分为5份，交叉验证
    :param target_object: 需要切分的数据集，有可能是标签
    :return: 切分好的数据
    """
    kf = KFold(n_splits=2)
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = numpy.array(X)[train_index], numpy.array(X)[test_index]  # 这里必须改为numpy.array(X)
    return X_train, X_test


def support_vector_machine(train_data, test_data, train_tags, test_tags):
    """
    线性支持向量机
    :param train_data: 训练样本
    :param train_tags: 训练样本标签
    :param test_data: 测试样本
    :param test_tags: 测试样本标签
    :return:
    """
    clf = LinearSVC()
    # clf=svm.SVC(kernel='linear')
    clf.fit(train_data, train_tags)
    c_predict = clf.predict(test_data)
    pprint(c_predict)
    test_tags = list(test_tags)
    # d_count = 0
    testing_acc.append(clf.score(test_data, test_tags))
    training_acc.append(clf.score(train_data, train_tags))
    print('1', clf.score(test_data, test_tags))
    '''
    for i in range(int((len(c_predict)+len(test_tags))/2)):
        if c_predict[i] == test_tags[i]:
            d_count += 1
    return print(d_count/len(test_tags))
    '''
    return 0


def get_confident_data(url):
    """
    取出置信度大于0.4的数据，并且将正负中三类数据平衡
    :param url: 数据源
    :return: 评论列表，标签列表
    """
    input_file = pandas.read_csv(url, encoding='gbk')
    worth_0_confident = []
    worth_0_tags = []
    worth_1_confident = []
    worth_1_tags = []
    worth_2_confident = []
    worth_2_tags = []
    for i in range(0, len(input_file)):
        if input_file['confidence'][i] >= 0.3 and input_file['sentiment'][i] == 0:
            worth_0_confident.append(input_file['附加评论'][i])
            worth_0_tags.append(input_file['sentiment'][i])
        if input_file['confidence'][i] >= 0.3 and input_file['sentiment'][i] == 1:
            worth_1_confident.append(input_file['附加评论'][i])
            worth_1_tags.append(input_file['sentiment'][i])
        if input_file['confidence'][i] >= 0.3 and input_file['sentiment'][i] == 2:
            worth_2_confident.append(input_file['附加评论'][i])
            worth_2_tags.append(input_file['sentiment'][i])
    data_0 = worth_0_confident[0:len(worth_1_confident)]
    tag_0 = worth_0_tags[0:len(worth_1_tags)]
    data_1 = worth_1_confident
    tag_1 = worth_1_tags
    data_2 = worth_2_confident[0:len(worth_1_confident)]
    tag_2 = worth_2_tags[0:len(worth_1_tags)]
    all_data = []
    all_tag = []
    all_data.extend(data_0)
    all_tag.extend(tag_0)
    all_data.extend(data_1)
    all_tag.extend(tag_1)
    all_data.extend(data_2)
    all_tag.extend(tag_2)
    return all_data, all_tag


if __name__ == "__main__":
    all_comment_list, all_tag_list = get_confident_data(r'D:\PINGLUN\7_13_short.csv')
    comments_words = []
    for comment in all_comment_list:
        comments_words.append(preprocessing(comment))
    c_counter = creat_words_count_dictionary(comments_words)
    c_bag = creat_words_bag(c_counter, min_count=3)
    c_numpy = creat_numpy_for_sklearn(comments_words, c_bag)
    tag_list = numpy.array(all_tag_list)
    '''
    c_train, c_test = split_data(c_numpy)
    print(c_train, c_test)
    c_train_tags, c_test_tags = split_data(all_tag_list)
    print(c_train_tags, c_test_tags)
    '''

    kf = KFold(n_splits=5, shuffle=True, random_state=33)
    training_acc = []
    testing_acc = []
    for c_train_index, c_test_index in kf.split(X=c_numpy, y=tag_list):
        c_train, c_test = c_numpy[c_train_index], c_numpy[c_test_index]
        c_train_tags, c_test_tags = tag_list[c_train_index], tag_list[c_test_index]
        # c_train, c_test, c_train_tags, c_test_tags = split_data_copy(c_numpy, tag_list)
        support_vector_machine(c_train, c_test, c_train_tags, c_test_tags)

    print('training acc for k-fold', sum(training_acc) / len(training_acc))
    print('testing acc for k-fold', sum(testing_acc)/len(testing_acc))
