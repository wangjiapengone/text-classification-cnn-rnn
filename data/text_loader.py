# coding: utf-8


from collections import Counter
import os
import re
import numpy as np
import pandas as pd
import tensorflow.contrib.keras as kr


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')


def preprocess(text):
    '''
    对原始文本进行预处理

    :param text: 文本
    :return:
    '''
    # 如果是以”答：“开头，则去掉这两个字
    if text.startswith('答：'):
        text = text[2:]
    # 去掉网站链接
    text = re.sub(r'((http[s]?|ftp)://|www)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                  ' ', text)
    # 去掉数字内容
    text = re.sub('\d+', ' ', text)
    # 英文小写
    text = text.lower()
    # 去掉多余空白字符
    text = re.sub('\s+', ' ', text)
    # 去掉 '', ' '
    text = [char for char in text if char not in ('', ' ')]
    return text


def read_data_file(filename):
    """读取文件数据"""
    df = pd.read_excel(filename)
    contents = df['data'].apply(list).tolist()
    labels = df['label'].tolist()
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_data_file(train_dir)

    all_data = [word for sent in data_train for word in sent]
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [line.strip() for line in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category(problem):
    """读取分类标签"""
    data_dir = 'train_test_sets/{}'.format(problem)
    files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('xlsx')]
    categories = set()
    for file in files:
        categories.update(pd.read_excel(file)['label'].tolist())
    categories = sorted(list(categories))
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_data_file(filename)

    data_id, label_id = [], []
    for content, label in zip(contents, labels):
        data_id.append([word_to_id[x] for x in content if x in word_to_id])
        label_id.append(cat_to_id[label])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
