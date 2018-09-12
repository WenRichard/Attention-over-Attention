# -*- coding: utf-8 -*-
# @Time    : 2018/9/8 8:55
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : data_helper.py
# @Software: PyCharm

import os
import tensorflow as tf
import pickle
from collections import Counter

def counts():
    cache = 'counter.pickle'
    if os.path.exists(cache):
        with open(cache,'rb') as f:
            return pickle.load(f)

    directories = ['E:/NLP/Attention_data/data/cnn/cnn/questions/training/',
                   'E:/NLP/Attention_data/data/cnn/cnn/questions/validation/',
                   'E:/NLP/Attention_data/data/cnn/cnn/questions/test/']
    files = [directory + file_name for directory in directories for file_name in os.listdir(directory)]
    counter = Counter()
    for file_name in files:
        with open(file_name,'r',encoding='gb18030',errors='ignore') as f:
            lines = f.readlines()
            document = lines[2].split()
            query = lines[4].split()
            answer = lines[6].split()
            for token in document + query + answer:
                counter[token] +=1
    print(counter)
    with open(cache,'wb') as f:
        pickle.dump(counter,f)
    return counter


def tokenize(index,word):
    '''
    TFRecord就是对于输入数据做统一管理的格式.加上一些多线程的处理方式,使得在训练期间对于数据管理把控的效率和舒适度都好于暴力的方法.
    :param index:
    :param word:
    :return:
    '''
    directories = ['E:/NLP/Attention_data/data/cnn/cnn/questions/training/',
                   'E:/NLP/Attention_data/data/cnn/cnn/questions/validation/',
                   'E:/NLP/Attention_data/data/cnn/cnn/questions/test/']
    for directory in directories:
        out_name = directory.split('/')[-2] + '.tfrecords'
        writer = tf.python_io.TFRecordWriter(out_name)
        files = map(lambda file_name: directory + file_name,os.listdir(directory))
        for file_name in files:
            with open(file_name,'r',encoding='gb18030',errors='ignore') as f:
                lines = f.readlines()
                document = [index[token] for token in lines[2].split()]
                query = [index[token] for token in lines[4].split()]
                answer = [index[token] for token in lines[6].split()]
                example = tf.train.Example(
                    features = tf.train.Features(
                        feature = {
                            'document': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=document)),
                            'query':tf.train.Feature(
                                int64_list=tf.train.Int64List(value=query)),
                            'answer': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=answer))
                        }
                    )
                )
                seriaized = example.SerializeToString()
                writer.write(seriaized)

def main():
    counter = counts()
    print('num words',len(counter))
    word, _ = zip(*counter.most_common())
    index = {token:i for i,token in enumerate(word)}
    tokenize(index,word)
    print('Done')


if __name__ == '__main__':
    main()