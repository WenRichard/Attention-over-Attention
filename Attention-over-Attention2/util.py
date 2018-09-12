# -*- coding: utf-8 -*-
# @Time    : 2018/9/8 9:10
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : util.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf

# Softmax over axis
def softmax(target, axis, mask, epsilon=1e-12, name=None):
  with tf.op_scope([target], name, 'softmax'):
    max_axis = tf.reduce_max(target, axis, keepdims=True)
    # target-max_axis将e的指数缩小，减少计算量，并且用mask将padding0的部分不计入计算权重
    target_exp = tf.exp(target-max_axis) * mask
    normalize = tf.reduce_sum(target_exp, axis, keepdims=True)
    softmax = target_exp / (normalize + epsilon)
    return softmax

def orthogonal_initializer(scale = 1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    print('Warning -- You have opted to use the orthogonal_initializer function')
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.0, 1.0, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      print('you have initialized one orthogonal matrix.')
      return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer