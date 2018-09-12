# -*- coding: utf-8 -*-
# @Time    : 2018/9/8 11:31
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : model.py
# @Software: PyCharm

import os
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import sparse_ops
from util import softmax, orthogonal_initializer

flags = tf.app.flags
FLAGS = flags.FLAGS
# 原来是119662
flags.DEFINE_integer('vocab_size',119635,'Vocabulary size')
flags.DEFINE_integer('embedding_size',384,'Embedding dimension')
flags.DEFINE_integer('hidden_size',256,'Hidden units')
flags.DEFINE_integer('batch_size',32,'Batch size')
flags.DEFINE_integer('epochs',2,'Number of epochs to train/test')
flags.DEFINE_boolean('training',True,'Training or testing a model')
flags.DEFINE_string('name','lc_model','Model name(uesd for statistics and model path')
flags.DEFINE_float('dropout_keep_prob',0.9,'keep prob for embedding dropout')
flags.DEFINE_float('l2_reg',0.0001,'l2 regularization for embeddings')

model_path = 'E:/NLP/Attention-over-Attention2/models_result/' + FLAGS.name

if not os.path.exists(model_path):
    os.makedirs(model_path)

def read_records(index = 0):
    # 生成读取数据的队列，要指定epoches
    train_queue = tf.train.string_input_producer(['training.tfrecords'],num_epochs = FLAGS.epochs)
    validation_queue = tf.train.string_input_producer(['validation.tfrecords'], num_epochs=FLAGS.epochs)
    test_queue = tf.train.string_input_producer(['test.tfrecords'], num_epochs=FLAGS.epochs)

    queue = tf.QueueBase.from_list(index,[train_queue,validation_queue,test_queue])
    # 定义一个recordreader对象，用于数据的读取
    reader = tf.TFRecordReader()
    # 从之前的队列中读取数据到serialized_example
    _,serialized_example = reader.read(queue)
    # 调用parse_single_example函数解析数据
    features = tf.parse_single_example(
        serialized_example,
        features={
            'document': tf.VarLenFeature(tf.int64),
            'query': tf.VarLenFeature(tf.int64),
            'answer': tf.FixedLenFeature([], tf.int64)
        }
    )

    # 返回索引、值、shape的三元组信息(1-D tensor)
    document = sparse_ops.serialize_sparse(features['document'])
    query = sparse_ops.serialize_sparse(features['query'])
    answer = features['answer']

    # 生成batch切分数据
    document_batch_serialized, query_batch_serialized, answer_batch = tf.train.shuffle_batch(
        [document,query,answer],
        batch_size= FLAGS.batch_size,
        capacity= 2000,
        min_after_dequeue= 1000
    )

    sparse_document_batch = sparse_ops.deserialize_many_sparse(document_batch_serialized,dtype=tf.int64)
    sparse_query_batch = sparse_ops.deserialize_many_sparse(query_batch_serialized,dtype=tf.int64)

    # 变相将不同长短的句子用0 padding
    # tf.sparse_tensor_to_dense ----return:[batch_size,seq_length]
    document_batch = tf.sparse_tensor_to_dense(sparse_document_batch)

    # 赋值权重，将每个batch多条句子中的word用1表示，其余用0表示
    # tf.sparse_to_dense( sparse_indices,output_shape,sparse_values(指定元素,默认为1),default_value(其他元素，默认为0))
    # tf.sparse_to_dense ---- return:[batch_size, seq_length]
    document_weights = tf.sparse_to_dense(sparse_document_batch.indices, sparse_document_batch.dense_shape, 1)

    query_batch = tf.sparse_tensor_to_dense(sparse_query_batch)
    query_weights = tf.sparse_to_dense(sparse_query_batch.indices,sparse_query_batch.dense_shape,1)

    return document_batch,document_weights,query_batch,query_weights,answer_batch

def inference(documents, doc_mask, query, query_mask):
    '''

    :param documents: document_batch -------[batch_size,seq_length]
    :param doc_mask: document_weights ------[batch_size, seq_length]
    :param query: query_batch
    :param query_mask: query_weights
    :return:
    '''

    # 1.Contextual Embedding：将one-hot向量输入embedding层
    #   这里的文档嵌入层和问题嵌入层的权值矩阵共享，通过共享词嵌入，文档和问题都可以参与嵌入的学习过程，
    #   然后使用双向GRU分别对文档和问题进行编码，文档和问题的编码都拼接正向和反向GRU的隐藏层输出，
    #   这时编码得到的文档和问题词向量都包含了上下文信息

    embedding = tf.get_variable('embedding',
                                [FLAGS.vocab_size, FLAGS.embedding_size],
                                initializer=tf.random_uniform_initializer(minval=-0.05,maxval=0.05))

    regularizer = tf.nn.l2_loss(embedding)

    doc_emb = tf.nn.dropout(tf.nn.embedding_lookup(embedding,documents), FLAGS.dropout_keep_prob)
    doc_emb.set_shape([None, None, FLAGS.embedding_size])

    query_emb = tf.nn.dropout(tf.nn.embedding_lookup(embedding, query), FLAGS.dropout_keep_prob)
    query_emb.set_shape([None, None, FLAGS.embedding_size])

    with tf.variable_scope('document', initializer= orthogonal_initializer()):
        fwd_cell = tf.contrib.rnn.GRUCell(FLAGS.hidden_size)
        back_cell = tf.contrib.rnn.GRUCell(FLAGS.hidden_size)

        # 得到一个batch内各个句子长度的列表，[batch_size,1]
        doc_len = tf.reduce_sum(doc_mask, reduction_indices=1)

        # tf.nn.bidirectional_dynamic_rnn,实现双向lstm,下面的h为outputs
        # return:(outputs, output_states)
        # outputs为(output_fw, output_bw)，是一个包含前向cell输出tensor和后向cell输出tensor组成的元组。
        # 假设 time_major=false,tensor的shape为[batch_size, max_time, depth],相当于[batch_size, seq_len, embedding_size]。
        # 实验中使用tf.concat(outputs, 2)将其拼接。
        # ------------------------------------------------------------------------------------------------------
        # output_states为(output_state_fw, output_state_bw)，包含了前向和后向最后的隐藏状态的组成的元组。
        # output_state_fw和output_state_bw的类型为LSTMStateTuple。
        # LSTMStateTuple由（c，h）组成，分别代表memory cell和hidden state。
        # sequence_length: (optional) An int32/int64 vector, size [batch_size],
        # containing the actual lengths for each of the sequences in the batch.
        h, _ = tf.nn.bidirectional_dynamic_rnn(
            fwd_cell,back_cell,doc_emb,sequence_length=tf.to_int64(doc_len),dtype=tf.float32
        )
        h_doc = tf.concat(h, 2)

    with tf.variable_scope('query', initializer=orthogonal_initializer()):
        fwd_cell = tf.contrib.rnn.GRUCell(FLAGS.hidden_size)
        back_cell = tf.contrib.rnn.GRUCell(FLAGS.hidden_size)

        query_len = tf.reduce_sum(query_mask, reduction_indices=1)
        h, _ = tf.nn.bidirectional_dynamic_rnn(
            fwd_cell, back_cell, query_emb, sequence_length=tf.to_int64(query_len), dtype=tf.float32)
        # h_query = tf.nn.dropout(tf.concat(2, h), FLAGS.dropout_keep_prob)
        h_query = tf.concat(h, 2)


    # 2.Pair-wise Matching Score
    # 将h_doc和h_query进行点积计算得到成对匹配矩阵（pair-wise matching matrix）
    # M(i, j)代表文档中第i个单词的Contextual Embedding与问题中第j个单词的Contextual Embedding的点积之和.
    # M(i, j)这个矩阵即图1中点积之后得到的矩阵，横向表示问题，纵向表示文档，故M(i, j)的维度为|D| * |Q|。

    # adjoint_b: 如果为真, b则在进行乘法计算前进行共轭和转置
    M = tf.matmul(h_doc, h_query, adjoint_b=True)
    # -1表示最后一维，将doc和query最后再加一维向量，变成M_mask:[batch_size,D_size,1] x [batch_size,1，Q_size]
    M_mask = tf.to_float(tf.matmul(tf.expand_dims(doc_mask, -1), tf.expand_dims(query_mask, 1)))

    # M:[batch_size,D_size,Q_size],
    # alpha:[batch_size,D_size,Q_size],
    # beta:[batch_size,D_size,Q_size]
    alpha = softmax(M, 1, M_mask)
    beta = softmax(M, 2, M_mask)

    # tf.reduce_sum(beta, 1)---[batch_size,1,Q_size]
    # tf.expand_dims(doc_len, -1)---[batch_size,1,1]
    # query_importance---[1,Q_size,1]
    query_importance = tf.expand_dims(tf.reduce_sum(beta, 1) / tf.to_float(tf.expand_dims(doc_len, -1)), -1)

    # s:  [batch_size,D_size,1] ---> [batch_size,D_size]
    s = tf.squeeze(tf.matmul(alpha, query_importance), [2])

    # tf.unstack(s, FLAGS.batch_size) ---> [1,D_Size]
    # unpacked_s: [1, D_szie] ,里面的元素为：(s, document)
    unpacked_s = zip(tf.unstack(s, FLAGS.batch_size), tf.unstack(documents, FLAGS.batch_size))
    y_hat = tf.stack([tf.unsorted_segment_sum(attentions, sentence_ids, FLAGS.vocab_size) for (attentions, sentence_ids) in unpacked_s])

    return y_hat, regularizer

def train(y_hat, regularizer, document, doc_weight, answer):
    index = tf.range(0, FLAGS.batch_size) * FLAGS.vocab_size +tf.to_int32(answer)
    # 将矩阵t变换为一维矩阵
    flat = tf.reshape(y_hat, [-1])
    #根据index选出flat中的值
    relevant = tf.gather(flat, index)

    loss = -tf.reduce_mean(tf.log(relevant)) + FLAGS.l2_reg * regularizer

    global_step = tf.Variable(0, name = 'global_step', trainable=False)

    accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(y_hat, 1), answer)))

    optimizer = tf.train.AdamOptimizer()
    grads_and_vars = optimizer.compute_gradients(loss)
    capped_grads_and_vars = [(tf.clip_by_value(grad, -5, -5), var) for (grad, var) in grads_and_vars]
    train_op = optimizer.apply_gradients(capped_grads_and_vars,global_step = global_step)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    return loss, train_op, global_step, accuracy

def main():
    dataset = tf.placeholder_with_default(0, [])
    document_batch, document_weights, query_batch,query_weights, answer_batch  =  read_records(dataset)

    y_hat, reg = inference(document_batch, document_weights, query_batch, query_weights)
    loss, train_op, global_step, accuracy = train(y_hat, reg, document_batch, document_weights, answer_batch)
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(model_path, sess.graph)
        saver_variables = tf.all_variables()
        if not FLAGS.training:
            saver_veriables = filter(lambda var: var.name != 'input_producer/limit_epochs/epochs:0', saver_variables)
            saver_variables = filter(lambda var: var.name != 'smooth_acc:0', saver_variables)
            saver_variables = filter(lambda var: var.name != 'avg_acc:0', saver_variables)
        saver = tf.train.Saver(saver_veriables)

        sess.run([
            tf.initialize_all_variables(),
            tf.initialize_local_variables()
        ])

        model = tf.train.latest_checkpoint(model_path)
        if model:
            print('Restoring' + model)
            saver.restore(sess, model)

        # tf.train.Coordinator()来创建一个线程管理器（协调器）对象。
        coord = tf.train.Coordinator()
        # 调用tf.train.start_queue_runners，把tensor推入内存序列中，供计算单元调用，否则会由于内存序列为空，数据流图会处于一直等待状态
        threads = tf.train.start_queue_runners(coord = coord)

        start_time = time.time()
        accumulated_accuracy = 0

        try:
            if FLAGS.training:
                while not coord.should_stop():
                    loss_t, _, step, acc = sess.run([loss,train_op,accuracy],
                                                    feed_dict={dataset: 0})
                    elapsed_time, start_time = time.time() - start_time,time.time()
                    print(step, loss_t, acc, elapsed_time)
                    if step % 100 == 0:
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)
                    if step % 1000 == 0:
                        saver.save(sess, model_path + '/aoa', global_step= step)


            else:
                step = 0
                while not coord.should_stop():
                    acc = sess.run(accuracy, feed_dict={dataset: 2})
                    step +=1
                    accumulated_accuracy += (acc - accumulated_accuracy) / step
                    elapsed_time, start_time = time.time() - start_time, time.time()
                    print(accumulated_accuracy, acc, elapsed_time)
        except tf.errors.OutOfRangeError:
            print('Done!')
        finally:
            coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()