# @Time     : Jan. 12, 2019 19:01
# @Author   : Veritas YIN
# @FileName : base_model.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from models.layers import *
from os.path import join as pjoin
import tensorflow as tf


def build_model(inputs, n_his, Ks, Kt, blocks, keep_prob):
    '''
    Build the base model.
    :param inputs: placeholder.
    :param n_his: int, size of historical records for training.
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param blocks: list, channel configs of st_conv blocks.
    :param keep_prob: placeholder.
    '''
    #build_model(x, n_his, Ks, Kt, blocks, keep_prob)
    #x = tf.placeholder(tf.float32, [None, n_his + 1, n, 1], name='data_input')
    #keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    #inputs: [None, n_his + 1, n, 1] [, 13,228,1]
    x = inputs[:, 0:n_his, :, :]

    # Ko>0: kernel size of temporal convolution in the output layer.
    Ko = n_his #12
    # ST-Block
    
    #blocks: [[1, 32, 64], [64, 32, 128]]
    for i, channels in enumerate(blocks):
        x = st_conv_block(x, Ks, Kt, channels, i, keep_prob, act_func='GLU')
        Ko -= 2 * (Ks - 1)
        #print(i,channels,x,Ko)
    #0 [1, 32, 64] Tensor("dropout/mul:0", shape=(?, 8, 228, 64), dtype=float32) 8
    #1 [64, 32, 128] Tensor("dropout_1/mul:0", shape=(?, 4, 228, 128), dtype=float32) 4
    

    # Output Layer
    if Ko > 1:
        y = output_layer(x, Ko, 'output_layer',blocks[0][0])
    else:
        raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    tf.add_to_collection(name='copy_loss',
                         value=tf.nn.l2_loss(inputs[:, n_his - 1:n_his, :, :] - inputs[:, n_his:n_his + 1, :, :]))
    print(y)
    print(inputs)
    #不加 最后一层的全连接
    # Tensor("output_layer_out/Sigmoid:0", shape=(?, 1, 16, 128), dtype=float32) 
    # Tensor("data_input:0", shape=(?, 11, 16, 8), dtype=float32)
    
    #加最后一层的全连接
    #Tensor("add:0", shape=(?, 1, 16, 1), dtype=float32)
    #Tensor("data_input:0", shape=(?, 11, 16, 8), dtype=float32)
    #import sys
    #sys.exit(0)
    
    train_loss = tf.nn.l2_loss(y - inputs[:, n_his:n_his + 1, :, :])
    single_pred = y[:, 0, :, :]
    tf.add_to_collection(name='y_pred', value=single_pred)
    
    #y: Tensor("add:0", shape=(?, 1, 228, 1), dtype=float32)
    #train_loss: Tensor("L2Loss_2:0", shape=(), dtype=float32)
    #single_pred: Tensor("strided_slice_4:0", shape=(?, 228, 1), dtype=float32)

    return train_loss, single_pred


def model_save(sess, global_steps, model_name, save_path='./output/models/'):
    '''
    Save the checkpoint of trained model.
    :param sess: tf.Session().
    :param global_steps: tensor, record the global step of training in epochs.
    :param model_name: str, the name of saved model.
    :param save_path: str, the path of saved model.
    :return:
    '''
    saver = tf.train.Saver(max_to_keep=3)
    prefix_path = saver.save(sess, pjoin(save_path, model_name), global_step=global_steps)
    print(f'<< Saving model to {prefix_path} ...')
