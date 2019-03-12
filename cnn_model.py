# coding: utf-8

import tensorflow as tf
from tensorflow.python.ops import array_ops


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 256  # 词向量维度
    seq_length = 200  # 序列长度
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目, default: 256
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元, default: 128

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    l2_reg_lambda = 0.0  # l2 norm

    batch_size = 128  # 每批训练大小
    num_epochs = 15  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'), tf.name_scope('embed'):
            embedding = tf.Variable(tf.random_uniform([self.config.vocab_size, self.config.embedding_dim], -1.0, 1.0), name='embedding')
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            # self.loss = tf.reduce_mean(cross_entropy)
            self.loss = focal_loss_on_object_detection(self.logits, self.input_y)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def focal_loss_on_object_detection(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):

    """Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Link:
    https://github.com/ailias/Focal-Loss-implement-on-Tensorflow

    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)


def focal_loss_sigmoid_on_multi_classification(labels, logits, gamma=2):
    """
    link:
    https://github.com/hxtkyne/focal-loss-with-tensorflow/blob/master/focal_loss.py

    description:
        基于多类别的focal loss计算（其实就是转换成one-hot处理）
    Args:
        labels: [batch_size], dtype=int32，值为0或者1
        logits: [batch_size, num_classes], dtype=float32
        gamma：focal loss的超参数
    Returns:
        tensor: [batch_size]
    """
    y_pred = tf.nn.softmax(logits, dim=-1)  # [batch_size, num_classes]
    # labels = tf.to_float(labels) # label example: [0,1,2,3]
    # labels = tf.one_hot(labels, depth=y_pred.shape[1])  # [0,1,2,3] -> [[0.,0.,0.,0.], [0.,1.,0.,.0], xxx], dtype=float32

    loss = -labels * ((1 - y_pred) ** gamma) * tf.log(y_pred)
    loss = tf.reduce_sum(loss)
    return loss
