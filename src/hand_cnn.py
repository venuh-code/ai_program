import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import scipy
from PIL import Image
from scipy import ndimage
import pdb

def load_dataset():
    train_dataset = h5py.File('../datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('../datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
    
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])
    return X, Y
    
def initialize_parameters():
    
    tf.set_random_seed(1)                            
        
    #使用`tf.contrib.layers.xavier_initializer(seed = 0)`来初始化W1。
    # W1的维度是[4, 4, 3, 8],表示第一个卷积层过滤器矩阵的[高，宽，深度，个数]
    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    #初始化W2
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    
    #有同学会问，为什么不初始化阈值和全连接层的相关参数呢？
    #因为TensorFlow会自动初始化它们，不需要我们操心

    parameters = {"W1": W1,
                  "W2": W2}
    
    return parameters
    
def forward_propagation(X, parameters):
    """
    这个函数会实现如下的前向传播流程:
    CONV2D卷积 -> RELU激活 -> MAXPOOL池化 -> CONV2D卷积 -> RELU激活 -> MAXPOOL池化 -> FLATTEN扁平化 -> 全连接层
    
    参数:
    X -- 输入特征的占位符
    parameters -- 之前我们初始化好的"W1", "W2"参数

    Returns:
    Z3 -- 最后一个全连接层的输出
    """
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize = [1, 8, 8, 1], strides = [1, 8, 8, 1], padding='SAME')
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding='SAME')
    P = tf.contrib.layers.flatten(P2)
    # 指定该全连接层有6个神经元。
    # activation_fn=None表示该层没有激活函数，因为后面我们会再接一个softmax层
    Z3 = tf.contrib.layers.fully_connected(P, 6, activation_fn=None)

    return Z3
    
def compute_cost(Z3, Y):
    """
    参数:
    Z3 -- 前面forward_propagation的输出结果，维度是(6, 样本数)
    Y -- 真实标签的占位符，维度当然也是(6, 样本数)
    
    返回值:
    cost - 返回一个tensorflow张量，它代表了softmax激活以及成本计算操作。
    """
    
    print("Z3.shape is " + str(Z3.shape))
    print("Y.shape is " + str(Y.shape))
    # tf.nn.softmax_cross_entropy_with_logits函数不仅仅执行了softmax激活，还将成本也给计算了。
    # tf.reduce_mean本用来获取平均值。在这里被用于获取多个样本的平均损失，即获取成本。
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    
    return cost
    
def model(X_train, Y_train, X_test, Y_test, learning_rate=0.002,
          num_epochs=300, minibatch_size=128, print_cost=True):
    """
    参数:
    X_train -- 训练集数据，维度是(1080, 64, 64, 3)
    Y_train -- 训练集标签, 维度是(1080, 6)
    X_test -- 测试集数据, 维度是(120, 64, 64, 3)
    Y_test -- 测试集标签, 维度是(120, 6)
    
    返回值:
    train_accuracy -- 训练集上的预测精准度
    test_accuracy -- 测试集上的预测精准度
    parameters -- 训练好的参数
    """
    
    ops.reset_default_graph()                         # 重置一下tf框架
    tf.set_random_seed(1)                       
    seed = 3                                       
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]   # n_y是标签的类别数量，这里是6
    costs = []                                     
    
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    # 我们使用adam来作为优化算法
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
     
    with tf.Session() as sess:
        sess.run(init)     
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) 
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
                # 执行session。训练正式开始。每一次训练一个子训练集minibatch
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches
                

            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        #pdb.set_trace()

        # 计算预测精准度
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("训练集预测精准度:", train_accuracy)
        print("测试集预测精准度:", test_accuracy)
                
        return train_accuracy, test_accuracy, parameters

if __name__ == '__main__':
    np.random.seed(1)
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T
    print ("训练样本数 = " + str(X_train.shape[0]))
    print ("测试样本数 = " + str(X_test.shape[0]))
    print ("X_train的维度: " + str(X_train.shape))
    print ("Y_train的维度: " + str(Y_train.shape))
    print ("X_test的维度: " + str(X_test.shape))
    print ("Y_test的维度: " + str(Y_test.shape))

    _, _, parameters = model(X_train, Y_train, X_test, Y_test)
    
'''    
Cost after epoch 795: 0.012185
Tensor("Mean_1:0", shape=(), dtype=float32)
训练集预测精准度: 1.0
测试集预测精准度: 0.94166666
'''