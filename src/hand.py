import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
import scipy
from PIL import Image
from scipy import ndimage
import skimage.transform as sktf # 这里我们用它来缩放图片

def create_placeholders(n_x, n_y):
    """
    参数:
    n_x -- 图片向量的大小，本例中是12288
    n_y -- 类别数量，因为是0到5的数字，所以数量是6
    """

    ### 注意下面代码中，样本数量的位置我们填写的是None，因为在执行训练时，用到的样本数量是不同的。
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
    
    return X, Y
    
def initialize_parameters():
    
    # 将tensorflow中的随机数种子设为1，这样可以确保我们每次执行代码时随机数都是一样的。
    # 如果随机数不一样，那么我执行代码的结果与你们执行的结果就会不一样。
    tf.set_random_seed(1)  
        
    # 我们用tensorflow内置的xavier_initializer函数来进行w的初始化。
    # 教程前面我曾专门花了一篇文章来给大家介绍参数的初始化。
    # 初始化步骤非常重要，好的初始化可以让神经网络学得很快。
    # 之前我们用纯python时，为了实现高质量的参数初始化，需要写不少代码。
    # 而tensorflow已经给我们实现了高效的函数，我们只需要用一行代码调用它就可以对w进行初始化了。
    W1 = tf.get_variable("W1", [25, 12288], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters
    
def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1,X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    
    return Z3
    
def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost
    

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, 
                num_epochs = 800, minibatch_size = 64, print_cost = True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
 
    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m/minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
                
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
            
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        parameters = sess.run(parameters)
        print("Parameters have been trained!")
        
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters
        
def get_dataset():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    index = 0
    #plt.imshow(X_train_orig[index])
    #print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
    # 扁平化
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    # 简单的归一化
    X_train = X_train_flatten / 255.
    X_test = X_test_flatten / 255.
    # one hot编码
    Y_train = convert_to_one_hot(Y_train_orig, 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6)
    
    return X_train, X_test, Y_train, Y_test
    
def hand_predict(parameters):
    my_image = "hand.jpg"
    fname = "../images/" + my_image
    image = np.array(plt.imread(fname))
    my_image = sktf.resize(image,(64, 64),mode='reflect').reshape((1, 64 * 64 * 3)).T
               #sktf.resize(image,(num_px,num_px), mode='reflect').reshape((1, num_px*num_px*3)).T
    my_image_prediction = predict(my_image, parameters)

    #plt.imshow(image)
    print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
    
if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = get_dataset()
    parameters = model(X_train, Y_train, X_test, Y_test)
    hand_predict(parameters)

