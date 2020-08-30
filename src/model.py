import numpy as np # 
import matplotlib.pyplot as plt # 这个库是用来画图的
import sklearn # 这个新库是用于数据挖掘，数据分析和机器学习的库，例如它里面就内置了很多人工智能函数
import sklearn.datasets
import sklearn.linear_model
import math
from utils import *
from algorithm import *

def model_one_nn(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """    
    参数:
    X_train -- 训练图片,维度是(12288, 209)
    Y_train -- 训练图片对应的标签,维度是 (1, 209)
    X_test -- 测试图片,维度是(12288, 50)
    Y_test -- 测试图片对应的标签,维度是 (1, 50)
    """
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)
    
    # 打印出预测的准确率
    print("对训练图片的预测准确率为: {}%".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("对测试图片的预测准确率为: {}%".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
    
def run_model_one_nn():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    train_set_x, test_set_x = dataset_flatten(train_set_x_orig, test_set_x_orig)
    num_px = test_set_x_orig.shape[1]
    
    d = model_one_nn(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 4000, learning_rate = 0.005, print_cost = True)
    costs = np.squeeze(d['costs'])
    show_costs(costs, d['learning_rate'])
    
    print("看，这张图片是什么")
    my_image = open_img("cat.jpg" , num_px)
    my_predicted_image = predict(d["w"], d["b"], my_image)
  
    if str(int(np.squeeze(my_predicted_image))):
        print("预测结果为: 猫 ")
    else:
        print("预测结果为: 不是猫 ")

'''======================================== 浅神经元BP, batch ================================================='''
def model_shallow_nn(X, Y, n_h, num_iterations = 10000, print_cost = False):
    np.random.seed(3)
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    parameters = initialize_parameters_shallow(n_x,n_h,n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    for i in range(0, num_iterations):
        A2, cache = forward_propagation_shallow(X, parameters)
        cost = compute_cost_shallow(A2, Y, parameters)
        grads = backward_propagation_shallow(parameters, cache, X, Y)
        parameters = update_parameters_shallow(parameters, grads)
        if print_cost and i % 1000 == 0:
            print("在训练%i次后， 成本是：%f" %(i, cost))
            
    return parameters
    
def run_model_shallow_nn(n_h=4, mode=1):
    X, Y = load_planar_dataset()
    
    if mode:
        parameters = model_shallow_nn(X, Y, n_h , num_iterations=10000, print_cost=True)
        predictions = predict_shallow(parameters, X)
        print ('预测准确率是: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
    else:
        clf = sklearn.linear_model.LogisticRegressionCV();
        clf.fit(X.T, Y.T.ravel());
        LR_predictions = clf.predict(X.T)
        print ('预测准确度是: %d ' % float((np.dot(Y, LR_predictions) + np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) + '% ')
    
'''========================================  深度 ================================================='''
def dnn_train(X, Y, parameters, t, v, s, learning_rate, use_adam, overfit, check_grd):
    if overfit == 0:
        AL, caches = L_model_forward(X, parameters,0)
        cost = compute_cost(AL, Y)
    elif overfit == 2: #dropout
        AL, caches = L_model_forward(X, parameters,1)
        cost = compute_cost(AL, Y)
    elif overfit == 1: #L2
        AL, caches = L_model_forward(X, parameters,0)
        cost = compute_cost_with_regularization(AL, Y, parameters, lambd=0.7)
    grads = L_model_backward(AL, Y, caches, overfit)
    if (check_grd):
        gradient_check_n(parameters, grads, X, Y)
  
    if (use_adam):
        t = t + 1
        parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,t, learning_rate)
    else:
        parameters = update_parameters(parameters, grads, learning_rate)
    
    return cost

def dnn_model(X, Y, layers_dims, learning_rate=0.0075, num_epochs=100, print_cost=False): 
    np.random.seed(1)
    costs = []   
    seed = 10
    t = 0
    parameters = initialize_parameters_deep(layers_dims)
    v,s = initialize_adam(parameters)
    mini_batch_size = 32
    minibatch = 0
    use_adam = 0
    overfit = 0
    
    for i in range(0, num_epochs):
        if i == 0:
            check_grd = 1
        else:
            check_grd = 0
            
        if minibatch:
            seed = seed + 1
            minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
            
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                cost = dnn_train(minibatch_X, minibatch_Y, parameters, t, v, s, learning_rate, use_adam, overfit,check_grd)
        else:
            cost = dnn_train(X, Y, parameters, t, v, s, learning_rate, use_adam, overfit, check_grd)                                                                                           
        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # 画出成本曲线图
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
'''
1、模型：4层
2、数据集：查看数据元方法 str(train_set_x_orig.shape)
训练集：209
测试集：50，是否同源？
贝叶斯误差0.01%，训练集误差0.001%，测试误差0.16%，过拟合
'''
def run_model_dnn():
    #layers_dims = [12288, 41, 21, 11, 3, 1]
    layers_dims = [12288, 1]
    
    train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()
    train_x, test_x = dataset_flatten(train_x_orig, test_x_orig)
    
    print("训练样本数：%d" %(train_x.shape[1]))
    parameters = dnn_model(train_x, train_y, layers_dims, num_epochs=1, print_cost=True, learning_rate=0.0075)
    pred_train = predict_dnn(train_x,parameters)
    
    print("预测训练集准确率是: "  + str(np.sum((pred_train == train_y) / train_x.shape[1])))
    pred_test = predict_dnn(test_x,parameters)
    print("预测测试集准确率是: "  + str(np.sum((pred_test == test_y) / test_x.shape[1])))

if __name__ == '__main__':
    #run_model_one_nn()
    #run_model_shallow_nn(10,1)
    run_model_dnn()