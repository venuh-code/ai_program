import numpy as np # 
import math
import pdb

def sigmoid(z):
    s = 1/(1 + np.exp(-z))
    return s
    
def sigmoid_backward(dA, cache):   
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ
    
def relu(Z):
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)
    return A
    
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ
    
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    
    permutation = list(np.random.permutation(m)) # 这行代码会生成m范围内的随机整数，如果m是3，那么结果可能为[2, 0, 1]
    shuffled_X = X[:, permutation]# 这个代码会将X按permutation列表里面的随机索引进行洗牌。为什么前面是个冒号，因为前面是特征，后面才代表样本数 
    shuffled_Y = Y[:, permutation].reshape((1,m))
    
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]
        
    mini_batch = (mini_batch_X, mini_batch_Y)
    mini_batches.append(mini_batch)
    
    return mini_batches

""" =========================单个神经元BP, batch算法 ========================= """
def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    return w, b
    
def progagate(w, b, X, Y):
    """
    参数:
    w -- 权重数组，维度是(12288, 1)
    b -- 偏置bias
    X -- 图片的特征数据，维度是 (12288, 209)
    Y -- 图片对应的标签，0或1，0是无猫，1是有猫，维度是(1,209)

    返回值:
    cost -- 成本
    dw -- w的梯度
    db -- b的梯度
    """
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b) #行向量(1,209)
    cost = -np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))/m #交叉熵
    dZ = A - Y #行向量(1,209)
    dw = np.dot(X, dZ.T)/m #(12288, 1)
    db = np.sum(dZ)/m #(1,209)
    
    grads = {"dw":dw,"db":db}
    
    return grads, cost
    
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    
    for i in range(num_iterations):
        grads, cost = progagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate*dw #(12288,1)
        b = b - learning_rate*b #(1,209)
        
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("优化%i次后成本是: %f" %(i, cost))
                
    params = {"w":w, "b":b}
    
    return params, costs
  
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    A = sigmoid(np.dot(w.T, X) + b) #行向量(1,209)
    
    for i in range(A.shape[1]):
        if A[0,i] >= 0.5:
            Y_prediction[0,i] = 1
            print("Y_prediction" + str(A[0,i]))
       
    return Y_prediction
    
""" =========================浅神经元BP, batch算法 ========================= """
''' 输入层 + 隐藏层 + 输出层（1个神经元） '''
def initialize_parameters_shallow(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros(shape=(n_h, 1))
    
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros(shape=(n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
    
def forward_propagation_shallow(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    # 实现前向传播算法
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1) # 第一层的激活函数我们使用tanh。numpy库里面已经帮我们实现了tanh工具函数
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2) # 第二层我们使用sigmoid，因为我们要解决的这个问题属于二分问题。这个函数是我们自己在planar_utils里面实现的。

    # 将这些前向传播时得出的值保存起来，因为在后面进行反向传播计算时会用到他们。
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache
    
def compute_cost_shallow(A2, Y, parameters):
    m = Y.shape[1]
    
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = -np.sum(logprobs)/m
    
    return cost
    
def backward_propagation_shallow(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    A1 = cache['A1']
    A2 = cache['A2']
    
    dZ2 = A2 - Y
    dW2 = (1/m)*np.dot(dZ2, A1.T)
    db2 = (1/m)*np.sum(dZ2, axis=1, keepdims=True)
        
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1, 
                "db1": db1,
                "dW2": dW2,
                "db2": db2}
    
    return grads # 返回计算得到的梯度
    
def update_parameters_shallow(parameters, grads, learning_rate=1.2):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    # 根据梯度和学习率来更新参数，使其更优
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2}
    
    return parameters
    
def predict_shallow(parameters, X):
    A2, cache = forward_propagation_shallow(X, parameters)
    predictions = np.round(A2)
    
    return predictions
    
""" =========================深度神经网络BP算法 ========================= """
g_dropout = 0.7
def initialize_parameters_deep(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1,L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])/np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
    return parameters
    
def initialize_adam(parameters) :    
    L = len(parameters) // 2 
    v = {}
    s = {}
    
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

        s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l + 1)])
        s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l + 1)])
    
    return v, s
    
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache
    
def linear_forward_with_dropout(A, W, b, keep_prob=0.5):
    np.random.seed(1)
    D = np.random.rand(A.shape[0], A.shape[1])     # 第一步
    D = D < keep_prob                            # 第二步
    A = A * D                                      # 第三步
    A = A / keep_prob                               # 第四步
    
    Z = np.dot(W, A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b, D)
    
    return Z, cache
    
def linear_activation_forward(A_prev, W, b, dropout, activation):
    if dropout:
        Z, linear_cache = linear_forward_with_dropout(A_prev, W, b, dropout)
    else:
        Z, linear_cache = linear_forward(A_prev, W, b)
    
    if activation == "sigmoid": # 如果该层使用sigmoid        
        A = sigmoid(Z) 
    elif activation == "relu":
        A = relu(Z)
        
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, Z) # 缓存一些变量，后面的反向传播会用到它们

    return A, cache
    
def L_model_forward(X, parameters, dropout_switch):
    caches = []
    A = X
    L = len(parameters)//2
    
    for l in range(1, L):
        A_prev = A
        if l == 1:
            dropout = 0
        elif dropout_switch:
            dropout = g_dropout
        else:
            dropout = 0
        A, cache = linear_activation_forward(A_prev,
                                             parameters['W' + str(l)], 
                                             parameters['b' + str(l)],
                                             dropout,
                                             activation='relu')
        caches.append(cache)# 把一些变量数据保存起来，以便后面的反向传播使用
        
    AL, cache = linear_activation_forward(A, 
                                          parameters['W' + str(L)], 
                                          parameters['b' + str(L)], 
                                          dropout = 0,
                                          activation='sigmoid')
                                          
    caches.append(cache)
   
    assert(AL.shape == (1, X.shape[1]))
            
    return AL, caches
    

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    
    cost = np.squeeze(cost)# 确保cost是一个数值而不是一个数组的形式
    assert(cost.shape == ())
    
    return cost 
    
def compute_cost_with_regularization(AL, Y, parameters,lambd):
    m = Y.shape[1]
    L = len(parameters)//2
     
    # 获得常规的成本
    cross_entropy_cost = compute_cost(AL, Y) 
    L2_regularization_tail = 0
    for l in range(1, L):
        L2_regularization_tail += np.sum(np.square(parameters['W' + str(l)]))
        
    cost = cross_entropy_cost + lambd * L2_regularization_tail / (2 * m)
    
    return cost
    
    
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ, cache[0].T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(cache[1].T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db
    
def linear_backward_with_dropout(dZ, cache, keep_prob):
    A_prev, W, b, D = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ, cache[0].T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(cache[1].T, dZ)
  
    dA_prev = dA_prev * D              # 第一步
    dA_prev = dA_prev / keep_prob              # 第二步
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db
    
def linear_backward_with_regularization(dZ, cache, lambd):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ, cache[0].T)/m + (lambd * W)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(cache[1].T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db
   
def linear_activation_backward(dA, cache, overfit, dropout, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    
    # 这里我们又顺带根据本层的dZ算出本层的dW和db以及前一层的dA
    #dA_prev, dW, db = linear_backward(dZ, linear_cache)
    if overfit == 1:
        dA_prev, dW, db = linear_backward_with_regularization(dZ, linear_cache, 0.9)
    elif overfit == 2 and dropout:
        dA_prev, dW, db = linear_backward_with_dropout(dZ, linear_cache, dropout)
    else :
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db
    
def L_model_backward(AL, Y, caches, overfit):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(
                            dAL, 
                            current_cache,
                            overfit,
                            0,
                            activation = "sigmoid")
                                                                                            
    for c in reversed(range(1,L)):
        if c == 1:
            dropout = 0
        else:
            dropout = g_dropout
        grads["dA" + str(c-1)], grads["dW" + str(c)], grads["db" + str(c)] = linear_activation_backward(
            grads["dA" + str(c)], 
            caches[c-1],
            overfit,
            dropout,
            activation = "relu")

    return grads
    
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 # 获取层数。//除法可以得到整数

    for l in range(1,L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
        
    return parameters
    
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    
    L = len(parameters) // 2                 
    v_corrected = {} # 修正后的值
    s_corrected = {}                        
    
    for l in range(L):
        # 算出v值
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]
        

        # 对v值进行修正
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))
        

        # 算出s值
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads['dW' + str(l + 1)], 2)
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads['db' + str(l + 1)], 2)
    

        # 对s值进行修正
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))
 

        # 更新参数
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon)


    return parameters, v, s
    
def predict_dnn(X, parameters):
    m = X.shape[1]
    n = len(parameters)//2
    p = np.zeros((1, m))
    
    probas, caches = L_model_forward(X, parameters,0)
    
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    
    return p
    
''' =============梯度检验 , 转化为列向量 =================='''  
def L_model_forward_n(X, Y, parameters):
    AL, _ = L_model_forward(X, parameters,0)
    cost = compute_cost(AL, Y)
     
    return cost
    
def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    shapes = {}
    for key in parameters.keys():  
        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]
        shapes[key] = parameters[key].shape
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys, shapes

def vector_to_dictionary(theta, L, keys, shapes):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    W_len = 0
    b_len = 0
    last_len = 0
    for l in range(1, L+1):
        W_len = keys.count("W"+str(l))
        b_len = keys.count("b"+str(l))
        
        #print("L:%d, W_len:%d, b_len:%d" %(l, W_len, b_len))
        #print("W.shape" +str(shapes["W"+str(l)]))
        #print("b.shape" +str(shapes["b"+str(l)]))
        #print("W range,%d - %d" %(last_len, W_len+last_len))
        #print("b range,%d - %d" %(W_len+last_len, W_len+b_len+last_len))
        parameters["W"+str(l)] = theta[last_len:W_len+last_len].reshape(shapes["W"+str(l)])
        parameters["b"+str(l)] = theta[W_len+last_len:(W_len+b_len+last_len)].reshape(shapes["b"+str(l)])
        last_len += W_len + b_len
        
    return parameters

def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    
    count = 0
    for key in gradients.keys():
        # flatten parameter
        
        if key[0:2] == 'dA':
            continue 
            
        new_vector = np.reshape(gradients[key], (-1,1))
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta
    
def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
     
    parameters_values, keys, shapes = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    L = len(parameters) // 2
    #pdb.set_trace()
    # 计算gradapprox
    for i in range(num_parameters):
        thetaplus =  np.copy(parameters_values)                                      
        thetaplus[i][0] = thetaplus[i][0] + epsilon                                 
        J_plus[i] =  L_model_forward_n(X, Y, vector_to_dictionary(thetaplus, L, keys, shapes))  
        
        thetaminus = np.copy(parameters_values)                                     
        thetaminus[i][0] = thetaminus[i][0] - epsilon                                      
        J_minus[i] = L_model_forward_n(X, Y, vector_to_dictionary(thetaminus, L, keys, shapes)) 
        
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
    
    numerator = np.linalg.norm(grad - gradapprox)                                
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)               
    difference = numerator / denominator                                         

    if difference > 1e-7:
        print("反向传播有问题! difference = " + str(difference) )
    else:
        print( "反向传播很完美! difference = " + str(difference))
    
    return difference