
import numpy as np # 
import matplotlib.pyplot as plt # 这个库是用来画图的
import h5py #        
import skimage.transform as tf # 这里我们用它来缩放图片

def load_dataset():
    train_dataset = h5py.File('../datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    
    test_dataset = h5py.File('../datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 
    
    classes = np.array(test_dataset["list_classes"][:])
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0])) # 把数组的维度从(209,)变成(1, 209)，这样好方便后面进行计算
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0])) # 从(50,)变成(1, 50)
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
 
def load_data():
    return load_dataset
    
def dataset_flatten(train_set_x_orig, test_set_x_orig):
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T 
    
    train_set_x = train_set_x_flatten/255.
    test_set_x = test_set_x_flatten/255.
    
    print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    
    return train_set_x, test_set_x
    
def load_dataset_test():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    
    print ("train_set_x_orig shape: " + str(train_set_x_orig.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x_orig shape: " + str(test_set_x_orig.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))
    print("----------------------------------------------")
    dataset_flatten(train_set_x_orig, test_set_x_orig)
    
def open_img(my_image, num_px):
    fname = "../images/" + my_image
    image = np.array(plt.imread(fname))
    plt.imshow(image)
    my_image = tf.resize(image,(num_px,num_px), mode='reflect').reshape((1, num_px*num_px*3)).T
    return my_image
    
def show_costs(costs, learning_rate):
    plt.plot(costs)
    plt.ylabel('cost') # 成本
    plt.xlabel('iterations (per hundreds)') # 横坐标为训练次数，以100为单位
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y
    
if __name__ == '__main__':
    load_dataset_test()
