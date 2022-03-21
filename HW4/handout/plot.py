
import numpy as np
import sys
import matplotlib.pyplot as plt

def read_in(file):
    data = np.genfromtxt(file, delimiter="\t", dtype=None, encoding=None)
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y

def read_t(f):
    data = np.genfromtxt(f, delimiter=" ", dtype=None, encoding=None)
    x_1 = np.array([data[i][0] for i in range(len(data))])
    y_1 = np.array([data[i][1] for i in range(len(data))])
    return x_1, y_1

def sigmoid(x):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def dJ(theta, X, Y, i):
    '''
    Return the gradient of the ith sample. Don't change theta
    '''
    #first fetch the ith X and Y
    label = Y[i]
    features = X[i]
    #compute gradient
    # #y^i - theta x
    p = label - sigmoid(np.dot(theta, features))
    g = p * features  
    #foo = np.vectorize(f)
    #g = foo(features, label, pre) 
    #g = np.dot((label - sigmoid(pre)), features)
    return g

def likely(theta, X, y):
    a = np.multiply(X, theta)
    h = sigmoid(np.sum(a, axis = 1))
    h[y==0] = 1 - h[y==0]
    res = np.sum(np.log(h), axis = 0)/len(X)
    return -res

def train(theta1, theta2, X, y, X2, y2, num_epoch, l_rate):
    x_ax = []
    y_1 = []
    y_2 = []
    for epoch in range(num_epoch):
        for i in range(len(X)):
            g1 = dJ(theta1, X, y, i)
            theta1 = theta1 + l_rate* g1
            g2 = dJ(theta2, X2, y2, i)
            theta2 = theta2 + l_rate* g2
        l1 = likely(theta1, X, y)
        x_ax.append(epoch)
        l2 = likely(theta2, X2, y2)
        y_1.append(l1)
        y_2.append(l2)
    return theta1, theta2, x_ax, y_1, y_2

def add_bias(X):
    ones = np.ones((np.shape(X)[0], 1), dtype = X.dtype)
    X_new = np.hstack((ones, X))
    return X_new


if __name__ == '__main__':
    val1_in = sys.argv[1]
    val2_in = sys.argv[2]
    val3_in = sys.argv[3]

    #num_epoch = int(sys.argv[2])
    #l_rate = float(sys.argv[4])

    val_1X, val1_Y = read_in(val1_in)
    val_2X, val2_Y = read_in(val2_in)
    val_3X, val3_Y = read_t(val3_in)

    Theta1 = np.zeros((np.shape(val_1X)[1] + 1, ))
    Theta2 = np.zeros((np.shape(val_2X)[1] + 1, ))
    x1 = add_bias(val_1X)
    x2 = add_bias(val_2X)

    theta1, theta2, x_ax, y_1, y_2 = train(Theta1, Theta2, x1, val1_Y, x2, val2_Y, 5000, 0.00001)
    
    
    plt.plot(x_ax, y_1, label = "model 1")
    plt.plot(x_ax, y_2, label = "model 2")
    plt.plot(val_3X, val3_Y, label = "model 3")
    # naming the x axis
    plt.xlabel('epoch')
    # naming the y axis
    plt.ylabel('-log likelyhood')
    # giving a title to my graph
    plt.title('epoch v. negative log likelyhood')
    
    # show a legend on the plot
    plt.legend()
    
    # function to show the plot
    plt.show()


#python plot.py largeoutput/model1_formatted_valid.tsv largeoutput/model2_formatted_valid.tsv model3_val_nll.txt
