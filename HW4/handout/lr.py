import numpy as np
import sys

def read_in(file):
    data = np.genfromtxt(file, delimiter="\t", dtype=None, encoding=None)
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y

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

def train(theta, X, y, num_epoch, learning_rate):
    for epoch in range(num_epoch):
        for i in range(len(X)):
            g = dJ(theta, X, y, i)
            theta = theta + learning_rate * g
    return theta


def predict(theta, X):
    a = np.multiply(X, theta)
    h = sigmoid(np.sum(a, axis = 1))
    res = np.zeros(np.shape(h), dtype=int)
    res[h>0.5] = 1
    return res


def likely(theta, X, y):
    a = np.multiply(X, theta)
    h = sigmoid(np.sum(a, axis = 1))
    h[y==0] = 1 - h[y==0]
    res = np.sum(np.log(h), axis = 0)/len(X)
    return res

def compute_error(y_pred, y):
    Y = y_pred + y
    error = np.count_nonzero(Y == 1)
    return round(error/len(y), 6)

def add_bias(X):
    ones = np.ones((np.shape(X)[0], 1), dtype = X.dtype)
    X_new = np.hstack((ones, X))
    return X_new

if __name__ == '__main__':
    train_in = sys.argv[1]
    valid_in = sys.argv[2]
    test_in = sys.argv[3]

    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

    num_epoch = int(sys.argv[7])
    l_rate = float(sys.argv[8])

    train_X, train_Y = read_in(train_in)
    test_X, test_Y = read_in(test_in)

    Theta = np.zeros((np.shape(train_X)[1] + 1, ))
    tr_X = add_bias(train_X)
    te_X = add_bias(test_X)
    theta = train(Theta, tr_X, train_Y, num_epoch, l_rate)
    
    y_pred_tr = predict(theta, tr_X)
    y_pred_te = predict(theta, te_X)
    tr_err = compute_error(y_pred_tr, train_Y)
    te_err = compute_error(y_pred_te, test_Y)

    with open(train_out, 'w') as f1:
        f1.write(str(y_pred_tr[0]))
        for i in range(1, len(y_pred_tr)):
            f1.write('\n' + str(y_pred_tr[i]))

    with open(test_out, 'w') as f2:
        f2.write(str(y_pred_te[0]))
        for i in range(1, len(y_pred_te)):
            f2.write('\n' + str(y_pred_te[i]))
    
    with open(metrics_out, 'w') as f:
        f.write('error(train): ' + str(tr_err) + '\n')    
        f.write('error(test): ' + str(te_err))



#python lr.py largeoutput/model1_formatted_train.tsv largeoutput/model1_formatted_valid.tsv largeoutput/model1_formatted_test.tsv largeoutput/train_out.labels largeoutput/test_out.labels largeoutput/metrics_out.txt 5000 0.00001
