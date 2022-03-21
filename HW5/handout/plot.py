import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


def args2data(parser):
    """
    Parse argument, create data and label.
    :return:
    X_tr: train data (numpy array)
    y_tr: train label (numpy array)
    X_te: test data (numpy array)
    y_te: test label (numpy array)
    out_tr: predicted output for train data (file)
    out_te: predicted output for test data (file)
    out_metrics: output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """

    # # Get data from arguments
    out_tr = parser.train_out
    out_te = parser.validation_out
    out_metrics = parser.metrics_out
    n_epochs = parser.num_epoch
    n_hid = parser.hidden_units
    init_flag = parser.init_flag
    lr = parser.learning_rate

    X_tr = np.loadtxt(parser.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr[:, 0] = 1.0 #add bias terms

    X_te = np.loadtxt(parser.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te[:, 0]= 1.0 #add bias terms

    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)



def shuffle(X, y, epoch):
    """
    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]

def random_init(shape):
    """
    Randomly initialize a numpy array of the specified shape
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    # DO NOT CHANGE THIS
    np.random.seed(np.prod(shape))

    # Implement random initialization here
    A = np.random.uniform(low = -0.1, high = 0.1, size = (shape[0], shape[1] - 1))
    zeros = np.zeros((np.shape(A)[0],1))
    res = np.hstack((zeros, A))
    return res


def zero_init(shape):
    """
    Initialize a numpy array of the specified shape with zero
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    A = np.zeros(shape)
    return A

def add_bias(X):
    '''
    Add a bias 1 entry to matrix 
    '''
    ones = np.zeros((np.shape(X)[0], 1), dtype = X.dtype)
    X_new = np.hstack((ones, X))
    return X_new

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

def softMax(b):
    '''
    Implementation of the softmax matrix
    
    :param b: input vector
    :return : output layer of same size
    '''
    B = np.exp(b)
    return B/np.sum(B)


class NN(object):
    def __init__(self, lr, n_epoch, weight_init_fn, input_size, hidden_size, output_size):
        """
        Initialization
        :param lr: learning rate
        :param n_epoch: number of training epochs
        :param weight_init_fn: weight initialization function
        :param input_size: number of units in the input layer
        :param hidden_size: number of units in the hidden layer
        :param output_size: number of units in the output layer
        """
        self.lr = lr
        self.n_epoch = n_epoch
        self.weight_init_fn = weight_init_fn
        self.n_input = input_size
        self.n_hidden = hidden_size
        self.n_output = output_size

        # initialize weights and biases for the models
        # HINT: pay attention to bias here
        # this is bias folded 
        # actual output size of alpha : [self.n_hidden, self.n_input ]
        self.alpha = weight_init_fn([self.n_hidden, self.n_input])
        # actual output size of beta : [self.n_output, self.n_hidden + 1]
        self.beta = weight_init_fn([self.n_output, self.n_hidden + 1])

        # initialize parameters for adagrad
        self.epsilon = 10 ** (-5)
        self.grad_sum_w1 = np.zeros(np.shape(self.alpha))
        self.grad_sum_w2 = np.zeros(np.shape(self.beta))

        # feel free to add additional attributes


def print_weights(nn):
    """
    An example of how to use logging to print out debugging infos.

    Note that we use the debug logging level -- if we use a higher logging
    level, we will log things with the default logging configuration,
    causing potential slowdowns.

    Note that we log NumPy matrices on separate lines -- if we do not do this,
    the arrays will be turned into strings even when our logging is set to
    ignore debug, causing potential massive slowdowns.
    :param nn: your model
    :return:
    """
    logging.debug(f"shape of w1: {nn.w1.shape}")
    logging.debug(nn.w1)
    logging.debug(f"shape of w2: {nn.w2.shape}")
    logging.debug(nn.w2)


def forward(X, nn):
    """
    Neural network forward computation.
    Follow the pseudocode!
    :param X: input data
    :param nn: neural network class
    :return: output probability
    """
    a = np.matmul(nn.alpha, X)
    z = np.insert(sigmoid(a), 0, 1.)
    # need to prepend another entry 1 on top of z, np.insert 1.
    b = np.matmul(nn.beta, z)
    y = softMax(b)
    nn.z = z
    return y


def backward(X, y, y_hat, nn):
    """
    Neural network backward computation.
    Follow the pseudocode!
    :param X: input data
    :param y: label
    :param y_hat: prediction
    :param nn: neural network class
    :return:
    d_w1: gradients for w1
    d_w2: gradients for w2
    """
    #a = np.matmul(nn.alpha, X)
    #z = np.insert(sigmoid(a), 0, 1.)
    # need to prepend another entry 1 on top of z, np.insert 1.
    #b = np.matmul(nn.beta, z)
    #y = softMax(b)
    y_vec = np.zeros((nn.n_output, ))
    y_vec[y] = 1
    d_b = y_hat - y_vec
    d_beta = np.matmul(np.transpose([d_b]), [nn.z])
    
    beta_sim = nn.beta[:, 1:]
    d_z = np.matmul(d_b, beta_sim)
    z_sim = nn.z[1:]
    d_a = np.multiply(d_z, np.multiply(z_sim, 1-z_sim))
    d_alpha = np.matmul(np.transpose([d_a]), [X])
    return d_alpha, d_beta


def test(X, y, nn):
    """
    Compute the label and error rate.
    :param X: input data
    :param y: label
    :param nn: neural network class
    :return:
    labels: predicted labels
    error_rate: prediction error rate
    """
    res = np.zeros(np.shape(y))
    for i in range(np.shape(X)[0]):
        y_pred = forward(X[i], nn)
        idx = np.argmax(y_pred)
        res[i] = idx
    count = (res == y).sum()
    return res, (np.shape(y)[0] - count)/(np.shape(y)[0])

def train(X_tr, y_tr, X_te, y_te, nn):
    """
    Train the network using SGD for some epochs.
    :param X_tr: train data
    :param y_tr: train label
    :param X_te: test data
    :param y_te: test label
    :param nn: neural network class
    """
    J_train = []
    J_val = []
    for e in range(nn.n_epoch):
        X_new, y_new = shuffle(X_tr, y_tr, e)
        for i in range(np.shape(X_new)[0]):
            ob = forward(X_new[i], nn)
            g_alpha, g_beta = backward(X_new[i], y_new[i], ob, nn)
            nn.grad_sum_w1 = nn.grad_sum_w1 + np.square(g_alpha)
            nn.grad_sum_w2 = nn.grad_sum_w2 + np.square(g_beta)
            inter_1 = np.divide(nn.lr, np.sqrt(np.add(nn.grad_sum_w1, nn.epsilon)))
            inter_2 = np.divide(nn.lr, np.sqrt(np.add(nn.grad_sum_w2, nn.epsilon)))
            nn.alpha = nn.alpha - np.multiply(inter_1, g_alpha)
            nn.beta = nn.beta - np.multiply(inter_2, g_beta)

            #val = val + np.log(ob)[y_new[i]]
        #calculate J
        val_1, val_2 = 0, 0
        for j in range(np.shape(X_new)[0]):
            ob = forward(X_new[j], nn)
            val_1 = val_1 + np.log(ob)[y_new[j]]
        J_tr = -1/(np.shape(X_new)[0]) * val_1
        J_train.append((e, J_tr))

        for h in range(np.shape(X_te)[0]):
            ob = forward(X_te[h], nn)
            val_2 = val_2 + np.log(ob)[y_te[h]]
        J_te = -1/(np.shape(X_te)[0]) * val_2
        J_val.append((e, J_te))
    return J_train, J_val


if __name__ == "__main__":

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    # Note: You can access arguments like learning rate with args.learning_rate

    # initialize training / test data and labels
    (X_tr, y_tr, X_te, y_te, out_tr, 
        out_te, out_metrics, n_epochs, 
        n_hid, init_flag, lr) = args2data(args)
    # Build model
    if init_flag == 1:
        weight_init_fn = random_init
    elif init_flag == 2:
        weight_init_fn = zero_init
    #my_nn_1 = NN(lr, n_epochs, weight_init_fn, len(X_tr[0]), 5, 10)
    #my_nn_2 = NN(lr, n_epochs, weight_init_fn, len(X_tr[0]), 20, 10)
    my_nn = NN(lr, n_epochs, weight_init_fn, len(X_tr[0]), n_hid, 10)
    #my_nn_4 = NN(lr, n_epochs, weight_init_fn, len(X_tr[0]), 100, 10)
    #my_nn_5 = NN(lr, n_epochs, weight_init_fn, len(X_tr[0]), 200, 10)

    # train model
    #J_train1, J_val1 = train(X_tr, y_tr, X_te, y_te, my_nn_1)
    #J_train2, J_val2 = train(X_tr, y_tr, X_te, y_te, my_nn_2)
    J_train, J_val = train(X_tr, y_tr, X_te, y_te, my_nn)
    #J_train4, J_val4 = train(X_tr, y_tr, X_te, y_te, my_nn_4)
    #J_train5, J_val5 = train(X_tr, y_tr, X_te, y_te, my_nn_5)
    print(my_nn.alpha)
    print(my_nn.beta)


    with open(out_tr, 'w') as f1:
        f1.write(str(my_nn.alpha))

    
    with open(out_te, 'w') as f2:
        f2.write(str(my_nn.beta))

    '''
    x = [5, 20, 50, 100, 200]
    y_1 = []
    y_2 = []
    y_1.append(J_train1[-1][1])
    y_1.append(J_train2[-1][1])
    y_1.append(J_train3[-1][1])
    y_1.append(J_train4[-1][1])
    y_1.append(J_train5[-1][1])

    y_2.append(J_val1[-1][1])
    y_2.append(J_val2[-1][1])
    y_2.append(J_val3[-1][1])
    y_2.append(J_val4[-1][1])
    y_2.append(J_val5[-1][1])

    plt.plot(x, y_1, label = "avg train cross entropy loss")
    plt.plot(x, y_2, label = "avg validation cross entropy loss")
    plt.xlabel('hidden units')
    plt.ylabel('avg train and validation cross entropy loss')
    plt.title('avg train and validation cross entropy loss v. hidden units')
    plt.legend()
    plt.show()
    '''

    '''

    x_1, y_1 = [J_train1[i][0] for i in range(len(J_train1))], [J_train1[i][1] for i in range(len(J_train1))]
    plt.plot(x_1, y_1, label = "train: h_unit = 5")
    x_2, y_2 = [J_train2[i][0] for i in range(len(J_train2))], [J_train2[i][1] for i in range(len(J_train2))]
    plt.plot(x_2, y_2, label = "train: h_unit = 20")
    x_3, y_3 = [J_train3[i][0] for i in range(len(J_train3))], [J_train3[i][1] for i in range(len(J_train3))]
    plt.plot(x_3, y_3, label = "train: h_unit = 50")
    x_4, y_4 = [J_train4[i][0] for i in range(len(J_train4))], [J_train4[i][1] for i in range(len(J_train4))]
    plt.plot(x_4, y_4, label = "train: h_unit = 100")
    x_5, y_5 = [J_train5[i][0] for i in range(len(J_train5))], [J_train5[i][1] for i in range(len(J_train5))]
    plt.plot(x_5, y_5, label = "train: h_unit = 200")


    x_1b, y_1b = [J_val1[i][0] for i in range(len(J_val1))], [J_val1[i][1] for i in range(len(J_val1))]
    plt.plot(x_1b, y_1b, label = "validation: h_unit = 5")
    x_2b, y_2b = [J_val2[i][0] for i in range(len(J_val2))], [J_val2[i][1] for i in range(len(J_val2))]
    plt.plot(x_2b, y_2b, label = "validation: h_unit = 20")
    x_3b, y_3b = [J_val3[i][0] for i in range(len(J_val3))], [J_val3[i][1] for i in range(len(J_val3))]
    plt.plot(x_3b, y_3b, label = "validation: h_unit = 50")
    x_4b, y_4b = [J_val4[i][0] for i in range(len(J_val4))], [J_val4[i][1] for i in range(len(J_val4))]
    plt.plot(x_4b, y_4b, label = "validation: h_unit = 100")
    x_5b, y_5b = [J_val5[i][0] for i in range(len(J_val5))], [J_val5[i][1] for i in range(len(J_val5))]
    plt.plot(x_5b, y_5b, label = "validation: h_unit = 200")

    plt.xlabel('epoch')
    plt.ylabel('avg train and validation cross entropy loss')
    plt.title('epoch v. avg train and validation cross entropy loss')
    plt.legend()
    plt.show()
    # test model and get predicted labels and errors

    y_tr_pred, err_tr = test(X_tr, y_tr, my_nn)
    y_te_pred, err_te = test(X_te, y_te, my_nn)
    # write predicted label and error into file
    
    with open(out_tr, 'w') as f1:
        f1.write(str(y_tr_pred[0]))
        for i in range(1, len(y_tr_pred)):
            f1.write('\n' + str(y_tr_pred[i]))

    with open(out_te, 'w') as f2:
        f2.write(str(y_te_pred[0]))
        for i in range(1, len(y_te_pred)):
            f2.write('\n' + str(y_te_pred[i]))

    with open(out_metrics, 'w') as f:
        for i in range(len(J_train)):
            f.write('epoch=' + str(J_train[i][0]) + ' crossentropy(train): ' 
                    + str(J_train[i][1]) + '\n')
            f.write('epoch=' + str(J_val[i][0]) + ' crossentropy(validation): ' 
                    + str(J_val[i][1]) + '\n')
        f.write('error(train): ' + str(err_tr) + '\n')    
        f.write('error(validation): ' + str(err_te))
    '''
#python plot.py small_train_data.csv small_validation_data.csv small_train_out.labels small_validation_out.labels small_metrics_out.txt 100 4 1 0.1

#python plot.py small_train_data.csv small_validation_data.csv small_train_out.labels small_validation_out.labels small_metrics_out.txt 1 4 2 0.1