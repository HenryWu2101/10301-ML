import numpy as np
import sys

def read_data(input):
    #load data
    data = np.genfromtxt(input, delimiter="\t", dtype=None, encoding=None)
    #strip the title
    data = data[1:]
    #get the last col as label
    x_train = data[:, :-1]
    y_train = data[:, -1]
    return x_train, y_train

def train(x_train, y_train):
    #calculate total
    total = y_train.size
    #calculate one of them
    num = 0
    label_1 = y_train[0]
    label_2 = ''
    for item in y_train:
        if item == label_1:
            num += 1
        else: 
            label_2 = item
    if 2*num > total: 
        model = label_1
    elif 2*num == total:
        model = min(label_1, label_2)
    else:
        model = label_2
    return model

def predict(model, x_train):
    length = len(x_train)
    output = [model for i in range(length)]
    return output

def error(y_train, y_pred):
    num = 0
    for i in range(len(y_train)):
        if y_train[i] != y_pred[i]:
            num += 1
    return num/(len(y_train))

if __name__ == '__main__':
    trset = sys.argv[1]
    teset = sys.argv[2]
    x_train, y_train = read_data(trset)
    x_test, y_test = read_data(teset)

    model_tr = train(x_train, y_train)
    model_te = train(x_test, y_test)

    y_pred_tr = predict(model_tr, x_train)
    y_pred_te = predict(model_te, x_test)

    train_error_tr = str(error(y_train, y_pred_tr))
    train_error_te = str(error(y_test, y_pred_te))

    #pred, train
    with open(sys.argv[3], 'w') as tr_out:
        for line in y_pred_tr:
            tr_out.write(line + '\n')
    
    #pred, test
    with open(sys.argv[4], 'w') as te_out:
        for line in y_pred_te:
            te_out.write(line + '\n')

    #err
    with open(sys.argv[5], 'w') as f_out:
        f_out.write('error(train): ' + train_error_tr + '\n')
        f_out.write('error(test): ' + train_error_te + '\n')


#python majority_vote.py inputs/politicians_train.tsv inputs/politicians_test.tsv train.labels test.labels metrics.txt