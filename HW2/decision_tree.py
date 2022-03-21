import numpy as np
import math
import sys


class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self, attr, v, x, y):
        self.left = None
        self.right = None
        self.attr = attr    #use num as indices instead, so that instance[attr] just gives value
        self.vote = v       #either 0, 1, as a y_label
        self.x, self.y = x, y   #~at testing, need to store both x, y
        self.order = None
        self.own = None

        #self.depth = depth  #keep track of tree depth

    '''
    Note:
        The following 3 compute functions does not modify the node
        Only accessing the information stored within
    '''
    

    #get entropy at curr root node data "y"
    #output H(y)
    #this is training time function only
    def compute_H(self):
        x, y = self.x, self.y
        length = len(y) #should be = len(x)
        #different label names
        labels, bin = np.unique(y, return_inverse = True)
        N1 = np.bincount(bin)[0]
        N2 = length - N1  #note : they should sum to length
        P1 = N1/length
        P2 = N2/length
        log_1 = math.log2(P1) if P1 != 0 else 0
        log_2 = math.log2(P2) if P2 != 0 else 0
        H = -log_1 * P1 - log_2*P2
        return H


    #at a leaf node
    #get the majority vote on data stored
    def m_vote(self):
        #majority vote on y
        x, y = self.x, self.y
        u, p = np.unique(y, return_inverse=True)
        c = np.bincount(p) 
        if (len(c) == 0): 
            return str(sorted(u)[1])
        if c[0] > len(p) - c[0]:
            vote = str(u[0])
        elif c[0] < len(p) - c[0]:
            vote = str(u[1])
        else:
            vote = str(sorted(u)[1])
        return vote


    #get mutual infomation with regards to respective attr
    #output I(node;x), attr(as the name of optimal attr)
    def compute_I(self):
        x, y = self.x, self.y
        info = 0
        a = -1
        H_y = self.compute_H()
        for attr in range(len(x[0])):
            (x1, y1), (x2, y2), u = sub(x, y, attr)
            curr_i = H_y - c_H(x1,y1)*len(x1)/len(x) - c_H(x2, y2)*len(x2)/len(x)
            if curr_i > info: 
                info = curr_i
                a = attr 
        return info, a   




#gives dataset d_L = Y|attr = 0, d_R = Y|attr = 1
def sub(x, y, attr):
    x1 = np.copy(x)
    y1 = np.copy(y)
    u, pos = np.unique(x,return_inverse=True)
    R = np.where(x1[:, attr] == u[1])
    L = np.where(x1[:, attr] == u[0])
    d_L = x1[L], y1[L]
    d_R = x1[R], y1[R]
    return d_L, d_R, u

#also compute H
def c_H(x, y):
    length = len(y) #should be = len(x)
    #different label names
    labels, bin = np.unique(y, return_inverse = True)
    if (len(bin) == 0) : return 0
    N1 = np.bincount(bin)[0]
    N2 = length - N1    #note : they should sum to length
    P1 = N1/length
    P2 = N2/length
    log_1 = math.log2(P1) if P1 != 0 else 0
    log_2 = math.log2(P2) if P2 != 0 else 0
    H = -log_1 * P1 - log_2*P2
    return H

#read stat from command line
#generates np array of data
#init a root node for DT, with data stored
#output is the root node, and x, y as data
def read_data(input):
    #load data into np from input
    data = np.genfromtxt(input, delimiter="\t", dtype=None, encoding=None)
    #transform into useful stuff 
    data = data[1:]
    x_attr = data[:, :-1]
    y_label = data[:, -1]
    d = x_attr, y_label
    #None will be replaced by actual values after prediction
    return d

#grow the tree obviously
def train(x, y, max_diff, own = None):
    #base case
    #if pure
    p1 = np.all(y == y[0])
    #if empty data set
    p2 = len(y) == 0
    #if max depth
    p3 = max_diff <= 0
    #if one of em correct
    if p1|p2|p3:
        node = Node(None, None, x, y)
        node.vote = node.m_vote()
        node.own = own
        return node
    #now we not facing empty data set
    #deal with the last case
    #if for each attr, all val same
    if np.all(x == x[0]):
        node = Node(None, None, x, y)
        node.vote = node.m_vote()
        node.own = own
        return node
    #as in internal node   
    node = Node(None, None, x, y)
    node.own = own
    info, attr = node.compute_I()
    if info <= 0:
        node.vote = node.m_vote()
        return node
    #split on attr
    node.attr = attr
    #now subset data
    (x1, y1), (x2, y2), u = sub(x, y, attr)
    node.order = u
    node.left = train(x1, y1, max_diff-1, attr)
    node.right = train(x2, y2, max_diff-1, attr)
    return node
    
        
def T_print(tree, x_names, d = 0):
    if tree.left == None & tree.right == None:
        #leaf
        print('|' * d)
        if (tree.own != None):
            print(x_names[tree.own] + " = " + tree.x[tree.own] + ": ")
        print("["
    return 





#recursively, predict the vote of the DT with test data
#node is from training time, data is from command line
def predict(node, data):
    if (node.vote != None):
        return node.vote
    else:
        val = data[node.attr]
        if (val == node.order[0]):
            return predict(node.left, data)
        else:
            return predict(node.right, data)


def error(y_train, y_pred):
    num = 0
    for i in range(len(y_train)):
        if y_train[i] != y_pred[i]:
            num += 1
    return num/(len(y_train))

if __name__ == '__main__':
    input_train = sys.argv[1]
    input_test = sys.argv[2]
    max = int(sys.argv[3])
    out_train = sys.argv[4]
    out_test = sys.argv[5]
    m_out = sys.argv[6]

    x, y = read_data(input_train)
    tree = train(x, y, max)
    v = np.array([predict(tree, line) for line in x])

    x_t, y_t = read_data(input_test)
    v_t = np.array([predict(tree, line) for line in x_t])

    e_train = error(y, v)
    e_test = error(y_t, v_t)

    with open(out_train, 'w') as tr_out:
        for line in v:
            tr_out.write(line + '\n')
    
    with open(out_test, 'w') as te_out:
        for line in v_t:
            te_out.write(line + '\n')
    
    with open(m_out, 'w') as f:
        f.write('error(train): ' + str(e_train) + '\n')    
        f.write('error(test): ' + str(e_test))
#python decision_tree.py politicians_train.tsv politicians_test.tsv 7 pol_7_train.labels pol_7_test.labels pol_7_metrics.txt
#python decision_tree.py small_train.tsv small_test.tsv 3 sml_3_train.labels sml_3_test.labels sml_3_metrics.txt
#python decision_tree.py education_train.tsv education_test.tsv 4 edu_4_train.labels edu_4_test.labels edu_4_metrics.txt


