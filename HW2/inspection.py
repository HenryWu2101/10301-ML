import numpy as np
import sys
import math

def inspect(input):
    #load data into np from input
    data = np.genfromtxt(input, delimiter="\t", dtype=None, encoding=None)
    #transform into useful stuff
    data = data[1:]
    x_attr = data[:, :-1]
    y_label = data[:, -1]
    return x_attr, y_label

def compute(x, y):  #actually don't need x in this case
    length = len(y) #should be = len(x)
    #different label names
    #x attr as binary 
    labels, bin = np.unique(y, return_inverse = True)
    N1 = np.bincount(bin)[0]
    N2 = length - N1    #note : they should sum to length
    P1 = N1/length
    P2 = N2/length
    log_1 = math.log2(P1) if P1 != 0 else 0
    log_2 = math.log2(P2) if P2 != 0 else 0
    H = -log_1 * P1 - log_2*P2
    major = N1 if N1 >= N2 else N2
    err = (length - major)/length
    return H, err


if __name__ == '__main__':
    input = sys.argv[1]
    output = sys.argv[2]
    x, y = inspect(input)
    H, err = compute(x, y)

    with open(output, 'w') as f:
        f.write('entropy: ' + str(H) + '\n')
        f.write('error: ' + str(err))


#Test command lines:
#python inspection.py small_train.tsv small_inspect.txt
#python inspection.py politicians_train.tsv politicians_inspect.txt
#python inspection.py education_train.tsv education_inspect.txt