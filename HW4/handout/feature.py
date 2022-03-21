import csv
import numpy as np
import sys

VECTOR_LEN = 300   # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An N x 2 np.ndarray. N is the number of data points in the tsv file. The
        first column contains the label integer (0 or 1), and the second column
        contains the movie review string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_dictionary(file):
    """
    Creates a python dict from the model 1 dictionary.

    Parameters:
        file (str): File path to the dictionary for model 1.

    Returns:
        A dictionary indexed by strings, returning an integer index.
    """
    dict_map = np.loadtxt(file, comments=None, encoding='utf-8',
                          dtype=f'U{MAX_WORD_LEN},l')
    return {word: index for word, index in dict_map}


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file for model 2.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map


####################
#  helpers above   #
####################

def mod_1_vec(dic, datum):
    #datum of len 2 only
    rating = datum[0]
    raw = datum[1]
    #output vec with max len of len(dict)
    vec = np.zeros((len(dic), ), dtype = int)
    for w in raw.split():
        if w in dic:
            vec[dic[w]] = 1
    return rating, vec


def mod_1(dic, data):
    '''
    return a 2D array, each entry contains the label and corresponding array
    '''
    result = []
    for i in range(len(data)):
        datum = data[i]
        label, vec = mod_1_vec(dic, datum)
        result.append([label,vec])
    return result

def mod_2_vec(dic, datum):
    #datum of len 2 only
    count = 0
    rating = datum[0]
    raw = datum[1]
    sample = list(dic.values())[0]
    val = np.zeros(np.shape(sample))
    for w in raw.split():
        if w in dic:
            count += 1
            val += dic[w]
    return rating, np.divide(val, count)

def mod_2(f_dict, data):
    '''
    return a 2D array, each entry contains the label and corresponding vec val
    '''
    res = []
    for i in range(len(data)):
        datum = data[i]
        label, val = mod_2_vec(f_dict, datum)
        res.append([label, val])
    return res



if __name__ == '__main__':
    train_in = sys.argv[1]
    valid_in = sys.argv[2]
    test_in = sys.argv[3]

    dict_in = sys.argv[4]
    f_dict_in = sys.argv[5]

    train_out = sys.argv[6]
    valid_out = sys.argv[7]
    test_out = sys.argv[8]

    flag = sys.argv[9]
    #flag == 1 model 1; flag == 2 model 2

    data_tr = load_tsv_dataset(train_in)
    data_val = load_tsv_dataset(valid_in)
    data_te = load_tsv_dataset(test_in)

    dic = load_dictionary(dict_in)
    f_dict = load_feature_dictionary(f_dict_in)

    if int(flag) == 1:
        result_tr = mod_1(dic, data_tr)
        result_val = mod_1(dic, data_val)
        result_te = mod_1(dic, data_te)

        with open(train_out, 'w') as f_tr:
            for array in result_tr:
                f_tr.write(str(array[0]))    #label
                for entry in array[1]:
                    f_tr.write('\t' + str(entry))
                f_tr.write('\n')
        
        with open(valid_out, 'w') as f_va:
            for array in result_val:
                f_va.write(str(array[0]))    #label
                for entry in array[1]:
                    f_va.write('\t' + str(entry))
                f_va.write('\n')
        
        with open(test_out, 'w') as f_te:
            for array in result_te:
                f_te.write(str(array[0]))    #label
                for entry in array[1]:
                    f_te.write('\t' + str(entry))
                f_te.write('\n')
    else:
        result_tr = mod_2(f_dict, data_tr)
        result_val = mod_2(f_dict, data_val)
        result_te = mod_2(f_dict, data_te)

        with open(train_out, 'w') as f_tr:
            for array in result_tr:
                f_tr.write(str(array[0]))    #label
                for entry in array[1]:
                    f_tr.write('\t' + str(entry))
                f_tr.write('\n')

        with open(valid_out, 'w') as f_va:
            for array in result_val:
                f_va.write(str(array[0]))    #label
                for entry in array[1]:
                    f_va.write('\t' + str(entry))
                f_va.write('\n')
        
        with open(test_out, 'w') as f_te:
            for array in result_te:
                f_te.write(str(array[0]))    #label
                for entry in array[1]:
                    f_te.write('\t' + str(entry))
                f_te.write('\n')



#python feature.py largedata/train_data.tsv largedata/valid_data.tsv largedata/test_data.tsv dict.txt word2vec.txt largeoutput/formatted_train.tsv largeoutput/formatted_valid.tsv largeoutput/formatted_test.tsv 1