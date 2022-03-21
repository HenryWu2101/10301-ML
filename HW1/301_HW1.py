import numpy as np
import sys

train_infile = sys.argv[1]
test_infile = sys.argv[2]

train_outfile = sys.argv[3]
test_outfile = sys.argv[4]
metrics_outfile = sys.argv[5]




train_data = np.genfromtxt(train_infile, delimiter="\t", dtype=str)
train_data = train_data[1:]
train_data = train_data[:,-1]


test_data = np.genfromtxt(test_infile, delimiter="\t", dtype=str)
test_data = test_data[1:]
test_data = test_data[:,-1]


#Convert the infile into single column data

train_total_count = train_data.shape[0]
test_total_count = test_data.shape[0]

# Get the most frequent label in training set

'''unique, pos = np.unique(train_data,return_inverse=True)
maps=dict()
for elem in unique:

'''

unique, pos = np.unique(train_data,return_inverse=True)
counts = np.bincount(pos) 


if counts[0]>counts[1]:
    most = str(unique[0])
    
elif counts[0]<counts[1]:
    most = str(unique[1])

else:
    most = str(sorted(unique)[1])




error_train = 0
error_test = 0


#Get the wrong predictions for training and testing sets


for elem in train_data:
    if elem!=most:
        error_train+=1

for elem in test_data:
    if elem!=most:
        error_test+=1


# Write three out files
with open(train_outfile, 'w') as train_outfile:
    train_outfile.write(most)
    for i in range(len(train_data)-1):
        train_outfile.write('\n')
        train_outfile.write(most)
        


with open(test_outfile, 'w') as test_outfile:
    test_outfile.write(most)
    for i in range(len(test_data)-1):
        test_outfile.write('\n')
        test_outfile.write(most)
        



with open(metrics_outfile, 'w') as metrics_out:
    metrics_out.write("error(train): ")
    metrics_out.write(str(float(error_train)/float(train_total_count)))
    metrics_out.write('\n')
    metrics_out.write("error(test): ")
    metrics_out.write(str(float(error_test)/float(test_total_count)))