import pickle

# dataset = "sample"
dataset = "yoochoose1_64"

train_data = pickle.load(open(dataset + '/train.txt', 'rb'))
test_data = pickle.load(open(dataset + '/test.txt', 'rb'))
(tr_seqs, tr_labs) = train_data

print(len(tr_seqs), len(tr_labs))
for i in range(len(tr_seqs)):
    print(tr_seqs[i], tr_labs[i])
    if i > 10:
        sadf