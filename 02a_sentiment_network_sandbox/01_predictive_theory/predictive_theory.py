from collections import Counter
import numpy as np


def split_on_condition(seq, condition):
    a, b = [], []
    for item in seq:
        (a if condition(item) else b).append(item)
    return a, b


def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")


def remove_eol(x):
    return x[:-1]


g = open('reviews.txt', 'r')  # What we know!
reviews = list(map(lambda x: x[:-1], g.readlines()))
g.close()

g = open('labels.txt', 'r')  # What we WANT to know!
labels = list(map(lambda x: remove_eol(x).upper(), g.readlines()))
g.close()

data_pairs = list(zip(labels, reviews))
negative_data, pos_data = split_on_condition(data_pairs, lambda x: 'NEGATIVE' == x[0])

neg_with_terrible = list(filter(lambda x: 'terrible' in x[1], negative_data))
pos_with_terrible = list(filter(lambda x: 'terrible' in x[1], pos_data))
print(len(neg_with_terrible))
print(len(pos_with_terrible))

positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()

for i in range(len(reviews)):
    if labels[i] == 'POSITIVE':
        for word in reviews[i].split(" "):
            positive_counts[word] += 1
            total_counts[word] += 1
    else:
        for word in reviews[i].split(" "):
            negative_counts[word] += 1
            total_counts[word] += 1

print(positive_counts.most_common())

pos_neg_ratios = Counter()

for term, cnt in list(total_counts.most_common()):
    if cnt > 100:
        pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1)
        pos_neg_ratios[term] = pos_neg_ratio

'''
If you remember in your first project of Building Neural Network using Bicycle, before feeding the data into the neural network we normalize the data as follows;

quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std


Similarly in text analysis it is common practice to scale the data into logarithmic distribution so that's why instructor has used np.log

for word,ratio in pos_neg_ratios.most_common():
    if(ratio > 1):
        pos_neg_ratios[word] = np.log(ratio)
    else:
        pos_neg_ratios[word] = -np.log((1 / (ratio+0.01)))

He splits the if/else statement into ratio >1 and ratio <=1.
In the case when ratio>1, he does not have to worry that his ratio will ever be zero,
however for the case that ratio <=1 he has to worry about zero.

np.log(ratio) is same as -np.log(1/ratio).

-log (1/(x+0.01)) => -log (1/x) = log (x)
Thus, you can expect both two values will have almost the same (also adding 0.01 to exclude the case of x=0).


'''


for word, ratio in pos_neg_ratios.most_common():
    if ratio > 1:
        pos_neg_ratios[word] = np.log(ratio)
    else:
        pos_neg_ratios[word] = -np.log((1 / (ratio+0.01)))

print(pos_neg_ratios.most_common())

print(list(reversed(pos_neg_ratios.most_common()))[0:30])

total_counts_list = list(total_counts)