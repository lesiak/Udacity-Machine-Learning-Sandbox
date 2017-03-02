from SentimentNetwork import SentimentNetwork

g = open('reviews.txt', 'r')  # What we know!
reviews = list(map(lambda x: x[:-1], g.readlines()))
g.close()

g = open('labels.txt', 'r')  # What we WANT to know!
labels = list(map(lambda x: x[:-1].upper(), g.readlines()))
g.close()

# Project 3: Building a Neural Network
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=20,polarity_cutoff=0.1,learning_rate=0.01)
# evaluate our model before training (just to show how horrible it is)
# mlp.test(reviews[-1000:], labels[-1000:])

# train the network
mlp.train(reviews[:-1000], labels[:-1000])

mlp.test(reviews[-1000:], labels[-1000:])
