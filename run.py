from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
import pickle

import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

filename = maybe_download('text8.zip', 31344016)

# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

vocabulary = read_data(filename)
print('Data size', len(vocabulary))

# Step 2 : Build the  dictionary and replace rare words with UNK token.
vocabulary_size = 50000

def maybe_build_dataset(words, n_words):
    """Process raw inputs into a dataset"""
    # Get generated file if it existed
    if os.path.exists('data.pkl'):
        pkl_file = open('data.pkl','rb')
        data = pickle.load(pkl_file)
        words2index = data['words2index']
        topN_list = data['topN_list']
        word_index_dict = data['word_index_dict']
        index_word_dict = data['index_word_dict']
        pkl_file.close()
    else:
        # Get most frequent words from corpus
        topN_list = [['UNK', -1]]
        topN_list.extend(collections.Counter(words).most_common(n_words - 1))

        # But a [word, index] dictionary which is based on topN_list, and the index is based on the frequency
        word_index_dict = dict()
        for word, _ in topN_list:
            word_index_dict[word] = len(word_index_dict)

        # Convert corpus to indexes which are based on word_index_dict
        words2index = list()
        unk_count = 0
        for word in words:
            if word in word_index_dict:
                index = word_index_dict[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            words2index.append(index)

        # Other rare words are regarded UNK
        topN_list[0][1] = unk_count
        index_word_dict = dict(zip(word_index_dict.values(), word_index_dict.keys()))

        # Save variables to disk
        data = {'words2index': words2index, 'topN_list': topN_list, 'word_index_dict': word_index_dict,
                'index_word_dict': index_word_dict}
        pkl_file = open('data.pkl', 'wb')
        pickle.dump(data, pkl_file, -1)
        pkl_file.close()

    return words2index, topN_list, word_index_dict, index_word_dict

words2index, topN_list, word_index_dict, index_word_dict = maybe_build_dataset(vocabulary, vocabulary_size)

del vocabulary # Hint to reduce memory
print('Most common words (+UNK)', topN_list[:5])
print('Sample data', words2index[:10], [index_word_dict[i] for i in words2index[:10]])

data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    """
    :param batch_size: The size of batch
    :param num_skips: Take how many words in the window
    :param skip_window: The size of one side of window
    :return:
    """
    global data_index  # The pointer to point to corpus
    assert batch_size % num_skips == 0 # batch_size should be n times of num_skips
    assert num_skips <= 2 * skip_window #The number of words to get must be less to the words in the window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    window = 2 * skip_window + 1  # [ skip_window target skip_window ]

    # Put word indexs into the buffer with the window size
    buffer = collections.deque(maxlen=window)
    for _ in range(window):
        buffer.append(words2index[data_index])
        data_index = (data_index + 1) % len(words2index)

    # Get num_skips of words from buffer
    for i in range(batch_size // num_skips):
        target = skip_window  # Middle word of the window
        context = skip_window # Content words to get
        context_to_avoid = [skip_window] # Avoid target and exiting words
        for j in range(num_skips):
            while context in context_to_avoid:
                context = random.randint(0, window - 1)
            context_to_avoid.append(context)
            batch[i * num_skips + j] = buffer[target] # Store target words
            labels[i * num_skips + j, 0] = buffer[context] # Store context word
        # Update buffer, the maxlen of buffer is equal to the window size
        buffer.append(words2index[data_index])
        data_index = (data_index + 1) % len(words2index)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(words2index) - window) % len(words2index)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], index_word_dict[batch[i]],
          '->', labels[i, 0], index_word_dict[labels[i, 0]])

# hyperpatameters
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
learning_rate = 1.0

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():
    # define variales
    with tf.name_scope('Train_Inputs'):
        train_inputs = tf.placeholder(tf.int32, batch_size, name='train_inputs')
        train_labels = tf.placeholder(tf.int32, (batch_size, 1), name='train_labels')

    with tf.name_scope('Variables'):
        # Define word embeddings, w and b
        embeddings = tf.Variable(
            tf.random_uniform((vocabulary_size, embedding_size), -1.0, 1.0), name='word_embedding')

        nce_W = tf.Variable(
            initial_value=tf.truncated_normal(
                shape=(vocabulary_size, embedding_size),
                stddev=1.0 / math.sqrt(embedding_size)), name='nce_W')

        nce_b = tf.Variable(tf.zeros(vocabulary_size), name='nce_b')

    # Look up the first dimension of 1st pram, which is based on train_inputs
    embeded = tf.nn.embedding_lookup(embeddings, train_inputs, name='Look_up')

    # http://www.usa-idc.com/news/idc/201700504.shtml
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_W,
                biases=nce_b,
                labels=train_labels,
                inputs=embeded,
                num_sampled=num_sampled,
                num_classes=vocabulary_size))
        tf.summary.scalar('loss', loss)


    with tf.name_scope('Valid_Inputs'):
        valid_inputs = tf.constant(valid_examples, tf.int32)

    with tf.name_scope('Normalizing'):
        # calculate column norm
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm

    # Look up embedding for valid_inputs
    valid_embeded = tf.nn.embedding_lookup(embeddings, valid_inputs, name='Look_Up')

    # The larger the more similar
    with tf.name_scope('Similarity'):
        similarity = tf.matmul(valid_embeded, normalized_embeddings, transpose_b=True)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    merged = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter("logs/", graph)

# Step 5: Begin training.

num_steps = 100001

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs:batch_inputs, train_labels:batch_labels}

        loss_result, _, _ = sess.run([loss, optimizer, merged],feed_dict=feed_dict)
        if step != 0:
            average_loss += loss_result

        if step % 500 == 0 and step != 0:
            average_loss /= 500
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0
        if step % 500 == 0 and step !=0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                # Get words from valid_examples
                valid_word = index_word_dict[valid_examples[i]]
                top_n = 8
                # Get the 8 most similar words, ignore the first one, since it is that word
                nearest = (-sim[i, :]).argsort()[1:top_n+1]
                log_str = 'Nearest to %s:' % valid_word
                for j in range(top_n):
                    close_word = index_word_dict[nearest[j]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    # get embeddings so we can know how similar thy are
    final_embeddings = normalized_embeddings.eval()

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x,y)
        plt.annotate(label,
                     xy=(x,y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)

try:
    # pylint: disable=g-import-not-at-top
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [index_word_dict[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels)
except ImportError:
    print('Please install sklearn, matplotlib, and scipy to show embeddings')
