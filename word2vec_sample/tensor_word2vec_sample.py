import helper
from os.path import isfile, isdir, abspath
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import tqdm
import pickle
from collections import deque
import random

filepath = abspath('./')
pickle_data_file = filepath + '/data.pickle'


## Loading data
helper.preparing_data(filepath)
with open(pickle_data_file, "rb") as f:
	data = pickle.load(f)
print(data[:10], len(data))

# Vocab Size
vocab_size = 50000

processed_data, count, vocab_dict, reversed_vocab = helper.build_data(data,vocab_size)
del data
print("Most Common words (+UNK)", count[:5])
print("Sample data", processed_data[:10], [reversed_vocab[i] for i in processed_data[:10]])

data_index = 0

# Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = deque(maxlen=span)
  if data_index + span > len(processed_data):
    data_index = 0
  buffer.extend(processed_data[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    if data_index == len(processed_data):
      buffer[:] = processed_data[:span]
      data_index = span
    else:
      buffer.append(processed_data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(processed_data) - span) % len(processed_data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reversed_vocab[batch[i]],
        '->', labels[i, 0], reversed_vocab[labels[i, 0]])

### Building the skip-gram graph ###
## Graph Hyperparameters
batch_size = 128 # Size of 1 batch
embedding_size = 128 # size of the word vector
num_skips = 2  # How many times to resue the input to generate a label
skip_window = 1 # How many words to consider left and right of the target


# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.
learning_rate = 1.0

def build_graph(valid_examples, batch_size, embedding_size, num_skips, skip_window,num_sampled):

  ## Inputs and Outputs ##
  train_inputs = helper.build_input_shape(batch_size)
  train_labels = helper.build_output_shape(batch_size)
  valid_dataset = helper.build_valid_dataset(valid_examples)

  embeddings = tf.Variable(tf.random_normal([vocab_size, embedding_size], -1.0, 1.0))
  embeds = tf.nn.embedding_lookup(embeddings, train_inputs)

  ## NCE weight and bias
  nce_weights = tf.Variable(tf.truncated_normal(mean=0,stddev=0.1,shape=[vocab_size, embedding_size]))
  nce_biases = tf.Variable(tf.zeros(shape=[vocab_size]))

  ## Computing loss ##
  loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                      biases=nce_biases,
                                      labels=train_labels,
                                      inputs=embeds,
                                      num_sampled=num_sampled,
                                      num_classes=vocab_size))

  ## Optimizer ##
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

  ## Computing cosine similarity between mismatches and all embeddings
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  return (train_inputs,train_labels), valid_dataset, similarity, normalized_embeddings, loss, optimizer

(train_inputs,train_labels), valid_dataset, similarity, normalized_embeddings, loss, optimizer = build_graph(valid_examples, batch_size,embedding_size,num_skips,skip_window,num_sampled)


### Begin Training
num_steps = 100001

with tf.Session() as sess:

  sess.run(tf.global_variables_initializer())
  print('Initialized')

  average_loss = 0
  for step in tqdm.tqdm(range(num_steps)):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in range(valid_size):
        valid_word = reversed_vocab[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in range(top_k):
          close_word = reversed_vocab[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()


## Visualising Results ##


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)
  plt.show()

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reversed_vocab[i] for i in range(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')