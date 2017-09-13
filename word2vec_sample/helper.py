from os.path import isfile, abspath
import numpy as np
import tensorflow as tf
import tqdm
from urllib.request import urlretrieve
import zipfile 
import pickle
from collections import Counter

## Absolute path to the target folder ##
filepath = abspath('./')
pickle_data_file = filepath + '/data.pickle'


## Downloading the files
# Zip file #
class DLProgress(tqdm.tqdm):

	last_block = 0

	def hook(self, block_num=1, block_size=1, total_size=None):
		self.total = total_size
		self.update((block_num - self.last_block) * block_size)
		self.last_block = block_num

if not isfile(filepath + '/Data.zip'):
	with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Raw Data') as pbar:
		urlretrieve("http://mattmahoney.net/dc/text8.zip",
					"Data.zip",
					pbar.hook)

# Unzipping the file
def preparing_data(filepath):
	if not isfile(pickle_data_file):
		zf = zipfile.ZipFile("Data.zip", "r")
		extract = zf.namelist()
		for filename in extract:
			try:
				byte_data = zf.read(filename)
			except KeyError:
				print("ERROR: Did not find %s in zipfile" % (filename))

		# Converting bytes to string
		data = tf.compat.as_str(byte_data).split()

		# Pickle data for future use
		pickle_data = open(pickle_data_file, "wb")
		pickle.dump(data,pickle_data)
		pickle_data.close()
	else:
		print("Using Pickled data")



# Building the dataset
def build_data(raw_data, vocab_size):

	# Making a label for rare words
	count = [['UNK', -1]]

	# Making a Vocab 
	count.extend(Counter(raw_data).most_common(vocab_size-1))
	
	# Allocating an index to each word in vocab
	vocab_dict = {}
	for word, _ in count:
		vocab_dict[word] = len(vocab_dict)

	# Making a list of indexes
	data = []
	unk_count = 0
	for word in raw_data:
		if word in vocab_dict:
			index = vocab_dict[word]
		else:
			index = 0
			unk_count +=1
		data += [index]

	# Making an integer to word converter
	reversed_vocab = dict(zip(vocab_dict.values(),vocab_dict.keys()))

	return data, count, vocab_dict, reversed_vocab


def build_input_shape(size):

	return tf.placeholder(tf.int32, shape=[size])

def build_output_shape(size):

	return tf.placeholder(tf.int32, shape=[size,1])

def build_valid_dataset(valid_examples):

	return tf.constant(value=valid_examples, dtype=tf.int32)
