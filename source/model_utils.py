import tensorflow as tf
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import shutil
import string
import os
import re


# Define globals:
max_features = 10000
sequence_length = 250
embedding_dim = 16


def custom_standardization(input_data):
	"""Standarizes input data by removing spaces and characters."""

	lowercase = tf.strings.lower(input_data)
	stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
	
	return tf.strings.regex_replace(
		stripped_html,
		'[%s]' % re.escape(string.punctuation),
		''
	)

def vectorizer():
	"""Converts text into a features vector."""

	vectorize_layer = TextVectorization(
		standardize=custom_standardization,
		max_tokens=max_features,
		output_mode='int',
		output_sequence_length=sequence_length
	)

	return vectorize_layer

def vec_model():
	"""Creates a sequential model for feature vectors."""

	model = tf.keras.Sequential([
		layers.Embedding(max_features + 1, embedding_dim),
		layers.Dropout(0.2),
		layers.GlobalAveragePooling1D(),
		layers.Dropout(0.2),
		layers.Dense(1)
	])

	model.compile(
		loss=losses.BinaryCrossentropy(from_logits=True),
		optimizer='adam',
		metrics=tf.metrics.BinaryAccuracy(threshold=0.0)
	)

	return model


def full_model():
	"""Pastes the different processes to have a end-to-end model."""

	model = tf.keras.Sequential([
		vectorizer(),
		vec_model(),
		layers.Activation('sigmoid')
	])

	model.compile(
		loss=losses.BinaryCrossentropy(from_logits=False),
		optimizer="adam",
		metrics=['accuracy']
	)

	return model

def load_model(checkpoint_path):
	"""Loads weights into a full model."""

	model = full_model()
	model.load_weights(checkpoint_path)

	return model
