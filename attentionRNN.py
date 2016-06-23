#!/usr/bin/python

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import collections
from dataReader import *

def rnn_output(x):
    # Forward direction cell
    gru_fw_cell = rnn_cell.GRUCell( n_hidden, forget_bias=1.0)
    # Backward direction cell
    gru_bw_cell = rnn_cell.GRUCell( n_hidden, forget_bias=1.0)
    try:
        outputs, _, _ = rnn.bidirectional_rnn( gru_fw_cell, gru_bw_cell, x, dtype=tf.float32)
    except Exception:
        outputs = rnn.bidirectional_rnn( gru_fw_cell, gru_bw_cell, x, dtype=tf.float32)	
    return outputs

def get_pred(probs):
    # Initialize probabilities tensor
    and_prob_list = [0]*n_words
    for i in get_answer_indices(answer):
        and_prob_list[i] = 1
    eliminate_prob = tf.Variable(np.asarray(and_prob_list))
    pred = tf.reduce_sum( tf.mul(eliminate_prob,probs) )
    return pred

##################
# x_doc: placeholder for document word list
# x_query: placeholder for query word list
# sent: sentence word list
# answer: answer option
##################

def attention_model(x_doc,x_query):
    X_Doc = rnn_output( x_doc )
    X_Query = rnn_output( x_query )
    probs = tf.nn.softmax( tf.mul( w, X_Query[-1], name='dot_products'), 'softmax probabilities' )
    pred = get_pred(probs)
    return pred

#################
## Model Setup ##
#################

# Resources
filename = 'data/test'

# Parameters
learning_rate = 0.01
training_iters = 100000
batch_size = 1
display_step = 10

# Network Parameters
n_input = 300
n_steps = 10
n_hidden = 1000
n_prob = 1

# tf Graph input
n_doc_words = tf.Variable( tf.int32, [1] )
x_doc = tf.placeholder( tf.float32, [None,n_input] )
x_query = tf.placeholder( tf.float32, [None,n_input] )
y = tf.placeholder( tf.float32, [None,n_prob] )

#################
## Build Model ##
#################

pred = attention_model(x_doc,x_query)
cost = tf.pow(pred-y, 2)
optimizer = tf.train.GradientDescentOptimizer( learning_rate ).minimize(cost)

#################
## Train Model ##
#################

init = tf.initialize_all_variables()
with tf.Session as sess:
    sess.run(init)
	for doc,question,options in next_batch(filename):
		options = options.split(',')
		for option in options:
			# Get new batch
			doc_list = doc.split()
		    qa_pair = question.replace('*_*',option)



