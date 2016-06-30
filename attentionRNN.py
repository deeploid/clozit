import theano
import theano.tensor as T
import theano.tensor.nnet as TN
import random
import numpy as np
from itertools import izip
from theano.gradient import grad_clip
import time
import operator
from gruCell import *

class attentionRNN(object):
    """docstring for attentionRNN"""
    def __init__(self, word_dim, hidden_dim=128, bptt_truncate=-1):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        ##############
        # doc params # 
        ##############
        # Initialize the network parameters
        E_doc = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        U_doc = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
        W_doc = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
        V_doc = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        b_doc = np.zeros((6, hidden_dim))
        c_doc = np.zeros(word_dim)
        # Theano: Created shared variables
        self.E_doc = theano.shared(name='E_doc', value=E_doc.astype(theano.config.floatX))
        self.U_doc = theano.shared(name='U_doc', value=U_doc.astype(theano.config.floatX))
        self.W_doc = theano.shared(name='W_doc', value=W_doc.astype(theano.config.floatX))
        self.V_doc = theano.shared(name='V_doc', value=V_doc.astype(theano.config.floatX))
        self.b_doc = theano.shared(name='b_doc', value=b_doc.astype(theano.config.floatX))
        self.c_doc = theano.shared(name='c_doc', value=c_doc.astype(theano.config.floatX))
        # SGD / rmsprop: Initialize parameters
        self.mE_doc = theano.shared(name='mE_doc', value=np.zeros(E_doc.shape).astype(theano.config.floatX))
        self.mU_doc = theano.shared(name='mU_doc', value=np.zeros(U_doc.shape).astype(theano.config.floatX))
        self.mV_doc = theano.shared(name='mV_doc', value=np.zeros(V_doc.shape).astype(theano.config.floatX))
        self.mW_doc = theano.shared(name='mW_doc', value=np.zeros(W_doc.shape).astype(theano.config.floatX))
        self.mb_doc = theano.shared(name='mb_doc', value=np.zeros(b_doc.shape).astype(theano.config.floatX))
        self.mc_doc = theano.shared(name='mc_doc', value=np.zeros(c_doc.shape).astype(theano.config.floatX))
        
        ###################
        # question params # 
        ###################

        # Initialize the network parameters
        E_qn = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        U_qn = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
        W_qn = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
        V_qn = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        b_qn = np.zeros((6, hidden_dim))
        c_qn = np.zeros(word_dim)
        # Theano: Created shared variables
        self.E_qn = theano.shared(name='E_qn', value=E_qn.astype(theano.config.floatX))
        self.U_qn = theano.shared(name='U_qn', value=U_qn.astype(theano.config.floatX))
        self.W_qn = theano.shared(name='W_qn', value=W_qn.astype(theano.config.floatX))
        self.V_qn = theano.shared(name='V_qn', value=V_qn.astype(theano.config.floatX))
        self.b_qn = theano.shared(name='b_qn', value=b_qn.astype(theano.config.floatX))
        self.c_qn = theano.shared(name='c_qn', value=c_qn.astype(theano.config.floatX))
        # SGD / rmsprop: Initialize parameters
        self.mE_qn = theano.shared(name='mE_qn', value=np.zeros(E_qn.shape).astype(theano.config.floatX))
        self.mU_qn = theano.shared(name='mU_qn', value=np.zeros(U_qn.shape).astype(theano.config.floatX))
        self.mV_qn = theano.shared(name='mV_qn', value=np.zeros(V_qn.shape).astype(theano.config.floatX))
        self.mW_qn = theano.shared(name='mW_qn', value=np.zeros(W_qn.shape).astype(theano.config.floatX))
        self.mb_qn = theano.shared(name='mb_qn', value=np.zeros(b_qn.shape).astype(theano.config.floatX))
        self.mc_qn = theano.shared(name='mc_qn', value=np.zeros(c_qn.shape).astype(theano.config.floatX))

        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        x = T.fvector('x')
        y_hat = T.fvector('y')
        doc_seq = gru_cell(x, y_hat, self.E_doc, self.V_doc, self.U_doc, self.W_doc, self.b_doc, self.c_doc)
        qn_seq = gru_cell(x, y_hat, self.E_qn, self.V_qn, self.U_qn, self.W_qn, self.b_qn, self.c_qn)

        ## perform dot product multiplication ##
        ## need to implement selective indexing ##
        prob = T.sum(doc_seq * qn_seq)

        # Total cost (could add regularization here)
        cost = T.sum( (prob - y_hat)**2 )

        # Gradients
        dE_doc = T.grad(cost, E_doc)
        dU_doc = T.grad(cost, U_doc)
        dW_doc = T.grad(cost, W_doc)
        db_doc = T.grad(cost, b_doc)
        dV_doc = T.grad(cost, V_doc)
        dc_doc = T.grad(cost, c_doc)

        dE_qn = T.grad(cost, E_qn)
        dU_qn = T.grad(cost, U_qn)
        dW_qn = T.grad(cost, W_qn)
        db_qn = T.grad(cost, b_qn)
        dV_qn = T.grad(cost, V_qn)
        dc_qn = T.grad(cost, c_qn)
        
        # Assign functions
        self.predict = theano.function([x], o)
        self.predict_class = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], cost)
        self.bptt = theano.function([x, y], \
        	[dE_doc, dU_doc, dW_doc, db_doc, dV_doc, dc_doc, \
        	dE_qn, dU_qn, dW_qn, db_qn, dV_qn, dc_qn])
        
        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')
        
        # rmsprop cache updates
        mE_doc = decay * self.mE_doc + (1 - decay) * dE_doc ** 2
        mU_doc = decay * self.mU_doc + (1 - decay) * dU_doc ** 2
        mW_doc = decay * self.mW_doc + (1 - decay) * dW_doc ** 2
        mV_doc = decay * self.mV_doc + (1 - decay) * dV_doc ** 2
        mb_doc = decay * self.mb_doc + (1 - decay) * db_doc ** 2
        mc_doc = decay * self.mc_doc + (1 - decay) * dc_doc ** 2

        mE_qn = decay * self.mE_qn + (1 - decay) * dE_qn ** 2
        mU_qn = decay * self.mU_qn + (1 - decay) * dU_qn ** 2
        mW_qn = decay * self.mW_qn + (1 - decay) * dW_qn ** 2
        mV_qn = decay * self.mV_qn + (1 - decay) * dV_qn ** 2
        mb_qn = decay * self.mb_qn + (1 - decay) * db_qn ** 2
        mc_qn = decay * self.mc_qn + (1 - decay) * dc_qn ** 2

        
        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.Param(decay, default=0.9)],
            [], 
            updates=[(E_doc, E_doc - learning_rate * dE_doc / T.sqrt(mE_doc + 1e-6)),
                     (U_doc, U_doc - learning_rate * dU_doc / T.sqrt(mU_doc + 1e-6)),
                     (W_doc, W_doc - learning_rate * dW_doc / T.sqrt(mW_doc + 1e-6)),
                     (V_doc, V_doc - learning_rate * dV_doc / T.sqrt(mV_doc + 1e-6)),
                     (b_doc, b_doc - learning_rate * db_doc / T.sqrt(mb_doc + 1e-6)),
                     (c_doc, c_doc - learning_rate * dc_doc / T.sqrt(mc_doc + 1e-6)),
                     (self.mE_doc, mE_doc),
                     (self.mU_doc, mU_doc),
                     (self.mW, mW_doc),
                     (self.mV_doc, mV_doc),
                     (self.mb_doc, mb_doc),
                     (self.mc_doc, mc_doc),
                     (E_qn, E_qn - learning_rate * dE_qn / T.sqrt(mE_qn + 1e-6)),
                     (U_qn, U_qn - learning_rate * dU_qn / T.sqrt(mU_qn + 1e-6)),
                     (W_qn, W_qn - learning_rate * dW_qn / T.sqrt(mW_qn + 1e-6)),
                     (V_qn, V_qn - learning_rate * dV_qn / T.sqrt(mV_qn + 1e-6)),
                     (b_qn, b_qn - learning_rate * db_qn / T.sqrt(mb_qn + 1e-6)),
                     (c_qn, c_qn - learning_rate * dc_qn / T.sqrt(mc_qn + 1e-6)),
                     (self.mE_qn, mE_qn),
                     (self.mU_qn, mU_qn),
                     (self.mW, mW_qn),
                     (self.mV_qn, mV_qn),
                     (self.mb_qn, mb_qn),
                     (self.mc_qn, mc_qn)
                    ])    

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)


