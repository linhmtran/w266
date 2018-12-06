from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

import tensorflow as tf
import numpy as np

def matmul3d(X, W):
    """Wrapper for tf.matmul to handle a 3D input tensor X.
    Will perform multiplication along the last dimension.

    Args:
      X: [m,n,k]
      W: [k,l]

    Returns:
      XW: [m,n,l]
    """
    Xr = tf.reshape(X, [-1, tf.shape(X)[2]])
    XWr = tf.matmul(Xr, W)
    newshape = [tf.shape(X)[0], tf.shape(X)[1], tf.shape(W)[1]]
    return tf.reshape(XWr, newshape)


# useful functions for building LSTM
def MakeFancyRNNCell(hidden_dims, keep_prob, layers=1, isTrain=True):
    """Make a fancy RNN cell.

    Use tf.nn.rnn_cell functions to construct an LSTM cell.
    Initialize forget_bias=0.0 for better training.

    Args:
      H: hidden state sizes, provided in array
      keep_prob: dropout keep prob (same for input and output)

    Returns:
      (tf.nn.rnn_cell.RNNCell) multi-layer LSTM cell with dropout
    """
    if isTrain == False:
        keep_prob = 1.0

    cells = []
    for _ in range(layers):
#         cell = tf.nn.rnn_cell.BasicLSTMCell(H, forget_bias=0.0) #deprecated?
        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dims, forget_bias=0.0,state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, 
                                             input_keep_prob=keep_prob, 
                                             output_keep_prob=keep_prob)
        cells.append(cell)
    return tf.nn.rnn_cell.MultiRNNCell(cells)

def with_self_graph(function):
    def wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return function(self, *args, **kwargs)
    return wrapper


class LSTM(object):
    def __init__(self, graph=None, *args, **kwargs):
        """Init function.

        This function just stores hyperparameters. You'll do all the real graph
        construction in the Build*Graph() functions below.

        Args:
          V: vocabulary size
          H: hidden state dimension
          num_layers: number of RNN layers (see tf.nn.rnn_cell.MultiRNNCell)
        """
        # Set TensorFlow graph. All TF code will work on this graph.
        self.graph = graph or tf.Graph()
        self.SetParams(*args, **kwargs)

    @with_self_graph
    def SetParams(self, V, H, softmax_ns=200, num_layers=1):
        # Model structure; these need to be fixed for a given model.
        self.V = V
        self.H = H
        self.num_layers = num_layers

        # Training hyperparameters; these can be changed with feed_dict,
        # and you may want to do so during training.
        with tf.name_scope("Training_Parameters"):
            # Number of samples for sampled softmax.
            self.softmax_ns = softmax_ns

            self.learning_rate_ = tf.placeholder(tf.float32, [], name="learning_rate")

            # For gradient clipping, if you use it.
            # Due to a bug in TensorFlow, this needs to be an ordinary python
            # constant instead of a tf.constant.
            self.max_grad_norm_ = 1.0

            self.use_dropout_ = tf.placeholder_with_default(
                False, [], name="use_dropout")

            # If use_dropout is fed as 'True', this will have value 0.5.
            self.dropout_keep_prob_ = tf.cond(
                self.use_dropout_,
                lambda: tf.constant(0.5),
                lambda: tf.constant(1.0),
                name="dropout_keep_prob")

            # Dummy for use later.
            self.no_op_ = tf.no_op()

    def graph(self):
        # Create input placeholder
        with tf.name_scope("Inputs"):
            x_text_ = tf.placeholder(tf.int32, [None, None], name="x_text") #batch x text_length
            y_type_ = tf.placeholder(tf.int32, [None,classes], name="y_type") #batch x classes


        # Model params
        with tf.name_scope("Dynamic_Params"):
            # Get dynamic shape info from inputs
            batch_size_ = tf.shape(x_text_)[0]
            ns_ = tf.tile([text_length], [batch_size_, ], name="ns")
            isTrain_ = tf.placeholder(tf.bool, shape=())

        # Construct embedding layer
        with tf.name_scope("Embedding_Layer"):
        #     W_in_ = tf.get_variable(tf.random_uniform(-1.0, 1.0),shape=[V, embed_dim], name="W_in")
            W_in_ = tf.Variable(tf.random_uniform([V, embed_dim], -1.0, 1.0), name="W_in")
            x_ = tf.nn.embedding_lookup(W_in_, x_text_,name='x')
            print("Embedded Input: ", x_.shape)

        # Construct RNN/LSTM cell and recurrent layer.
        with tf.name_scope("Recurrent_Layer"):
            # 2 Layer LSTM
            cell_lstm2_ = MakeFancyRNNCell(hidden_dims, dropout_keep_prob, num_layers,isTrain_)
            initial_h_ = cell_lstm2_.zero_state(batch_size_, dtype=tf.float32)
            output_, final_h_= tf.nn.dynamic_rnn(cell=cell_lstm2_, inputs=x_, 
                                                 sequence_length= ns_, initial_state = initial_h_, dtype=tf.float32)
            print("LSTM Cell output shape: ",output_.shape)

        with tf.name_scope("Output_Layer"):
            output_ = tf.reshape(output_, [batch_size_,text_length*hidden_dims])
            print("flattened output shape: ",output_.shape)
            W_out_ = tf.Variable(tf.random_uniform([hidden_dims*text_length,classes],-1.0, 1.0), name="W_out")
            print("W_out: ",W_out_.shape)
            b_out_ = tf.Variable(tf.zeros([classes,], dtype=tf.float32), name="b_out")
            logits_ = tf.add(tf.matmul(output_,W_out_), b_out_, name="logits")
            print("Logits: ",logits_.shape)


    def graphTrain(self):
        with tf.name_scope("Cost_Function"):
            # Full softmax loss for training / scoring
            per_example_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_type_, 
                                                                           logits=logits_,
                                                                           name="per_example_loss")
            loss_ = tf.reduce_mean(per_example_loss_, name="loss")

            # Sampled softmax for training
        #     train_inputs_ = tf.reshape(self.output_, [batch_size_*text_length,-1])

        #     per_example_sampled_softmax_loss_ = tf.nn.sampled_softmax_loss(weights=W_out_,
        #                                                                    biases=b_out_,
        #                                                                    labels=y_type_,
        # #                                                                    labels=tf.reshape(y_train_, [-1, 1]),
        #                                                                    inputs=output_,
        #                                                                    num_sampled=ns_, 
        #                                                                    num_classes=classes,
        #                                                                    name="per_example_sampled_softmax_loss")

        #     sampled_softmax_loss_ = tf.reduce_mean(per_example_sampled_softmax_loss_, name="sampled_softmax_loss")

        with tf.name_scope("Train"):
            learning_rate_ = tf.placeholder(tf.float32, name="learning_rate")
            optimizer_ = tf.train.AdamOptimizer(learning_rate_)
            gradients, variables = zip(*optimizer_.compute_gradients(loss_))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            train_step_ = optimizer_.apply_gradients(zip(gradients, variables))
            
    def graphPredict(self):
        with tf.name_scope("Prediction"):
            pred_proba_ = tf.nn.softmax(logits_, name="pred_proba")
            pred_max_ = tf.argmax(logits_, 1, name="pred_max")
            pred_random_ = tf.reshape(tf.multinomial(tf.reshape(logits_ , [-1, classes]), 
                                                                  1, 
                                                                  output_dtype=tf.int32, 
                                                                  name="pred_random"),
                                                   [batch_size_,1])
            print("Pred Prob Shape: ",pred_proba_.shape)
            print("Pred Max Shape: ",pred_max_.shape)
            print("Sampling Shape: ",pred_random_.shape)


