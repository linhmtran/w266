
# coding: utf-8

# # MBTI Parallel Classification Model with LSTM
# 
# First, load libraries and useful functions from class:

# In[1]:


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import json, os, re, shutil, sys, time
from importlib import reload
import collections, itertools

# NumPy and TensorFlow
import numpy as np
import pandas as pd
import tensorflow as tf
assert(tf.__version__.startswith("1."))

# Utils and Helper libraries
# import nltk
from w266_common import utils, vocabulary
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import MBTI2x4_LSTM_model as model
import MBTI_data_setup as ds


# ## Load Corpus & Pre-Process

# In[2]:


#load data
df = pd.read_csv('./mbti_1.csv')
df.head(5)


# In[3]:


reload(ds)
post, mbti_type, user = ds.splitPosts(df)

# Split data: 80% train, 20% test
post_train, post_test, label_train, label_test = train_test_split(post, mbti_type, test_size=0.2, random_state=88)

print("MBIT posts", post_train[:5])
print('')
print("MBTI Labels: ",label_train[:5])

# Build a vocabulary (V size is defaulted to full text) for train corpus
vocab_mbti = vocabulary.Vocabulary((utils.canonicalize_word(w) for w in post_train))
print("Vocab Size: ",vocab_mbti.size)

# tokenize and canonicalize train and test sets
x_train = []
for post in post_train:
    x_train.append(vocab_mbti.words_to_ids(post.split()))

x_test = []
for post in post_test:
    x_test.append(vocab_mbti.words_to_ids(post.split()))
    
reload(ds)
y_train_id, y_test_id = ds.one_hot_label(mbti_type, label_train, label_test)
y_train, y_test = ds.label_to_id(mbti_type, label_train, label_test)

print(y_train_id[:2])
print(y_test_id[:2])
print(y_train[:2])
print(y_test[:2])


# In[7]:


# 4 dichotomomies: {Attitude: E/I, Information: N/S , Decision: F/T, Structure: J/P }
mbti_map = {'E':0, 'I':1, 'N':2, 'S':3, 'F':4, 'T':5, 'J':6,'P':7 } # indicates the the index column for each type

# split type string --> ['I' 'N' 'T' 'P']
dichotomy_train = np.array([list(mbti_type) for mbti_type in label_train])
dichotomy_test = np.array([list(mbti_type) for mbti_type in label_test])

# convert dichotomy indicators to index value --> [1 2 5 7]
y_train_id = np.array([list(map(lambda x: mbti_map[x],ind)) for ind in dichotomy_train])
y_test_id = np.array([list(map(lambda x: mbti_map[x],ind)) for ind in dichotomy_test])

# create 1-hot vector with MBTI dichotomies, map index value to 1 --> [0,1,1,0,1,0,1,0]
one_hot_train = np.zeros((len(y_train_id),8),dtype=int)
one_hot_test = np.zeros((len(y_test_id),8),dtype=int)
for i in range(4):
    one_hot_train[np.arange(len(dichotomy_train)), y_train_id[:,i]] = 1
    one_hot_test[np.arange(len(dichotomy_test)), y_test_id[:,i]] = 1
y_train = one_hot_train
y_test = one_hot_test

y_train = y_train[:,2:4]
print("Example Y Train one hot for Training: ", y_train[:5])
y_train_id = y_train_id[:,1]-2
print("Example Y Train ID for Accuracy Scoring: ", y_train_id[:5])


y_test = y_test[:,2:4]
print("Example Y Test one hot for Testing: ", y_test[:5])
y_test_id = y_test_id[:,1]-2
print("Example Y Train ID for Accuracy Scoring: ", y_test_id[:5])


# ## Bulid the LSTM Model

# In[8]:


# Model Parameters
V = vocab_mbti.size
classes = 2
batch_size = 25
text_length = 100 # fixed max length of text, but need to handle truncating
embed_dim = 300 #256
hidden_dims = 300 # Needs to be the same dims for LSTM [256,256]
num_layers = 1

# Training Parameters
dropout_keep_prob = .5
learning_rate = .01
softmax_ns = 4
num_epochs = 5


# In[9]:


tf.reset_default_graph()
tf.set_random_seed(88)
# reload(model)

# def matmul3d(X, W):
#     """Wrapper for tf.matmul to handle a 3D input tensor X.
#     Will perform multiplication along the last dimension.

#     Args:
#       X: [m,n,k]
#       W: [k,l]

#     Returns:
#       XW: [m,n,l]
#     """
#     Xr = tf.reshape(X, [-1, tf.shape(X)[2]])
#     XWr = tf.matmul(Xr, W)
#     newshape = [tf.shape(X)[0], tf.shape(X)[1], tf.shape(W)[1]]
#     return tf.reshape(XWr, newshape)

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


# Create input placeholder
with tf.name_scope("Inputs"):
    x_text_ = tf.placeholder(tf.int32, [None, None], name="x_text") #batch x text_length
    y_type_ = tf.placeholder(tf.int32, [None,classes], name="y_type") #batch x classes

    
# Model params
with tf.name_scope("Dynamic_Params"):
    # Get dynamic shape info from inputs
    batch_size_ = tf.shape(x_text_)[0]
    text_length_ = text_length
    ns_ = tf.tile([text_length_], [batch_size_, ], name="ns")
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
#     W_out_ = tf.Variable(tf.random_uniform([hidden_dims,classes],-1.0, 1.0), name="W_out")
#     b_out_ = tf.Variable(tf.zeros([classes,], dtype=tf.float32), name="b_out")
#     logits_ = tf.add(matmul3d(output_, W_out_), b_out_, name="logits")
#     print("W_out: ",W_out_.shape)
#     print("Logits: ",logits_.shape)
    output_ = tf.reshape(output_, [batch_size_,text_length*hidden_dims])
    print("flattened output shape: ",output_.shape)
    W_out_ = tf.Variable(tf.random_uniform([hidden_dims*text_length,classes],-1.0, 1.0), name="W_out")
    print("W_out: ",W_out_.shape)
    b_out_ = tf.Variable(tf.zeros([classes,], dtype=tf.float32), name="b_out")
    logits_ = tf.add(tf.matmul(output_,W_out_), b_out_, name="logits")
    print("Logits: ",logits_.shape)


with tf.name_scope("Cost_Function"):
    # Full softmax loss for training / scoring
    per_example_loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_type_, 
                                                                   logits=logits_,
                                                                   name="per_example_loss")
    loss_ = tf.reduce_mean(per_example_loss_, name="loss")
    
    # Sampled softmax for training
    train_inputs_ = tf.reshape(output_, [batch_size_*text_length_,-1])

    per_example_sampled_softmax_loss_ = tf.nn.sampled_softmax_loss(weights=tf.transpose(W_out_),
                                                                   biases=b_out_,
                                                                   labels=tf.reshape(y_type_, [-1, 1]),
                                                                   inputs=train_inputs_,
                                                                   num_sampled=softmax_ns, 
                                                                   num_classes=classes,
                                                                   name="per_example_sampled_softmax_loss")

    sampled_softmax_loss_ = tf.reduce_mean(per_example_sampled_softmax_loss_, name="sampled_softmax_loss")
    
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

    
with tf.name_scope("Train"):
    learning_rate_ = tf.placeholder(tf.float32, name="learning_rate")
    optimizer_ = tf.train.AdamOptimizer(learning_rate_)
    gradients, variables = zip(*optimizer_.compute_gradients(loss_))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    train_step_ = optimizer_.apply_gradients(zip(gradients, variables))

# Initializer step
init_ = tf.global_variables_initializer()


# In[14]:


# tensorboard --logdir="tf_graph" --port 6006
summary_writer = tf.summary.FileWriter("tf_graph", 
                                       tf.get_default_graph())


# ## Train Model

# In[13]:


# create batched arrays of size batch_size based on input x and input y
def pad_np_array(example_ids, max_len=100, pad_id=0):
    arr = np.full([len(example_ids), max_len], pad_id, dtype=np.int32)
    ns = np.zeros([len(example_ids)], dtype=np.int32)
    for i, ids in enumerate(example_ids):
        cpy_len = min(len(ids), max_len)
        arr[i,:cpy_len] = ids[:cpy_len]
        ns[i] = cpy_len
    return arr, ns

def batch_generator(x, y, batch_size):
    for i in range(0, len(x), batch_size):
        # padd the batch
        x_batch = x[i:i+batch_size]
        x_padded, _ = pad_np_array(x_batch, max_len=text_length, pad_id=0)
        yield (x_padded, y[i:i+batch_size]) # returns tuple of batched x, batched y

def train_batch(session, xbatch, ybatch, learning_rate):
    feed_dict = {x_text_:xbatch, #np array of texts
                 y_type_:ybatch, #np array of types
                 learning_rate_:learning_rate}
#     c, _ = session.run([sampled_softmax_loss_, train_step_],
    c, _ = session.run([loss_, train_step_],
                       feed_dict=feed_dict)
    return c


# In[ ]:


print_interval = 1000

np.random.seed(88)

session = tf.Session()
session.run(init_)

# #Override for pilot training
# batch_size = 50
# num_epochs = 5

t0 = time.time()
for epoch in range(1,num_epochs+1):
    t0_epoch = time.time()
    epoch_cost = 0.0
    total_batches = 0
    print ("")
    for i, (x,y) in enumerate(batch_generator(x_train, y_train, batch_size)):
        if (i % print_interval == 0):
            print("[epoch %d] seen %d batches" % (epoch, i))
            
        epoch_cost += train_batch(session, x, y, learning_rate)
        total_batches = i + 1

    avg_cost = epoch_cost / total_batches
    print("[epoch %d] Completed %d batches in %s" % (epoch, i, utils.pretty_timedelta(since=t0_epoch)))
    print("[epoch %d] Average cost: %.03f" % (epoch, avg_cost,))


# ## Evaluate Model

# In[79]:


def predict_type(session,x):
    inputs = {x_text_:x,#np array of texts
             isTrain_: False} 
    pred = session.run([pred_max_],feed_dict=inputs)
    return pred # batch x 1


# In[80]:


# Accuracy on train set
train_accuracy=[]
for i, (x,y) in enumerate(batch_generator(x_train, y_train_id, batch_size)):
    pred = predict_type(session,x)
    train_accuracy.append((pred[0] == y).mean())
print("Train accuracy: " ,np.mean(train_accuracy))


# In[81]:


# Accuray on test set
test_accuracy = []
for i, (x,y) in enumerate(batch_generator(x_test, y_test_id, batch_size)):
    pred = predict_type(session,x)
    test_accuracy.append((pred[0] == y).mean())
print("Test accuracy: ",np.mean(test_accuracy))


# In[82]:


def score_batch(session, x, y):
    feed_dict = {x_text_:x,
                 y_type_:y}
    return session.run(loss_, feed_dict=feed_dict)

def score_dataset(x, y):
    total_cost = 0.0
    total_batches = 0
    for (x,y) in batch_generator(x, y, 5):
        total_cost += score_batch(session, x, y)
        total_batches += 1

    return total_cost / total_batches


# In[ ]:


print ("Train set perplexity: %.03f" % np.exp(score_dataset(x_train,y_train)))
print ("Test set perplexity: %.03f" % np.exp(score_dataset(x_test,y_test)))

