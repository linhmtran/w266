from __future__ import print_function
from __future__ import division

import tensorflow as tf
import math

def embedding_layer(ids_, V, embed_dim, init_scale=0.001):


    W_embed_ = tf.get_variable("W_embed", shape = [V, embed_dim], initializer=tf.random_uniform_initializer(-init_scale, init_scale, dtype=tf.float32))
    xs_ = tf.nn.embedding_lookup(W_embed_, ids_)


    return xs_

def fully_connected_layers(h0_, hidden_dims, activation=tf.tanh,
                           dropout_rate=0, is_training=False):

    h_ = h0_
    for i, hdim in enumerate(hidden_dims):
        h_ = tf.layers.dense(h_, hdim, activation=activation, name=("Hidden_%d"%i))

        do_ = tf.layers.dropout(inputs=h_, rate=dropout_rate, training=is_training)

        if dropout_rate > 0:
            h_ = do_  
    return h_

def softmax_output_layer(h_, labels_, num_classes):

    with tf.variable_scope("Logits"):

        W_out_ = tf.get_variable("W_out", [h_.shape[1], num_classes], initializer = tf.random_normal_initializer())
        b_out_ = tf.get_variable("b_out", [num_classes], initializer = tf.zeros_initializer()) 
        logits_ = tf.matmul(h_, W_out_)+ b_out_  # replace with (h W + b)

    if labels_ is None:
        return None, logits_

    with tf.name_scope("Softmax"):

        loss_ = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels_, logits = logits_))


    return loss_, logits_

def BOW_encoder(ids_, ns_, V, embed_dim, hidden_dims, dropout_rate=0,
                is_training=None,
                **unused_kw):
  
    assert is_training is not None,  "is_training must be explicitly set to True or False"

    with tf.variable_scope("Embedding_Layer"):

        xs_ = embedding_layer(ids_, V, embed_dim, init_scale=0.001)


    mask_ = tf.expand_dims(tf.sequence_mask(ns_, xs_.shape[1],
                                            dtype=tf.float32), -1)
    xs_ = tf.multiply(xs_, mask_)
    h_ = tf.reduce_sum(xs_, 1)
    h_ = fully_connected_layers(h_, hidden_dims, activation=tf.tanh, dropout_rate=0, is_training=False)
   
    return h_, xs_

def classifier_model_fn(features, labels, mode, params):

    tf.set_random_seed(params.get('rseed', 10))
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    if params['encoder_type'] == 'bow':
        with tf.variable_scope("Encoder"):
            h_, xs_ = BOW_encoder(features['ids'], features['ns'],
                                  is_training=is_training,
                                  **params)
    else:
        raise ValueError("Error: unsupported encoder type "
                         "'{:s}'".format(params['encoder_type']))


    with tf.variable_scope("Output_Layer"):
        ce_loss_, logits_ = softmax_output_layer(h_, labels, params['num_classes'])

    with tf.name_scope("Prediction"):
        pred_proba_ = tf.nn.softmax(logits_, name="pred_proba")
        pred_max_ = tf.argmax(logits_, 1, name="pred_max")
        predictions_dict = {"proba": pred_proba_, "max": pred_max_}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions_dict)

    with tf.variable_scope("Regularization"):
        l2_penalty_ = tf.nn.l2_loss(xs_) 
        for var_ in tf.trainable_variables():
            if "Embedding_Layer" in var_.name:
                continue
            l2_penalty_ += tf.nn.l2_loss(var_)
        l2_penalty_ *= params['beta'] 
        tf.summary.scalar("l2_penalty", l2_penalty_)
        regularized_loss_ = ce_loss_ + l2_penalty_

    with tf.variable_scope("Training"):
        if params['optimizer'] == 'adagrad':
            optimizer_ = tf.train.AdagradOptimizer(params['lr'])
        else:
            optimizer_ = tf.train.GradientDescentOptimizer(params['lr'])
        train_op_ = optimizer_.minimize(regularized_loss_,
                                        global_step=tf.train.get_global_step())

    tf.summary.scalar("cross_entropy_loss", ce_loss_)
    eval_metrics = {"cross_entropy_loss": tf.metrics.mean(ce_loss_),
                    "accuracy": tf.metrics.accuracy(labels, pred_max_),

          
                       
                   }


    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions_dict,
                                      loss=regularized_loss_,
                                      train_op=train_op_,
                                      eval_metric_ops=eval_metrics)
