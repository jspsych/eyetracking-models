#!/usr/bin/env python
# coding: utf-8

# #Data Preparation 

# In[ ]:


import numpy as np 
import tensorflow as tf
from tensorflow import keras
from keras import layers
from google.colab import drive 
import os
import json
import random
drive.mount('/content/drive/')

# In[ ]:


json_files = os.listdir('/content/drive/Shareddrives/Eye Tracking Research/Eye Tracking ML/json_inputs')
raw_data = []
raw_labels = []
raw_cnn_data = []
for name in json_files:
  with open('/content/drive/Shareddrives/Eye Tracking Research/Eye Tracking ML/json_inputs/'+name) as j:
    allfiles = json.loads(j.read())[name[0:8]]
  with open('/content/drive/Shareddrives/Eye Tracking Research/Eye Tracking ML/raw_cnn_data/'+name[0:8]+'.json') as i:
    cnnfiles = json.loads(i.read())
  raw_data.append(allfiles['x'])
  raw_labels.append(allfiles['y'])
  ordered_cnn = []
  for x,y in raw_labels[:]:
    for data in cnnfiles:
      if(int(data['x'])==x and int(data['y'])==y):
          ordered_cnn.append(data)
          break
  raw_cnn_data.append(ordered_cnn)

for i in range(len(raw_data)):
  swapIndex = random.randint(i,len(raw_data)-1)
  tempData = raw_data[i][:]
  tempLabels = raw_labels[i][:]
  raw_data[i] = raw_data[swapIndex][:]
  raw_labels[i] = raw_labels[swapIndex][:]
  raw_data[swapIndex] = tempData
  raw_labels[swapIndex] = tempLabels

nonflat_train_data = []
nonflat_val_data = []
train_labels = []
val_labels = []

for a in raw_data[:115]:
  for i in a:
    nonflat_train_data.append(i)
for a in raw_data[115:]:
  for i in a:
    nonflat_val_data.append(i)
for a in raw_labels[:115]:
  for i in a:
    train_labels.append(i)
for a in raw_labels[115:]:
  for i in a:
    val_labels.append(i)



# In[ ]:


def flatten_data(nonflat_data):
  data = []
  for sample in nonflat_data:
    flat_sample = []
    for point in sample:
      flat_point = []
      for vae_points in point[0]:
        flat_point.append(vae_points)
      for face_points in point[1:]:
        for point in face_points:
          for xy in point:
            flat_point.append(xy)
      flat_sample.append(flat_point)
    data.append(flat_sample)
  return data

train_data = flatten_data(nonflat_train_data)
val_data = flatten_data(nonflat_val_data)

train_data = np.array(train_data)
val_data = np.array(val_data)
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
train_data = train_data.astype(np.float32)
train_labels = train_labels.astype(np.float32)
val_data = val_data.astype(np.float32)
val_labels = val_labels.astype(np.float32)

std = np.std(tf.concat([train_data, val_data], 0))
mean = np.mean(tf.concat([train_data, val_data], 0))

train_data -= mean
train_data /= std
val_data -=mean
val_data /=std

prev_output = tf.tile(tf.constant([[[10,10],[10,50],[10,90],[30,30],[30,70],[50,10],[50,50],[50,90],
                                  [70,30],[70,70],[90,10],[90,50],[90,90]]], dtype = 'float32'),
                          [tf.shape(train_data)[0],1,1])
val_prev_output = tf.tile(tf.constant([[[10,10],[10,50],[10,90],[30,30],[30,70],[50,10],[50,50],[50,90],
                                  [70,30],[70,70],[90,10],[90,50],[90,90]]], dtype = 'float32'),
                          [tf.shape(val_data)[0],1,1])

# In[ ]:


def rearrange_data (arr):
  result_array = []
  for i in arr:
    result_array.append(i[::-1])
  return result_array

rnn_train_data = np.array(rearrange_data(train_data))
rnn_val_data = np.array(rearrange_data(val_data))
rnn_val_prev_output = np.array(rearrange_data(val_prev_output))
rnn_prev_output = np.array(rearrange_data(prev_output))

vae_train_data = rnn_train_data[:,:,:6]
eyeR_train_data = rnn_train_data[:,:,6:14]
eyeL_train_data = rnn_train_data[:,:,14:]

vae_val_data = rnn_val_data[:,:,:6]
eyeR_val_data = rnn_val_data[:,:,6:14]
eyeL_val_data = rnn_val_data[:,:,14:]

# In[ ]:


class Euclidian_Distance_Error(keras.metrics.Metric):
  def __init__ (self, name = 'distance', **kwargs):
    super().__init__(name=name, **kwargs)    
    self.total_dist = self.add_weight(name = 'true_cords', initializer = 'zeros', dtype = 'float32')
    self.total_samples = self.add_weight(name = 'total samples', initializer = 'zeros', dtype = 'int32')

  def update_state(self, y_true, y_pred, sample_weight = None):
    for i in range(tf.shape(y_true)[0]):
      self.total_dist.assign_add(tf.sqrt(tf.math.square(y_true[i][0]-y_pred[i][0])+tf.math.square(y_true[i][1]-y_pred[i][1])))
    self.total_samples.assign_add(tf.shape(y_true)[0])

  def result(self):
    return self.total_dist/tf.cast(self.total_samples,'float32')

  #ensure that this class does not need to be reinstantiated to reset the sum and total samples
  def reset_state(self):
    self.total_dist.assign(0.)
    self.total_samples.assign(0)

class RootMeanSquaredError(keras.metrics.Metric):
  def __init__ (self, name = 'rmse', **kwargs):
    super().__init__(name=name, **kwargs)    
    self.mse_sum = self.add_weight(name = 'mse_sume', initializer = 'zeros')
    self.total_samples = self.add_weight(name = 'totla_samples', initializer = 'zeros', dtype = 'int32')

  def update_state(self, y_true, y_pred, sample_weight = None):
    ##y_true is the actual output and y_pred is the one from the model
    #converts y_true into a one side hot vector so its like the model
    print(y_true)
    y_true = tf.one_hot(y_true, depth = tf.shape(y_pred)[1])
    #computes mean squared error
    mse = tf.reduce_sum(tf.square(y_true - y_pred))
    self.mse_sum.assign_add(mse)
    num_samples = tf.shape(y_pred)[1]
    self.total_samples.assign_add(num_samples)
    

  def result(self):
    #takes information given by update_state and returns the loss 
    #must divide by total so it gives the average loss over the entire epoch
    return tf.sqrt(self.mse_sum/tf.cast(self.total_samples, tf.float32))

  #ensure that this class does not need to be reinstantiated to reset the sum and total samples
  def reset_state(self):
    self.mse_sum.assign(0.)
    self.total_samples.assign(0)


# In[ ]:




# #Currently Developing Models

# Layer Definitions

# In[ ]:


class attention_layer (keras.layers.Layer):
  def __init__ (self, num_heads, key_dims, value_dim = None, **kwargs):
    super().__init__(**kwargs)
    self.num_heads = num_heads 
    self.key_dims = key_dims 
    self.value_dim = value_dim
    self.resize_queries = keras.layers.Dense(key_dims)
    self.resize_keys = keras.layers.Dense(key_dims)
    
    self.embed_keys = keras.layers.Embedding(
        input_dim = 13, output_dim = key_dims
    )
    self.embed_values = keras.layers.Embedding(
        input_dim = 13, output_dim = value_dim
    )
    self.attention_layer = keras.layers.MultiHeadAttention(num_heads, key_dims, value_dim = self.value_dim)
    self.mixing_layer = keras.layers.Dense(self.key_dims, activation = 'LeakyReLU')
    self.flatten_layer = keras.layers.Flatten()
    self.dense_layer = keras.layers.Dense(2)       
    self.key_length = None
    self.value_length = None
    self.positions = None
  
  def get_config(self):
    config = super().get_config()
    config.update({
        "num_heads" : self.num_heads,
        "key_dims": self.key_dims, 
        "value_dim": self.value_dim
    })
    return config

  def call(self, queries, values, keys):
    self.positions = tf.range(start = 0, limit = tf.shape(keys)[-2], delta = 1)
    
    resized_queries = self.resize_queries(queries)

    self.key_length = tf.shape(keys)[-2]
    key_position_embeddings = self.embed_keys(self.positions)
    resized_keys = self.resize_keys(keys)
    embedded_keys = key_position_embeddings+resized_keys
    
    self.value_length = tf.shape(values)[-2]
    value_position_embeddings = self.embed_values(self.positions)
    embedded_values = value_position_embeddings+values

    attention_output = self.attention_layer(resized_queries,embedded_keys,embedded_values)
    mixing_layer = self.mixing_layer(attention_output)
    flatten_layer = self.flatten_layer(mixing_layer)
    return self.dense_layer(flatten_layer)

# 

# In[ ]:


width = 16
heads = 8
input = keras.Input(shape = (14,22,))
values = keras.Input(shape = (13,2))

queries = input[:,13:14,:]
keys = input[:,:13,:]

outputs = attention_layer(heads, width, 2)(queries, values, keys)

model = keras.Model(inputs = [input, values], outputs = [outputs])

model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = 'mse',
    metrics = [Euclidian_Distance_Error()]
)
# def schedule(epoch, lr):
#   if epoch < 50:
#     return lr
#   elif lr<=0.0007:
#     return lr
#   else:
#     return lr * tf.math.exp(-0.00000001)

callbacks_list = [
    keras.callbacks.TensorBoard(
        log_dir = f'/modified_LSTM'
    ),
    keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 1000,
        restore_best_weights = True
    ), 
    keras.callbacks.ModelCheckpoint(
        monitor = 'val_loss',
        save_best_only = True, 
        filepath = 'special_lstm.keras'
    ),
]
model.summary()
model.fit(
    [rnn_train_data, rnn_prev_output], 
    [train_labels], 
    batch_size = 64, 
    epochs = 6500, 
    callbacks = callbacks_list,
    validation_data = [[rnn_val_data,rnn_val_prev_output], val_labels]
)

model.evaluate([rnn_val_data,rnn_val_prev_output], val_labels)
%reload_ext tensorboard
%tensorboard --logdir /modified_LSTM

# #Single Head

# ## Feedforward

# Manual Testing DNN shape(3-7,1024) + 0.4 dropout and L2 Regularizer

# In[ ]:


inputs = keras.layers.Input(shape = (14,22,))

features = keras.layers.Flatten()(inputs)
features = keras.layers.Dense(64, activation = 'relu')(features)
features = keras.layers.Dropout(.0)(features)
features = keras.layers.Dense(64, activation = 'relu')(features)
features = keras.layers.Dropout(.0)(features)
features = keras.layers.Dense(64, activation = 'relu')(features)
features = keras.layers.Dropout(.0)(features)


outputs = keras.layers.Dense(2)(features)

model = keras.Model(inputs,outputs)
model.compile(
    optimizer = 'adam',
    loss = 'mse',
    metrics = [Euclidian_Distance_Error()]
)

callbacks_list = [

    keras.callbacks.ModelCheckpoint(
        filepath = f'1024x3+l2d4.keras',
        monitor = 'val_loss',
        save_best_only = True
    ),
    keras.callbacks.TensorBoard(
        log_dir = f'/1024x3'
    ),
    keras.callbacks.EarlyStopping(
        monitor = 'val_distance',
        patience = 100
    )
]

model.fit(
    train_data, train_labels,
    epochs = 10000,
    batch_size = 64,
    callbacks = callbacks_list,
    validation_data = (val_data, val_labels),
)
model.summary()
model.evaluate(val_data, val_labels)

%load_ext tensorboard
%tensorboard --logdir /1024x3

# Keras Tuner Automatic Testing

# In[ ]:


!pip install -q -U keras-tuner
import keras_tuner as kt
def build_model (hp):
  inputs = keras.Input(shape = (14,22,))
  features = keras.layers.Flatten()(inputs)
  features = keras.layers.Dense(1024, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.03))(features)
  features = keras.layers.Dropout(0.4)(features)
  features = keras.layers.Dense(2048, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.03))(features)
  features = keras.layers.Dropout(0.4)(features)
  features = keras.layers.Dense(1024, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.03))(features)
  features = keras.layers.Dropout(0.4)(features)
  
  outputs = keras.layers.Dense(2)(features)
  model = keras.Model(inputs, outputs)
  
  hp_learning = hp.Choice('learning_rate', values = [0.04,0.03,0.02,0.01,0.005,0.001])
  model.compile(
      optimizer = keras.optimizers.Adam(learning_rate = hp_learning),
      loss = 'mse',
      metrics = ['mae']
  )
  return model

tuner = kt.Hyperband(
    build_model,
    objective = 'val_mae',
    max_epochs = 10000,
    factor = 3,
    project_name = 'optimize_dnn_for_gaze_prediction',
)

callbacks_list = [
    keras.callbacks.EarlyStopping(
        patience = 1000
    )
]

tuner.search(
    train_data, train_labels, 
    epochs = 50, 
    batch_size = 64,
    validation_data = [val_data,val_labels],
    callbacks = callbacks_list)

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(best_hps.get('learning_rate'))


# ##RNN

# LSTM Ideas 
# - change resize layer to before each lstm block
# - add residual connections
# - add multiheaded attention before layer

# In[ ]:


class positional_embedding_layer (keras.layers.Layer):
  def __init__ (self, sequence_length, output_dim, **kwargs):
    super().__init__(**kwargs)
    self.sequence_length = sequence_length
    self.output_dim = output_dim
    self.embed_positions = keras.layers.Embedding(
        input_dim = sequence_length, output_dim = output_dim
    )
    self.resize_layer = keras.layers.Dense(output_dim, activation = 'LeakyReLU')


  def get_config(self):
    config = super().get_config()
    config.update({
        'sequence_length': self.sequence_length,
        'output_dim': self.output_dim
    })
    return config 

  def call(self, inputs):
    length = tf.shape(inputs)[-2]
    positions = tf.range(start=0, limit=length, delta=1)
    position_embeddings = self.embed_positions(positions)
    resized_inputs = self.resize_layer(inputs)
    embedded_vector = position_embeddings+resized_inputs
    return embedded_vector
  
  def compute_mask(self, inputs, mask = None):
    return tf.math.not_equal(inputs, 0)

class modified_LSTM_block (keras.layers.Layer):
  def __init__ (self, output_dim = 4, initializer = 'random_normal', **kwargs):
    super().__init__(**kwargs)
    self.initializer = initializer
    self.throughput = output_dim

  def get_config(self):
    config = super().get_config()
    config.update({
      'output_dim': self.throughput,
      'initializer': self.initializer  
    })
    return config


  def build (self, input_shape):
    #weights and biases for forget gate
    self.forget_W = self.add_weight(
        shape = (input_shape[-1], self.throughput), 
        initializer = self.initializer, 
        trainable = True
    )
    self.forget_U = self.add_weight(
        shape = (input_shape[-1], self.throughput), 
        initializer = self.initializer, 
        trainable = True
    )
    self.forget_b = self.add_weight(
        shape = (self.throughput,), 
        initializer = self.initializer, 
        trainable = True
    )
    #weights and biases for input gate
    self.input_W = self.add_weight(
        shape = (input_shape[-1], self.throughput), 
        initializer = self.initializer, 
        trainable = True
    )
    self.input_U = self.add_weight(
        shape = (input_shape[-1], self.throughput), 
        initializer = self.initializer, 
        trainable = True
    )
    self.input_b = self.add_weight(
        shape = (self.throughput,), 
        initializer = self.initializer, 
        trainable = True
    )
    #weights and biases for updating cell state 
    self.cell_W = self.add_weight(
        shape = (input_shape[-1], self.throughput), 
        initializer = self.initializer, 
        trainable = True
    )
    self.cell_U = self.add_weight(
        shape = (input_shape[-1], self.throughput), 
        initializer = self.initializer, 
        trainable = True
    )
    self.cell_b = self.add_weight(
        shape = (self.throughput,), 
        initializer = self.initializer, 
        trainable = True
    )

  def call (self, inputs):
    hidden_state_1 = tf.expand_dims(inputs[0],0)
    hidden_state = tf.expand_dims(inputs[1],0)
    cell_state = tf.expand_dims(inputs[2],0)
    forget_vector = tf.keras.activations.sigmoid(tf.matmul(hidden_state,self.forget_W)+tf.matmul(hidden_state_1,self.forget_U)+self.forget_b)
    input_vector = tf.keras.activations.sigmoid(tf.matmul(hidden_state,self.input_W)+tf.matmul(hidden_state_1,self.input_U)+self.input_b)
    cell_update_vector = tf.matmul(hidden_state,self.cell_W)+tf.matmul(hidden_state_1,self.cell_U)+self.cell_b
    cell_update_vector = tf.keras.layers.LeakyReLU()(cell_update_vector)
    new_cell = input_vector*cell_update_vector
    return cell_state*forget_vector+new_cell


class modified_LSTM_output_block(keras.layers.Layer):
  def __init__ (self, output_dim = 4, initializer = 'random_normal', **kwargs):
    super().__init__(**kwargs)
    self.throughput = output_dim
    self.initializer = initializer
  
  def get_config(self):
    config = super().get_config()
    config.update({
      'output_dim': self.throuhgput,
      'initializer': self.initializer  
    })
    return config


  def build (self, input_shape):
    self.input_W = self.add_weight(
        shape = (input_shape[-1], self.throughput), 
        initializer = self.initializer, 
        trainable = True
    )
    self.input_b = self.add_weight(
        shape = (self.throughput,), 
        initializer = self.initializer, 
        trainable = True
    )
    self.cell_W = self.add_weight(
        shape = (input_shape[-1], self.throughput), 
        initializer = self.initializer, 
        trainable = True
    )
  
  def call (self, inputs):
    hidden_state = inputs[0]
    cell_state = inputs[1]
    return (tf.matmul(hidden_state, self.input_W)+self.input_b)+keras.layers.LeakyReLU()(tf.matmul(cell_state,self.cell_W))

class initialize_cellstate(keras.layers.Layer):
  def __init__(self, output_dim = 4, initializer = 'random_normal', **kwargs):
    super().__init__(**kwargs)
    self.output_dim = output_dim
    self.initializer = initializer

  def get_config(self):
    config = super().get_config()
    config.update({
      'output_dim': self.output_dim,
      'initializer': self.initializer  
    })
    return config


  def build(self, input_shape):
    self.resize_W = self.add_weight(
        shape = (input_shape[-1], self.output_dim),
        initializer = self.initializer,
        trainable = True
    )
    self.reshape_b = self.add_weight(
        shape = (self.output_dim,), 
        initializer = self.initializer, 
        trainable = True
    )
  
  def call(self, inputs):
    orig_x = inputs[:,13]
    cellstate = keras.layers.LeakyReLU()(tf.matmul(orig_x,self.resize_W)+self.reshape_b)
    cellstate = tf.reshape(cellstate, (tf.shape(inputs)[0],1,self.output_dim))
    return cellstate



class self_attention_decoder (keras.layers.Layer):
  def __init__ (self, block_num = 13, width = 4, initializer = 'random_normal', dropout = .4, **kwargs):
    super().__init__(**kwargs)
    self.block_num = block_num
    self.width = width
    self.modified_LSTM_layers = []
    self.dropout_layers = []
    self.initializer = initializer
    self.dropout = dropout
    for i in range(block_num):
      self.modified_LSTM_layers.append(modified_LSTM_block(output_dim = width, initializer = initializer))
      self.dropout_layers.append(keras.layers.Dropout(dropout))
    self.modified_LSTM_output_layer = modified_LSTM_output_block(output_dim = width, initializer = initializer)
    self.regression_layer = keras.layers.Dense(2)

  def get_config(self):
    config = super().get_config()
    config.update({
      'block_num': self.block_num,
      'width': self.width,
      'initializer': self.initializer,
      'dropout': self.dropout
    })
    return config

  def call (self, inputs):
    cell_state = inputs[:,14]
    input = inputs[:,:14]
    features = self.modified_LSTM_layers[0](tf.concat([[input[:,13]],[input[:,0]],[cell_state]],0))
    print(features)
    for i,modified_LSTM_layer in enumerate(self.modified_LSTM_layers[1:13]):
      features = modified_LSTM_layer(tf.concat([[input[:,13]],[input[:,i]],features],0))
      features = self.dropout_layers[i](features)
    output = self.modified_LSTM_output_layer(tf.concat([[input[:,13]],features],0))
    return output 

# In[ ]:


width = 16

inputs = keras.Input(shape = (14,22,))

inputs = keras.layers.Dropout(0)(inputs)
cellstate = initialize_cellstate(output_dim = width)(inputs)
positional_embedding = keras.layers.Dense(width, activation = 'tanh')(inputs)
positional_embedding = keras.layers.Dropout(0)(positional_embedding)
modified_LSTM_layer = self_attention_decoder(block_num = 13, width = width, dropout = 0)(tf.concat([positional_embedding,cellstate],1))
mixing_layer = keras.layers.Dense(width, activation = 'tanh')(modified_LSTM_layer)

dropout_layer = keras.layers.Dropout(0)(mixing_layer)

outputs = keras.layers.Dense(2)(dropout_layer)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer = 'Adam',
    loss = 'mse',
    metrics = [Euclidian_Distance_Error()]
)
def schedule(epoch, lr):
  if epoch < 50:
    return lr
  elif epoch >4000:
    if lr<=0.0001:
      return lr
    return lr*tf.math.exp(-0.004)
  elif lr<=0.0001:
    return lr
  else:
    return lr * tf.math.exp(-0.01)

callbacks_list = [
    keras.callbacks.TensorBoard(
        log_dir = f'/modified_LSTM'
    ),
    keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 100,
        restore_best_weights = True
    ), 
    # keras.callbacks.ModelCheckpoint(
    #     monitor = 'val_loss',
    #     save_best_only = True, 
    #     filepath = 'special_lstm.keras'
    # )
    #keras.callbacks.LearningRateScheduler(schedule)
]
model.summary()

model.fit(
    rnn_train_data, train_labels, 
    batch_size = 64, 
    epochs = 10000, 
    callbacks = callbacks_list,
    validation_data = [rnn_val_data, val_labels]
)

model.evaluate(rnn_val_data, val_labels)
%load_ext tensorboard
%tensorboard --logdir /modified_LSTM

# ##Multi-Attention Only

# In[ ]:


class scaled_dot_product_attention (keras.layers.Layer):
  def __init__ (self, num_heads, key_dims, **kwargs):
    super().__init__(**kwargs)
    self.num_heads = num_heads 
    self.key_dims = key_dims 
    self.d_q = layers.Dense(key_dims, activation = 'LeakyReLU')
    self.d_k = layers.Dense(key_dims, activation = 'LeakyReLU')
    self.d_v = layers.Dense(key_dims, activation = 'LeakyReLU')
    self.d_output = layers.Dense(key_dims, activation = 'LeakyReLU')

  def get_config(self):
    config = super().get_config()
    config.update({
        "num_heads" : self.num_heads,
        "key_dims": self.key_dims
    })
    return config

  def call (self, queries, keys, values):
    reshaped_q = self.d_q(queries)
    reshaped_q = tf.reshape(reshaped_q, shape=(tf.shape(reshaped_q)[0], tf.shape(reshaped_q)[1], self.num_heads, -1))
    reshaped_q = tf.transpose(reshaped_q, perm=(0, 2, 1, 3))

    reshaped_k = self.d_k(keys)
    reshaped_k = tf.reshape(reshaped_k, shape = (tf.shape(reshaped_k)[0],1,tf.shape(reshaped_k)[-1]))
    reshaped_k = tf.reshape(reshaped_k, shape=(tf.shape(reshaped_k)[0], tf.shape(reshaped_k)[1], self.num_heads, -1))
    reshaped_k = tf.transpose(reshaped_k, perm=(0, 2, 1, 3))

    reshaped_v = self.d_v(values)
    reshaped_v = tf.reshape(reshaped_v, shape = (tf.shape(reshaped_v)[0],1,tf.shape(reshaped_v)[-1]))
    reshaped_v = tf.reshape(reshaped_v, shape=(tf.shape(reshaped_v)[0], tf.shape(reshaped_v)[1], self.num_heads, -1))
    reshaped_v = tf.transpose(reshaped_v, perm=(0, 2, 1, 3))

    attention = tf.matmul(keras.activations.softmax(tf.matmul(reshaped_q, reshaped_k, transpose_b = True))/tf.sqrt(tf.cast(self.key_dims, "float32")),reshaped_v)
    reshaped_attention = tf.transpose(attention, perm=(0, 2, 1, 3))
    reshaped_attention = tf.reshape(reshaped_attention, shape=(tf.shape(reshaped_attention)[0], tf.shape(reshaped_attention)[1], self.key_dims))

    return self.d_output(reshaped_attention)

#########################################################################################################################################################

input = keras.Input(shape = (14,22,))


keysvalues = input[:,13,:]
queries = input[:,:13,:]



self_attention_output = scaled_dot_product_attention(8, 32)(queries, keysvalues, keysvalues)
combined_values = keras.layers.Concatenate(axis = 1)([tf.reshape(keras.layers.Dense(32)(keysvalues), shape = (tf.shape(keysvalues)[0],1,32)),self_attention_output])
combined_values = keras.layers.Flatten()(combined_values)

outputs = keras.layers.Dense(2)(combined_values)

model = keras.Model(inputs = [input], outputs = [outputs])

model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = 'mse',
    metrics = [Euclidian_Distance_Error()]
)
# def schedule(epoch, lr):
#   if epoch < 50:
#     return lr
#   elif lr<=0.0007:
#     return lr
#   else:
#     return lr * tf.math.exp(-0.00000001)

callbacks_list = [
    keras.callbacks.TensorBoard(
        log_dir = f'/modified_LSTM'
    ),
    keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 100,
        restore_best_weights = True
    ), 
    keras.callbacks.ModelCheckpoint(
        monitor = 'val_loss',
        save_best_only = True, 
        filepath = 'special_lstm.keras'
    ),
]

model.fit(
    [rnn_train_data], 
    [train_labels], 
    batch_size = 64, 
    epochs = 6500, 
    callbacks = callbacks_list,
    validation_data = [[rnn_val_data], val_labels]
)

model.evaluate([rnn_val_data,rnn_val_prev_output], val_labels)
%reload_ext tensorboard
%tensorboard --logdir /modified_LSTM

# ##Transformer
# 

# In[ ]:


class positional_embedding_layer (keras.layers.Layer):
  def __init__ (self, sequence_length, output_dim, **kwargs):
    super().__init__(**kwargs)
    self.sequence_length = sequence_length
    self.output_dim = output_dim
    self.embed_positions = keras.layers.Embedding(
        input_dim = sequence_length, output_dim = output_dim
    )
    self.resize_layer = keras.layers.Dense(output_dim, activation = 'LeakyReLU')

  def get_config(self):
    config = super().get_config()
    config.update({
        'sequence_length': self.sequence_length,
        'output_dim': self.output_dim
    })
    return config 

  def call(self, inputs):
    resized_inputs = self.resize_layer(inputs)
    length = tf.shape(resized_inputs)[-2]
    positions = tf.range(start=0, limit=length, delta=1)
    position_embeddings = self.embed_positions(positions)
    embedded_vector = position_embeddings+resized_inputs
    return embedded_vector
  
  def compute_mask(self, inputs, mask = None):
    return tf.math.not_equal(inputs, 0)

class self_attention_encoder (keras.layers.Layer):
  def __init__ (self, output_size, dense_size, num_heads, dropout = 0, **kwargs):
    super().__init__(**kwargs)
    self.output_size = output_size
    self.dense_size = dense_size
    self.num_heads = num_heads
    self.multi_attention = keras.layers.MultiHeadAttention(
        num_heads = num_heads, 
        key_dim = output_size, 
    )
    self.dense_layer = keras.Sequential([
        keras.layers.Dense(dense_size, activation = 'relu'),
        keras.layers.Dense(output_size)
    ])
    self.attention_normalization = keras.layers.LayerNormalization()
    self.dense_normalization = keras.layers.LayerNormalization()
    self.attention_dropout = keras.layers.Dropout(dropout)
    self.dense_dropout = keras.layers.Dropout(dropout)
    self.dropout = dropout

  def get_config(self):
    config = super().get_config()
    config.update({
        'output_size': self.output_size,
        'dense_size': self.dense_size,
        'num_heads': self.num_heads,
        'dropout': self.dropout
    })
    return config

  def call(self, inputs, mask = None):
    if mask is not None:
      mask = mask[:, tf.newaxis, :]
    self_attention_output = self.multi_attention(inputs, inputs, attention_mask = None)
    residual_normalization_output = self.attention_normalization(self_attention_output+inputs)
    residual_normalization_output = self.attention_dropout(residual_normalization_output)
    residual_normalization_output = tf.concat(residual_normalization_output,axis = 1)
    dense_output = self.dense_layer(residual_normalization_output)
    dense_normalization_output = self.dense_normalization(dense_output+residual_normalization_output)
    dense_normalization_output = self.dense_dropout(dense_normalization_output)
    return dense_normalization_output

  def compute_mask(self, inputs, mask = None):
    return tf.math.not_equal(inputs, 0)

class cross_attention_encoder (keras.layers.Layer):
  def __init__ (self, output_size, dense_size, num_heads, dropout = 0, **kwargs):
    super().__init__(**kwargs)
    self.output_size = output_size
    self.dense_size = dense_size
    self.num_heads = num_heads
    self.multi_attention = keras.layers.MultiHeadAttention(
        num_heads = num_heads, 
        key_dim = output_size, 
    )
    self.dense_layer = keras.Sequential([
        keras.layers.Dense(dense_size, activation = 'relu'),
        keras.layers.Dense(output_size)
    ])
    self.attention_normalization = keras.layers.LayerNormalization()
    self.dense_normalization = keras.layers.LayerNormalization()
    self.attention_dropout = keras.layers.Dropout(dropout)
    self.dense_dropout = keras.layers.Dropout(dropout)
    self.dropout = dropout

  def get_config(self):
    config = super().get_config()
    config.update({
        'output_size': self.output_size,
        'dense_size': self.dense_size,
        'num_heads': self.num_heads,
        'dropout': self.dropout
    })
    return config

  def call(self, inputs, mask = None):
    if mask is not None:
      mask = mask[:, tf.newaxis, :]
    keys = inputs[:,:13,:]
    queries = tf.tile(inputs[:,13:14,:],[1,13,1])
    self_attention_output = self.multi_attention(queries,keys,attention_mask = None)
    residual_normalization_output = self.attention_normalization(tf.concat([self_attention_output,inputs[:,13:14,:]], axis = 1)+inputs)
    residual_normalization_output = self.attention_dropout(residual_normalization_output)
    dense_output = self.dense_layer(residual_normalization_output)
    dense_normalization_output = self.dense_normalization(dense_output+residual_normalization_output)
    dense_normalization_output = self.dense_dropout(dense_normalization_output)
    return dense_normalization_output

  def compute_mask(self, inputs, mask = None):
    return tf.math.not_equal(inputs, 0)

class self_attention_decoder(keras.layers.Layer):
  def __init__(self, output_size, dense_size, num_heads, dropout = 0, **kwargs):
    super().__init__(**kwargs)
    self.output_size = output_size
    self.dense_size = dense_size
    self.num_heads = num_heads
    self.dropout = dropout
    self.attention1 = keras.layers.MultiHeadAttention(
        num_heads = num_heads, 
        key_dim = output_size
    )
    self.normalization1 = keras.layers.LayerNormalization()
    self.attention2 = keras.layers.MultiHeadAttention(
        num_heads = num_heads, 
        key_dim = output_size
    )
    self.normalization2 = keras.layers.LayerNormalization()
    self.dense = keras.Sequential([
        keras.layers.Dense(dense_size, activation = 'relu'),
        keras.layers.Dense(output_size)
    ])
    self.normalization3 = keras.layers.LayerNormalization()
    self.mixing = keras.layers.Dense(dense_size, activation = 'relu')
    self.flatten = keras.layers.Flatten()
    
  def get_config(self):
    config = super().get_config()
    config.update({
        'output_size': self.output_size,
        'dense_size': self.dense_size,
        'num_heads': self.num_heads,
        'dropout': self.dropout
    })
    return config

  def call (self, inputs, prev_output):
    previous_output_attention = self.attention1(prev_output, prev_output, attention_mask = None)
    previous_output_attention = self.normalization1(previous_output_attention+prev_output)
    combined_attention = self.attention2(previous_output_attention,inputs,attention_mask = None)
    combined_attention = self.normalization2(combined_attention+previous_output_attention)
    mixing_layer = self.dense(combined_attention)
    mixing_layer = self.normalization3(mixing_layer+combined_attention)
    mixing_layer = self.mixing(mixing_layer)
    mixing_layer = self.flatten(mixing_layer)
    return mixing_layer

# In[ ]:


width = 16
inputs = keras.Input(shape = (14,22,))
previous_output = keras.Input(shape = (13,2,))

previous_output = keras.layers.Dropout(.1)(previous_output)
embedded_positions = positional_embedding_layer(22,width)(inputs)
embedded_previous_output = keras.layers.Dense(width,activation = 'LeakyReLU')(previous_output)
# transformer_encoder = cross_attention_encoder(width, width,2)(embedded_positions)
# transformer_encoder = keras.layers.Dropout(.0)(transformer_encoder)
transformer_encoder = self_attention_encoder(width,width,8)(embedded_positions)
transformer_encoder = keras.layers.Dropout(0.1)(transformer_encoder)
transformer_encoder = self_attention_encoder(width,width,8)(transformer_encoder)
transformer_encoder = keras.layers.Dropout(0.1)(transformer_encoder)
transformer_decoder = self_attention_decoder(width,width,8)(transformer_encoder,embedded_previous_output)
transformer_decoder = keras.layers.Dropout(0.1)(transformer_decoder)

outputs = keras.layers.Dense(2)(transformer_decoder)

model = keras.Model(inputs = [inputs, previous_output], outputs = [outputs])

model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = 'mse',
    metrics = [Euclidian_Distance_Error()]
)
# def schedule(epoch, lr):
#   if epoch < 50:
#     return lr
#   elif lr<=0.0007:
#     return lr
#   else:
#     return lr * tf.math.exp(-0.00000001)

callbacks_list = [
    keras.callbacks.TensorBoard(
        log_dir = f'/modified_LSTM'
    ),
    keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 100,
        restore_best_weights = True
    ), 
    keras.callbacks.ModelCheckpoint(
        monitor = 'val_loss',
        save_best_only = True, 
        filepath = 'special_lstm.keras'
    ),
]

model.fit(
    [rnn_train_data, rnn_prev_output], 
    [train_labels], 
    batch_size = 64, 
    epochs = 6500, 
    callbacks = callbacks_list,
    validation_data = [[rnn_val_data,rnn_val_prev_output], val_labels]
)

model.evaluate([rnn_val_data,val_prev_output], val_labels)
%reload_ext tensorboard
%tensorboard --logdir /modified_LSTM

# ##Transformer 2

# In[ ]:


class positional_embedding_layer (keras.layers.Layer):
  def __init__ (self, sequence_length, output_dim, **kwargs):
    super().__init__(**kwargs)
    self.sequence_length = sequence_length
    self.output_dim = output_dim
    self.embed_positions = keras.layers.Embedding(
        input_dim = sequence_length, output_dim = output_dim
    )
    self.resize_layer = keras.layers.Dense(output_dim, activation = 'LeakyReLU')

  def get_config(self):
    config = super().get_config()
    config.update({
        'sequence_length': self.sequence_length,
        'output_dim': self.output_dim
    })
    return config 

  def call(self, inputs):
    resized_inputs = self.resize_layer(inputs)
    length = tf.shape(resized_inputs)[-2]
    positions = tf.range(start=0, limit=length, delta=1)
    position_embeddings = self.embed_positions(positions)
    embedded_vector = position_embeddings+resized_inputs
    return embedded_vector
  
  def compute_mask(self, inputs, mask = None):
    return tf.math.not_equal(inputs, 0)

class self_attention_encoder (keras.layers.Layer):
  def __init__ (self, output_size, dense_size, num_heads, dropout = 0, **kwargs):
    super().__init__(**kwargs)
    self.output_size = output_size
    self.dense_size = dense_size
    self.num_heads = num_heads
    self.multi_attention = keras.layers.MultiHeadAttention(
        num_heads = num_heads, 
        key_dim = output_size, 
    )
    self.dense_layer = keras.Sequential([
        keras.layers.Dense(dense_size, activation = 'relu'),
        keras.layers.Dense(output_size)
    ])
    self.attention_normalization = keras.layers.LayerNormalization()
    self.dense_normalization = keras.layers.LayerNormalization()
    self.attention_dropout = keras.layers.Dropout(dropout)
    self.dense_dropout = keras.layers.Dropout(dropout)
    self.dropout = dropout

  def get_config(self):
    config = super().get_config()
    config.update({
        'output_size': self.output_size,
        'dense_size': self.dense_size,
        'num_heads': self.num_heads,
        'dropout': self.dropout
    })
    return config

  def call(self, inputs, mask = None):
    if mask is not None:
      mask = mask[:, tf.newaxis, :]
    self_attention_output = self.multi_attention(inputs, inputs, attention_mask = None)
    residual_normalization_output = self.attention_normalization(self_attention_output+inputs)
    residual_normalization_output = self.attention_dropout(residual_normalization_output)
    residual_normalization_output = tf.concat(residual_normalization_output,axis = 1)
    dense_output = self.dense_layer(residual_normalization_output)
    dense_normalization_output = self.dense_normalization(dense_output+residual_normalization_output)
    dense_normalization_output = self.dense_dropout(dense_normalization_output)
    return dense_normalization_output

  def compute_mask(self, inputs, mask = None):
    return tf.math.not_equal(inputs, 0)

# In[ ]:


width = 16
inputs = keras.Input(shape = (14,22,))

embedded_positions = positional_embedding_layer(22,width)(inputs)
transformer_encoder = self_attention_encoder(width,width,16)(embedded_positions)
transformer_encoder = self_attention_encoder(width,width,8)(embedded_positions)
print(transformer_encoder[:,13,:])
mixing_layer = keras.layers.Dense(width, activation = 'LeakyReLU')(transformer_encoder[:,13,:])
outputs = keras.layers.Dense(2)(mixing_layer)

model = keras.Model(inputs = inputs, outputs = outputs)

model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = 'mse',
    metrics = [Euclidian_Distance_Error()]
)
# def schedule(epoch, lr):
#   if epoch < 50:
#     return lr
#   elif lr<=0.0007:
#     return lr
#   else:
#     return lr * tf.math.exp(-0.00000001)

callbacks_list = [
    keras.callbacks.TensorBoard(
        log_dir = f'/modified_LSTM'
    ),
    keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 4000,
        restore_best_weights = True
    ), 
    # keras.callbacks.ModelCheckpoint(
    #     monitor = 'val_loss',
    #     save_best_only = True, 
    #     filepath = 'special_lstm.keras'
    # ),
    # keras.callbacks.LearningRateScheduler(schedule)
]

model.fit(
    [rnn_train_data], 
    [train_labels], 
    batch_size = 64, 
    epochs = 400, 
    callbacks = callbacks_list,
    validation_data = [rnn_val_data, val_labels]
)

model.evaluate([rnn_val_data,val_prev_output], val_labels)
%reload_ext tensorboard
%tensorboard --logdir /modified_LSTM

# ##Transformer 3

# In[ ]:


class positional_embedding_layer (keras.layers.Layer):
  def __init__ (self, sequence_length, width, **kwargs):
    super().__init__(**kwargs)
    self.sequence_length = sequence_length
    self.output_dim = width
    self.embed_positions = keras.layers.Embedding(
        input_dim = sequence_length, output_dim = width
    )

  def call(self, inputs):
    length = tf.shape(inputs)[-2]
    positions = tf.range(start=0, limit=length, delta=1)
    position_embeddings = self.embed_positions(positions)
    embedded_vector = position_embeddings+inputs
    return embedded_vector
  
  def compute_mask(self, inputs, mask = None):
    return tf.math.not_equal(inputs, 0)


# In[ ]:


 width = 8
dropout = 0
num_heads = 4

querieskeys = keras.Input(shape = (14,22,))
values = keras.Input(shape=(13,2,))

queries = querieskeys[:,13,:]
keys = querieskeys[:,:13,:]

queries = keras.layers.Dropout(dropout)(queries)
keys = keras.layers.Dropout(dropout)(keys)
values = keras.layers.Dropout(dropout)(values)

positioned_keys = positional_embedding_layer(13,22)(keys)
positioned_values = positional_embedding_layer(13,2)(values)
queries = keras.layers.RepeatVector(13)(queries)

attention_output = keras.layers.MultiHeadAttention(num_heads = num_heads, key_dim = width)(queries, positioned_values, positioned_keys)

pooling_layer = keras.layers.AveragePooling1D(pool_size = 13,strides = None)(attention_output)
pooling_layer = keras.layers.Reshape([22])(pooling_layer)
pooling_layer = keras.layers.Dropout(dropout)(pooling_layer)

mixing_layer = keras.layers.Dense(width/2, activation = 'LeakyReLU')(pooling_layer)
mixing_layer = keras.layers.Dropout(dropout)(mixing_layer)


outputs = keras.layers.Dense(2)(mixing_layer)

model = keras.Model(inputs = [querieskeys,values,], outputs = outputs)

model.compile(
    optimizer = tf.keras.optimizers.Adam(0.01),
    loss = 'mse',
    metrics = [Euclidian_Distance_Error()]
)
def schedule(epoch, lr):
  if epoch < 10:
    return lr
  elif lr<=0.0007:
    return lr
  else:
    return lr * tf.math.exp(-0.05)

callbacks_list = [
    keras.callbacks.TensorBoard(
        log_dir = f'/modified_LSTM'
    ),
    keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 100,
        restore_best_weights = True
    ), 
    # keras.callbacks.ModelCheckpoint(
    #     monitor = 'val_loss',
    #     save_best_only = True, 
    #     filepath = 'special_lstm.keras'
    # ),
    keras.callbacks.LearningRateScheduler(schedule)
]
model.summary()
model.fit(
    [rnn_train_data,rnn_prev_output], 
    [train_labels], 
    batch_size = 64, 
    epochs = 8000, 
    callbacks = callbacks_list,
    validation_data = [[rnn_val_data,rnn_val_prev_output], val_labels]
)

model.evaluate([rnn_val_data,rnn_val_prev_output], val_labels)
%reload_ext tensorboard
%tensorboard --logdir /modified_LSTM

# #MultiHeaded

# ##Data Config

# In[ ]:


vae_train_data = rnn_train_data[:,:,:6]
eyeR_train_data = rnn_train_data[:,:,6:14]
eyeL_train_data = rnn_train_data[:,:,14:]

vae_val_data = rnn_val_data[:,:,:6]
eyeR_val_data = rnn_val_data[:,:,6:14]
eyeL_val_data = rnn_val_data[:,:,14:]
print(tf.shape(eyeR_val_data))

# ##Basic Dense

# In[ ]:



width = 32
layers = 5
vae_input = keras.Input(shape = (14,6,))
eyeR_input = keras.Input(shape = (14,8,))
eyeL_input = keras.Input(shape = (14,8,))

flat_eyeR_input = keras.layers.Flatten()(eyeR_input)
flat_eyeL_input = keras.layers.Flatten()(eyeL_input)
flat_vae_input = keras.layers.Flatten()(vae_input)
eyeR_dense = keras.layers.Dense(width, activation = 'LeakyReLU')(flat_eyeR_input)
eyeL_dense = keras.layers.Dense(width, activation = 'LeakyReLU')(flat_eyeL_input)
for i in range (layers):
  eyeR_dense = keras.layers.Dense(width, activation = 'LeakyReLU')(eyeR_dense)
  eyeL_dense = keras.layers.Dense(width, activation = 'LeakyReLU')(eyeL_dense)

mixing_layer = keras.layers.Concatenate(axis = -1)([flat_vae_input, eyeR_dense, eyeL_dense])

output = keras.layers.Dense(2)(mixing_layer)

model = keras.Model([vae_input, eyeR_input, eyeL_input], output)

model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = 'mse',
    metrics = [Euclidian_Distance_Error()]
)
def schedule(epoch, lr):
  if epoch < 50:
    return lr
  elif lr<=0.001:
    return lr
  else:
    return lr * tf.math.exp(-0.01)

callbacks_list = [

    keras.callbacks.ModelCheckpoint(
        filepath = f'1024x3+l2d4.keras',
        monitor = 'val_loss',
        save_best_only = True
    ),
    keras.callbacks.TensorBoard(
        log_dir = f'/1024x3'
    ),
    keras.callbacks.EarlyStopping(
        monitor = 'val_distance',
        patience = 100, 
        restore_best_weights = True,
    ), 
    keras.callbacks.LearningRateScheduler(schedule)
]
model.summary()
model.fit(
    [vae_train_data, eyeR_train_data, eyeL_train_data], train_labels,
    epochs = 20000,
    batch_size = 64,
    callbacks = callbacks_list,
    validation_data = ([vae_val_data, eyeR_val_data, eyeL_val_data], val_labels)
)

model.evaluate([vae_val_data, eyeR_val_data, eyeL_val_data], val_labels)

%load_ext tensorboard
%tensorboard --logdir /1024x3

# In[ ]:


model.evaluate([vae_val_data, eyeR_val_data, eyeL_val_data], val_labels)

# ##Modified LSTM

# In[ ]:


class positional_embedding_layer (keras.layers.Layer):
  def __init__ (self, sequence_length, output_dim, **kwargs):
    super().__init__(**kwargs)
    self.sequence_length = sequence_length
    self.output_dim = output_dim
    self.embed_positions = keras.layers.Embedding(
        input_dim = sequence_length, output_dim = output_dim
    )
    self.resize_layer = keras.layers.Dense(output_dim, activation = 'LeakyReLU')


  def get_config(self):
    config = super().get_config()
    config.update({
        'sequence_length': self.sequence_length,
        'output_dim': self.output_dim
    })
    return config 

  def call(self, inputs):
    length = tf.shape(inputs)[-2]
    positions = tf.range(start=0, limit=length, delta=1)
    position_embeddings = self.embed_positions(positions)
    resized_inputs = self.resize_layer(inputs)
    embedded_vector = position_embeddings+resized_inputs
    return embedded_vector
  
  def compute_mask(self, inputs, mask = None):
    return tf.math.not_equal(inputs, 0)

class modified_LSTM_block (keras.layers.Layer):
  def __init__ (self, output_dim = 4, initializer = 'random_normal', **kwargs):
    super().__init__(**kwargs)
    self.initializer = initializer
    self.throughput = output_dim

  def get_config(self):
    config = super().get_config()
    config.update({
      'output_dim': self.throughput,
      'initializer': self.initializer  
    })
    return config


  def build (self, input_shape):
    #weights and biases for forget gate
    self.forget_W = self.add_weight(
        shape = (input_shape[-1], self.throughput), 
        initializer = self.initializer, 
        trainable = True
    )
    self.forget_U = self.add_weight(
        shape = (input_shape[-1], self.throughput), 
        initializer = self.initializer, 
        trainable = True
    )
    self.forget_b = self.add_weight(
        shape = (self.throughput,), 
        initializer = self.initializer, 
        trainable = True
    )
    #weights and biases for input gate
    self.input_W = self.add_weight(
        shape = (input_shape[-1], self.throughput), 
        initializer = self.initializer, 
        trainable = True
    )
    self.input_U = self.add_weight(
        shape = (input_shape[-1], self.throughput), 
        initializer = self.initializer, 
        trainable = True
    )
    self.input_b = self.add_weight(
        shape = (self.throughput,), 
        initializer = self.initializer, 
        trainable = True
    )
    #weights and biases for updating cell state 
    self.cell_W = self.add_weight(
        shape = (input_shape[-1], self.throughput), 
        initializer = self.initializer, 
        trainable = True
    )
    self.cell_U = self.add_weight(
        shape = (input_shape[-1], self.throughput), 
        initializer = self.initializer, 
        trainable = True
    )
    self.cell_b = self.add_weight(
        shape = (self.throughput,), 
        initializer = self.initializer, 
        trainable = True
    )

  def call (self, inputs):
    hidden_state_1 = tf.expand_dims(inputs[0],0)
    hidden_state = tf.expand_dims(inputs[1],0)
    cell_state = tf.expand_dims(inputs[2],0)
    forget_vector = tf.keras.activations.sigmoid(tf.matmul(hidden_state,self.forget_W)+tf.matmul(hidden_state_1,self.forget_U)+self.forget_b)
    input_vector = tf.keras.activations.sigmoid(tf.matmul(hidden_state,self.input_W)+tf.matmul(hidden_state_1,self.input_U)+self.input_b)
    cell_update_vector = tf.matmul(hidden_state,self.cell_W)+tf.matmul(hidden_state_1,self.cell_U)+self.cell_b
    cell_update_vector = tf.keras.layers.LeakyReLU()(cell_update_vector)
    new_cell = input_vector*cell_update_vector
    return cell_state*forget_vector+new_cell


class modified_LSTM_output_block(keras.layers.Layer):
  def __init__ (self, output_dim = 4, initializer = 'random_normal', **kwargs):
    super().__init__(**kwargs)
    self.throughput = output_dim
    self.initializer = initializer
  
  def get_config(self):
    config = super().get_config()
    config.update({
      'output_dim': self.throuhgput,
      'initializer': self.initializer  
    })
    return config


  def build (self, input_shape):
    self.input_W = self.add_weight(
        shape = (input_shape[-1], self.throughput), 
        initializer = self.initializer, 
        trainable = True
    )
    self.input_b = self.add_weight(
        shape = (self.throughput,), 
        initializer = self.initializer, 
        trainable = True
    )
    self.cell_W = self.add_weight(
        shape = (input_shape[-1], self.throughput), 
        initializer = self.initializer, 
        trainable = True
    )
  
  def call (self, inputs):
    hidden_state = inputs[0]
    cell_state = inputs[1]
    return (tf.matmul(hidden_state, self.input_W)+self.input_b)+keras.layers.LeakyReLU()(tf.matmul(cell_state,self.cell_W))

class initialize_cellstate(keras.layers.Layer):
  def __init__(self, output_dim = 4, initializer = 'random_normal', **kwargs):
    super().__init__(**kwargs)
    self.output_dim = output_dim
    self.initializer = initializer

  def get_config(self):
    config = super().get_config()
    config.update({
      'output_dim': self.output_dim,
      'initializer': self.initializer  
    })
    return config


  def build(self, input_shape):
    self.resize_W = self.add_weight(
        shape = (input_shape[-1], self.output_dim),
        initializer = self.initializer,
        trainable = True
    )
    self.reshape_b = self.add_weight(
        shape = (self.output_dim,), 
        initializer = self.initializer, 
        trainable = True
    )
  
  def call(self, inputs):
    orig_x = inputs[:,13]
    cellstate = keras.layers.LeakyReLU()(tf.matmul(orig_x,self.resize_W)+self.reshape_b)
    cellstate = tf.reshape(cellstate, (tf.shape(inputs)[0],1,self.output_dim))
    return cellstate



class self_attention_decoder (keras.layers.Layer):
  def __init__ (self, block_num = 13, width = 4, initializer = 'random_normal', dropout = .4, **kwargs):
    super().__init__(**kwargs)
    self.block_num = block_num
    self.width = width
    self.modified_LSTM_layers = []
    self.dropout_layers = []
    self.initializer = initializer
    self.dropout = dropout
    for i in range(block_num):
      self.modified_LSTM_layers.append(modified_LSTM_block(output_dim = width, initializer = initializer))
      self.dropout_layers.append(keras.layers.Dropout(dropout))
    self.modified_LSTM_output_layer = modified_LSTM_output_block(output_dim = width, initializer = initializer)
    self.regression_layer = keras.layers.Dense(2)

  def get_config(self):
    config = super().get_config()
    config.update({
      'block_num': self.block_num,
      'width': self.width,
      'initializer': self.initializer,
      'dropout': self.dropout
    })
    return config

  def call (self, inputs, cell_state):
    features = self.modified_LSTM_layers[0](tf.concat([inputs[:,13],inputs[:,0],cell_state],0))
    print(features)
    for i,modified_LSTM_layer in enumerate(self.modified_LSTM_layers[1:13]):
      features = modified_LSTM_layer(tf.concat([[inputs[:,13]],[inputs[:,i]],features],0))
      features = self.dropout_layers[i](features)
    output = self.modified_LSTM_output_layer(tf.concat([[inputs[:,13]],features],0))
    return output 

# In[ ]:


import keras.layers as layers

width = 8
layers = 2
vae_input = keras.Input(shape = (14,6,))
eyeR_input = keras.Input(shape = (14,8,))
eyeL_input = keras.Input(shape = (14,8,))

vae_dense = keras.layers.Dense(width, activation = 'LeakyReLU')(vae_input)
eyeR_dense = keras.layers.Dense(width, activation = 'LeakyReLU')(eyeR_input)
eyeL_dense = keras.layers.Dense(width, activation = 'LeakyReLU')(eyeL_input)

vae_cellstate = initialize_cellstate(output_dim = width)(vae_input)
eyeR_cellstate = initialize_cellstate(output_dim = width)(eyeR_input)
eyeL_cellstate = initialize_cellstate(output_dim = width)(eyeL_input)
print(vae_cellstate)
vae_dense = self_attention_decoder(width = width, dropout = 0)(vae_dense, vae_cellstate)
eyeR_dense = self_attention_decoder(width = width, dropout = 0)(eyeR_dense, eyeR_cellstate)
eyeL_dense = self_attention_decoder(width = width, dropout = 0)(eyeL_dense, eyeL_cellstate)

for i in range (layers):
  eyeR_dense = keras.layers.Dense(width, activation = 'LeakyReLU')(eyeR_dense)
  eyeL_dense = keras.layers.Dense(width, activation = 'LeakyReLU')(eyeL_dense)

mixing_layer = keras.layers.Concatenate(axis = -1)([vae_dense, eyeR_dense, eyeL_dense])

output = keras.layers.Dense(2)(mixing_layer)

model = keras.Model([vae_input, eyeR_input, eyeL_input], output)

model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = 'mse',
    metrics = [Euclidian_Distance_Error()]
)
def schedule(epoch, lr):
  if epoch < 50:
    return lr
  elif lr<=0.001:
    return lr
  else:
    return lr * tf.math.exp(-0.01)

callbacks_list = [

    keras.callbacks.ModelCheckpoint(
        filepath = f'1024x3+l2d4.keras',
        monitor = 'val_loss',
        save_best_only = True
    ),
    keras.callbacks.TensorBoard(
        log_dir = f'/1024x3'
    ),
    keras.callbacks.EarlyStopping(
        monitor = 'val_distance',
        patience = 100, 
        restore_best_weights = True,
    ), 
    keras.callbacks.LearningRateScheduler(schedule)
]
model.summary()
model.fit(
    [vae_train_data, eyeR_train_data, eyeL_train_data], train_labels,
    epochs = 20000,
    batch_size = 64,
    callbacks = callbacks_list,
    validation_data = ([vae_val_data, eyeR_val_data, eyeL_val_data], val_labels)
)

model.evaluate(val_data, val_labels)

%load_ext tensorboard
%tensorboard --logdir /1024x3

# ##Transformer

# In[ ]:


class positional_embedding_layer (keras.layers.Layer):
  def __init__ (self, sequence_length, output_dim, **kwargs):
    super().__init__(**kwargs)
    self.sequence_length = sequence_length
    self.output_dim = output_dim
    self.embed_positions = keras.layers.Embedding(
        input_dim = sequence_length, output_dim = output_dim
    )
    self.resize_layer = keras.layers.Dense(output_dim, activation = 'LeakyReLU')

  def get_config(self):
    config = super().get_config()
    config.update({
        'sequence_length': self.sequence_length,
        'output_dim': self.output_dim
    })
    return config 

  def call(self, inputs):
    resized_inputs = self.resize_layer(inputs)
    length = tf.shape(resized_inputs)[-2]
    positions = tf.range(start=0, limit=length, delta=1)
    position_embeddings = self.embed_positions(positions)
    embedded_vector = position_embeddings+resized_inputs
    return embedded_vector
  
  def compute_mask(self, inputs, mask = None):
    return tf.math.not_equal(inputs, 0)


class cross_attention_encoder (keras.layers.Layer):
  def __init__ (self, output_size, dense_size, num_heads, dropout = 0, **kwargs):
    super().__init__(**kwargs)
    self.output_size = output_size
    self.dense_size = dense_size
    self.num_heads = num_heads
    self.multi_attention = keras.layers.MultiHeadAttention(
        num_heads = num_heads, 
        key_dim = output_size, 
    )
    self.dense_layer = keras.Sequential([
        keras.layers.Dense(dense_size, activation = 'relu'),
        keras.layers.Dense(output_size)
    ])
    self.attention_normalization = keras.layers.LayerNormalization()
    self.dense_normalization = keras.layers.LayerNormalization()
    self.attention_dropout = keras.layers.Dropout(dropout)
    self.dense_dropout = keras.layers.Dropout(dropout)
    self.dropout = dropout

  def get_config(self):
    config = super().get_config()
    config.update({
        'output_size': self.output_size,
        'dense_size': self.dense_size,
        'num_heads': self.num_heads,
        'dropout': self.dropout
    })
    return config

  def call(self, inputs, mask = None):
    if mask is not None:
      mask = mask[:, tf.newaxis, :]
    keys = inputs[:,:13,:]
    queries = tf.tile(inputs[:,13:14,:],[1,13,1])
    self_attention_output = self.multi_attention(queries,keys,attention_mask = None)
    residual_normalization_output = self.attention_normalization(tf.concat([self_attention_output,inputs[:,13:14,:]], axis = 1)+inputs)
    residual_normalization_output = self.attention_dropout(residual_normalization_output)
    dense_output = self.dense_layer(residual_normalization_output)
    dense_normalization_output = self.dense_normalization(dense_output+residual_normalization_output)
    dense_normalization_output = self.dense_dropout(dense_normalization_output)
    return dense_normalization_output

###################################################################################################################################################################

width = 16
layers = 2
heads = 4
dropout = 0.1

vae_input = keras.Input(shape = (14,6,))
eyeR_input = keras.Input(shape = (14,8,))
eyeL_input = keras.Input(shape = (14,8,))
vae_input = keras.layers.Dropout(dropout)(vae_input)
eyeR_input = keras.layers.Dropout(dropout)(eyeR_input)
eyeL_input = keras.layers.Dropout(dropout)(eyeL_input)

vae_dense = keras.layers.Dense(width, activation = 'LeakyReLU')(vae_input)
eyeR_dense = positional_embedding_layer(14, width)(eyeR_input)
eyeL_dense = positional_embedding_layer(14, width)(eyeL_input)
vae_dense = keras.layers.Dropout(dropout)(vae_dense)
eyeR_dense = keras.layers.Dropout(dropout)(eyeR_dense)
eyeL_dense = keras.layers.Dropout(dropout)(eyeL_dense)

for i in range (layers):
  eyeR_dense = cross_attention_encoder(width, width, heads)(eyeR_dense)
  eyeL_dense = cross_attention_encoder(width,width,heads)(eyeL_dense)
  eyeR_dense = keras.layers.Dropout(dropout)(eyeR_dense)
  eyeL_dense = keras.layers.Dropout(dropout)(eyeL_dense)



flatten_vae = keras.layers.Flatten()(vae_dense)
flatten_eyeR = keras.layers.Flatten()(eyeR_dense)
flatten_eyeL = keras.layers.Flatten()(eyeL_dense)
mixing_layer = keras.layers.Concatenate(axis = -1)([flatten_vae, flatten_eyeR, flatten_eyeL])

output = keras.layers.Dense(2)(mixing_layer)

model = keras.Model([vae_input, eyeR_input, eyeL_input], output)

model.compile(
    optimizer = tf.keras.optimizers.Adam(0.005),
    loss = 'mse',
    metrics = [Euclidian_Distance_Error()]
)
def schedule(epoch, lr):
  if epoch < 50:
    return lr
  elif lr<=0.001:
    return 0.001
  else:
    return lr * tf.math.exp(-0.01)

callbacks_list = [

    keras.callbacks.ModelCheckpoint(
        filepath = f'1024x3+l2d4.keras',
        monitor = 'val_loss',
        save_best_only = True
    ),
    keras.callbacks.TensorBoard(
        log_dir = f'/1024x3'
    ),
    keras.callbacks.EarlyStopping(
        monitor = 'val_distance',
        patience = 100, 
        restore_best_weights = True,
    ), 
    keras.callbacks.LearningRateScheduler(schedule)
]
model.summary()
model.fit(
    [vae_train_data, eyeR_train_data, eyeL_train_data], train_labels,
    epochs = 20000,
    batch_size = 64,
    callbacks = callbacks_list,
    validation_data = ([vae_val_data, eyeR_val_data, eyeL_val_data], val_labels)
)

model.evaluate([vae_val_data, eyeR_val_data, eyeL_val_data], val_labels)

%load_ext tensorboard
%tensorboard --logdir /1024x3

# 
# #Final Model Training

# 1024x3 Model with RMSprop for 4000 epochs

# In[ ]:


inputs = keras.Input(shape = (14,22,))
features = keras.layers.Flatten()(inputs)
features = keras.layers.Dense(1024, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.02))(features)
features = keras.layers.Dropout(0.4)(features)
features = keras.layers.Dense(1024, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.02))(features)
features = keras.layers.Dropout(0.4)(features)
features = keras.layers.Dense(1024, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.02))(features)
features = keras.layers.Dropout(0.4)(features)
outputs = keras.layers.Dense(2)(features)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer = 'rmsprop',
    loss = 'mse',
    metrics = ['mae','accuracy']
)

model.fit(
    train_data, train_labels,
    epochs = 4000,
    batch_size = 64,
)
model.evaluate(val_data, val_labels)
model.save('m1024-2048-1024.keras')


# 1024-2048-1024 model with Adam and 5500 epochs 

# In[ ]:


inputs = keras.Input(shape = (14,22,))
features = keras.layers.Flatten()(inputs)
features = keras.layers.Dense(1024, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.03))(features)
features = keras.layers.Dropout(0.4)(features)
features = keras.layers.Dense(2048, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.03))(features)
features = keras.layers.Dropout(0.4)(features)
features = keras.layers.Dense(1024, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.03))(features)
features = keras.layers.Dropout(0.4)(features)
outputs = keras.layers.Dense(2)(features)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer = keras.optimizers.Adam(learning_rate = 0.001),
    loss = 'mse',
    metrics = ['mae','accuracy']
)

model.fit(
    all_train_data, all_train_labels,
    epochs = 5500,
    batch_size = 64,
)
model.evaluate(test_data,test_labels)
model.save('m1024-2048-1024.keras')


# Testing Models

# In[ ]:


#feed forward network with 3 1024 keras.layers 
ff3x1024 = keras.models.load_model('/content/drive/MyDrive/DNN Gaze Prediction Best So Far/1024x3.keras')
ff3x1024.evaluate(val_data,val_labels)
#feed forward network with 1024-2048-1024 keras.layers
ff121 = keras.models.load_model('/content/drive/MyDrive/DNN Gaze Prediction Best So Far/m1024-2048-1024.keras')
ff121.evaluate(val_data, val_labels)

# In[ ]:



