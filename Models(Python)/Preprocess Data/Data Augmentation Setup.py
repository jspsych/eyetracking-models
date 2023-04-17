#!/usr/bin/env python
# coding: utf-8

# Add necessary imports

# In[ ]:


import numpy as np
import os
import json
import tensorflow as tf
import random
!pip install mediapipe
import mediapipe as mp
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')

# Configure MediaPipe FaceMesh

# In[ ]:


mp_face_mesh = mp.solutions.face_mesh

# Grab data from training model

# In[ ]:


json_path = '/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/json/'
all_json_files = os.listdir(json_path)

json_data = {}
for filename in all_json_files:
  with open('/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/json/'+filename, 'r') as file:
    s_data = json.load(file)
    json_data = {**json_data, **s_data}

# In[ ]:


left_eye_point = set(sum(mp_face_mesh.FACEMESH_LEFT_EYE, ()))
right_eye_point = set(sum(mp_face_mesh.FACEMESH_RIGHT_EYE, ()))
left_iris_point = set(sum(mp_face_mesh.FACEMESH_LEFT_IRIS, ()))
right_iris_point = set(sum(mp_face_mesh.FACEMESH_RIGHT_IRIS, ()))

keypoints = left_eye_point.union(right_eye_point).union(left_iris_point).union(right_iris_point)

keypoints = sorted(list(keypoints))

# Set training data, training labels, validation data, and validation labels

# In[ ]:


train_x = []
train_y = []

val_x = []
val_y = []

for subject in json_data:
  subject_data = json_data[subject];
  train_vids = random.sample(range(0,len(subject_data)), 100)
  for idx, video in enumerate(subject_data):
    for all_features in video['features']:
    #all_features = video['features'][0] # 0 picks the first frame per video. change this eventually?
      if idx in train_vids:
        train_x.append([all_features[i] for i in keypoints])
        train_y.append([int(video['x'])/100, int(video['y'])/100])
      else:
        val_x.append([all_features[i] for i in keypoints])
        val_y.append([int(video['x'])/100, int(video['y'])/100])

# Manipulate amount of data passed into the model for training

# In[ ]:


print((train_x)[len(train_x) - 1])
print((train_x[len(train_x) - int((len(train_x)) / 2):len(train_x)])[-1])

# In[ ]:


vae_input = train_x
#vae_input = train_x[1:len(train_x)]

print(np.shape(vae_input))
vae_input_length = (len(vae_input))

counter = 0
random_matrix = np.random.rand(40, 3)

while(counter < vae_input_length):
    altered_data = []
    i = 0
    while(i < (len(vae_input[counter]))):
        x = (vae_input[counter][i][0] + random_matrix[i][0]) / 2
        y = (vae_input[counter][i][1] + random_matrix[i][1]) / 2
        z = vae_input[counter][i][2]
        altered_data.append([x, y, z])
        i += 1
    #print(np.shape(altered_data))
    vae_input.append(altered_data)
    counter += 1

print(np.shape(vae_input))



# Function for translating facial landmarks

# In[ ]:


def translate(arr, x_translation, y_translation):
    translated_arr = []
    for element in arr:
        x = element[0] + x_translation
        y = element[1] + y_translation
        translated_arr.append([x, y, element[2]])
    return translated_arr

# In[ ]:


length = len(train_x)
i = 0
while(i < length):
    train_x.append(translate(train_x[i], 0.002, 0.002))
    i += 1
print(np.shape(train_x))

# Function for rotating facial landmarks

# In[ ]:


def rotate(arr, angle):
    i = 0
    rotated_arr = []
    rotation = [[np.cos(angle * (np.pi / 180)), np.sin(angle * (np.pi / 180)) * -1],
     [np.sin(angle * (np.pi / 180)), np.cos(angle * (np.pi / 180))]]
    for element in arr:
        degree_rotation = (np.matmul(rotation, [element[0], element[1]]))
        rotated_arr_with_z = np.append(degree_rotation, (element[2]))
        rotated_arr.append(rotated_arr_with_z)
        print(rotated_arr)
        i += 1
    return rotated_arr

# In[ ]:


def no_matrix_rotate(arr, angle):
    i = 0
    rotated_arr = []
    for element in arr:
        x = element[0] * np.cos(angle * (np.pi / 180)) - element[1] * np.sin(angle * (np.pi / 180))
        y = element[0] * np.sin(angle * (np.pi / 180)) + element[1] * np.cos(angle * (np.pi / 180))
        z = element[2]
        rotated_arr.append([x, y, z])
        #print(rotated_arr)
        i += 1
    return rotated_arr


# In[ ]:


no_matrix_rotate([[3, 5, 6]], 50)

# In[ ]:


length = len(train_x)
i = 0
while(i < length):
    train_x.append(no_matrix_rotate(train_x[i], 45))
    i += 1
print(np.shape(train_x))

# VAE for compressing head/eye position

# In[ ]:


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a mesh."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        kl_loss = 1/300 * -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        self.add_metric(kl_loss, name='kl_loss')

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# In[ ]:


latent_dim = 2

vae_encoder_inputs = tf.keras.Input(shape=(40,3), name="vae_encoder_input")
vae_encoder_flatten = tf.keras.layers.Flatten(name="vae_flatten")(vae_encoder_inputs)
vae_encoder_dense_1 = tf.keras.layers.Dense(units=200, activation="relu", name="vae_dense_1")(vae_encoder_flatten)
vae_encoder_dense_2 = tf.keras.layers.Dense(units=50, activation="relu", name="vae_dense_2")(vae_encoder_dense_1)
z_mean = tf.keras.layers.Dense(units=latent_dim, name="z_mean")(vae_encoder_dense_2)
z_log_var = tf.keras.layers.Dense(units=latent_dim, name="z_log_var")(vae_encoder_dense_2)
vae_encoder = tf.keras.Model(inputs=vae_encoder_inputs, outputs=[z_mean, z_log_var])

vae_encoder.summary()

# In[ ]:


z = Sampling(name="vae_sampling")([z_mean, z_log_var])

vae_decoder_dense_1 = tf.keras.layers.Dense(50, activation="relu", name="vae_decoder_dense_1")(z)
vae_decoder_dense_2 = tf.keras.layers.Dense(200, activation="relu", name="vae_decoder_dense_2")(vae_decoder_dense_1)
vae_decoder_dense_3 = tf.keras.layers.Dense(120, activation="sigmoid", name="vae_decoder_dense_3")(vae_decoder_dense_2)
vae_decoder_outputs = tf.keras.layers.Reshape((40,3), name="vae_decoder_reshape")(vae_decoder_dense_3)

vae_decoder = tf.keras.Model(inputs=z, outputs=vae_decoder_outputs)

vae_decoder.summary()

# In[ ]:


vae = tf.keras.Model(inputs=vae_encoder_inputs, outputs=vae_decoder_outputs, name="VAE")

vae.summary()

# In[ ]:


vae.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])

# In[ ]:


vae.fit(x=train_x, y=train_x, epochs=20)

# In[ ]:


validation_metrics = vae.evaluate(val_x, val_x)
a = validation_metrics[0]
b = validation_metrics[1]
c = validation_metrics[2]
print(a)
print(b)
print(c)

# In[ ]:


test_arr = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
for i in test_arr:
    print(i[0])

vae_input2 = vae_input[(len(vae_input) - 1):len(vae_input)]
print(len(vae_input))
print(np.shape(vae_input[30829:len(vae_input)]))
print((vae_input)[1:2])

# In[ ]:


x = 16
units = []
validation_metrics = []
loss = []
mse = []
kl_loss = []

while (x >= 1):
    vae_input2 = vae_input[len(vae_input) - int((len(vae_input)) / x):len(vae_input)]
    vae.fit(x=vae_input2, y=vae_input2, epochs=20)

    units.append(np.shape(vae_input2)[0])
    validation_metrics.append(vae.evaluate(val_x, val_x))
    x = x / 2

for i in validation_metrics:
    loss.append(i[0])
    mse.append(i[1])
    kl_loss.append(i[2])

plt.plot(units, loss, label='loss')
plt.plot(units, mse, label='mse')
plt.plot(units, kl_loss, label='kl_loss')
plt.legend()



# In[ ]:


x = 2
vae_input = train_x[len(train_x) - int((len(train_x)) / x):len(train_x)]
vae.fit(x=vae_input, y=vae_input, epochs=20)

# In[ ]:


x = 16
units = []
validation_metrics = []
loss = []
mse = []
kl_loss = []

while (x >= 1):
    vae_input = train_x[len(train_x) - int((len(train_x)) / x):len(train_x)]
    vae.fit(x=vae_input, y=vae_input, epochs=20)

    units.append(np.shape(vae_input)[0])
    validation_metrics.append(vae.evaluate(val_x, val_x))
    x = x / 2

for i in validation_metrics:
    loss.append(i[0])
    mse.append(i[1])
    kl_loss.append(i[2])

plt.plot(units, loss, label='loss')
plt.plot(units, mse, label='mse')
plt.plot(units, kl_loss, label='kl_loss')
plt.legend()

# In[ ]:


a = vae_decoder.predict([[1,-1]])
b = a[0]
plt.plot(b[0:32,0],b[0:32,1], 'bo')

# In[ ]:


x = np.linspace(-2,2,10)
y = np.linspace(-2,2,10)
fig = plt.figure()

for x_val in x:
  for y_val in y:
    points = vae_decoder.predict([[x_val,y_val]])[0]
    x_coords = points[:,0]
    y_coords = points[:,1]
    
    plt.plot(x_coords, y_coords, 'bo')
