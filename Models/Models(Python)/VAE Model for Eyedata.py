#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
import json
import tensorflow as tf
import random
import itertools
!pip install mediapipe
import mediapipe as mp
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')

# # Configure MediaPipe FaceMesh
# 

# In[ ]:


mp_face_mesh = mp.solutions.face_mesh

# # Grab data for training model
# 
# 
# 
# 

# This is grabbing the head pose video JSON instead of the eye data video.

# In[ ]:


json_path = '/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/json_pose/'
all_json_files = os.listdir(json_path)

json_data = {}
for filename in all_json_files:
  with open(json_path + filename, 'r') as file:
    s_data = json.load(file)
    json_data = {**json_data, **s_data}

# In[ ]:


left_eye_point = set(sum(mp_face_mesh.FACEMESH_LEFT_EYE, ()))
right_eye_point = set(sum(mp_face_mesh.FACEMESH_RIGHT_EYE, ()))
left_iris_point = set(sum(mp_face_mesh.FACEMESH_LEFT_IRIS, ()))
right_iris_point = set(sum(mp_face_mesh.FACEMESH_RIGHT_IRIS, ()))

face_oval_point = set(sum(mp_face_mesh.FACEMESH_FACE_OVAL, ()))

#keypoints = left_eye_point.union(right_eye_point).union(left_iris_point).union(right_iris_point)

keypoints = left_eye_point.union(right_eye_point).union(face_oval_point)

keypoints = sorted(list(keypoints))

# In[ ]:


face_oval_point

# In[ ]:


keypoints

# In[ ]:


train_x = []

for subject in json_data:
  subject_data = json_data[subject];
  
  for video in subject_data:
    for all_features in video['features']:
      train_x.append([all_features[i] for i in keypoints])
        

# In[ ]:


len(train_x)

# # a VAE for compressing the head/eye position

# In[ ]:


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a mesh."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        kl_loss = 1/600 * -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        self.add_metric(kl_loss, name='kl_loss')

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# In[ ]:


len(train_x[0])

# In[ ]:


latent_dim = 4
features = len(train_x[0])

vae_encoder_inputs = tf.keras.Input(shape=(features,3), name="vae_encoder_input")
vae_encoder_flatten = tf.keras.layers.Flatten(name="vae_flatten")(vae_encoder_inputs)
vae_encoder_dense_1 = tf.keras.layers.Dense(units=200, activation="relu", name="vae_dense_1")(vae_encoder_flatten)
vae_encoder_dense_2 = tf.keras.layers.Dense(units=100, activation="relu", name="vae_dense_2")(vae_encoder_dense_1)
vae_encoder_dense_3 = tf.keras.layers.Dense(units=50, activation="relu", name="vae_dense_3")(vae_encoder_dense_2)
z_mean = tf.keras.layers.Dense(units=latent_dim, name="z_mean")(vae_encoder_dense_3)
z_log_var = tf.keras.layers.Dense(units=latent_dim, name="z_log_var")(vae_encoder_dense_3)
vae_encoder = tf.keras.Model(inputs=vae_encoder_inputs, outputs=[z_mean, z_log_var])

vae_encoder.summary()

# In[ ]:


z = Sampling(name="vae_sampling")([z_mean, z_log_var])

vae_decoder_dense_1 = tf.keras.layers.Dense(50, activation="relu", name="vae_decoder_dense_1")(z)
vae_decoder_dense_2 = tf.keras.layers.Dense(100, activation="relu", name="vae_decoder_dense_2")(vae_decoder_dense_1)
vae_decoder_dense_3 = tf.keras.layers.Dense(200, activation="relu", name="vae_decoder_dense_3")(vae_decoder_dense_2)
vae_decoder_dense_4 = tf.keras.layers.Dense(features*3, activation=None, name="vae_decoder_dense_4")(vae_decoder_dense_3)
vae_decoder_outputs = tf.keras.layers.Reshape((features,3), name="vae_decoder_reshape")(vae_decoder_dense_4)

vae_decoder = tf.keras.Model(inputs=z, outputs=vae_decoder_outputs)

vae_decoder.summary()

# In[ ]:


vae = tf.keras.Model(inputs=vae_encoder_inputs, outputs=vae_decoder_outputs, name="VAE")

vae.summary()

# In[ ]:


vae.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])

# In[ ]:


vae.fit(x=train_x, y=train_x, epochs=50)

# In[ ]:


vae_encoder_fixed = tf.keras.Model(inputs=vae_encoder_inputs, outputs=z_mean)

vae_fixed = tf.keras.Sequential([vae_encoder_fixed, vae_decoder])

vae_fixed.summary()

# This plots the latent space all on one graph

# In[ ]:


min_val = -2
max_val = 2
breaks = 5

fig = plt.figure()

prediction_vals = list(itertools.product(np.linspace(min_val, max_val, breaks), repeat = latent_dim))

len(prediction_vals)

len(prediction_vals[0])

points = vae_decoder.predict(prediction_vals)

x_coords = points[:, :, 0]
y_coords = points[:, :, 1]

plt.ylim([1,0])

for i in range(len(x_coords)):
  plt.plot(x_coords[i,:], y_coords[i,:], 'bo')

# In[ ]:


points = vae_decoder.predict([[2,2,0,0]])

x_coords = points[:, :, 0]
y_coords = points[:, :, 1]

plt.ylim([1,0])
plt.xlim([0,1])
plt.plot(x_coords[0,:], y_coords[0,:], 'bo')

# This plot shows that the reconstructions aren't quite good enough yet. Gets eyes in approximately the right location but orientation not right and outline not centered over original location.
# 
# But... this still has the random sampling component! **Need to create a network that doesn't randomly sample for Z but just uses the mean z value.**

# In[ ]:


example = np.array(train_x[100])
reconstructions = vae.predict(np.reshape(example, (1, -1, 3)))

plt.plot(example[:,0], example[:,1], 'bo')
plt.plot(reconstructions[0,:,0], reconstructions[0,:,1], 'ro')
plt.xlim([0,1])
plt.ylim([1,0])

# This next chunk creates a model that bypasses the random sampling layer and uses only the `z_mean` values as the latent representation. 

# Now we can rerun the prediction step and it will generate the same output each time. The predictions are much closer, but still not quite right.

# In[ ]:


example = np.array(train_x[500])
reconstructions = vae_fixed.predict(np.reshape(example, (1, -1, 3)))

plt.plot(example[:,0], example[:,1], 'bo')
plt.plot(reconstructions[0,:,0], reconstructions[0,:,1], 'ro')
plt.xlim([0,1])
plt.ylim([1,0])

# In[ ]:


from datetime import datetime
timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

vae_encoder_fixed.save('/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/vae_encoder/vae_'+timestamp)
