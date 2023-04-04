#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# ### Configure MediaPipe FaceMesh

# In[2]:


mp_face_mesh = mp.solutions.face_mesh

# ### Grab data from training model

# In[3]:


json_path = '/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/json_pose/'
all_json_files = os.listdir(json_path)

json_data = {}
for filename in all_json_files:
  with open(json_path + filename, 'r') as file:
    s_data = json.load(file)
    json_data = {**json_data, **s_data}

# ### Set training data/training labels

# In[4]:


left_eye_point = set(sum(mp_face_mesh.FACEMESH_LEFT_EYE, ()))
right_eye_point = set(sum(mp_face_mesh.FACEMESH_RIGHT_EYE, ()))
left_iris_point = set(sum(mp_face_mesh.FACEMESH_LEFT_IRIS, ()))
right_iris_point = set(sum(mp_face_mesh.FACEMESH_RIGHT_IRIS, ()))

face_oval_point = set(sum(mp_face_mesh.FACEMESH_FACE_OVAL, ()))

#keypoints = left_eye_point.union(right_eye_point).union(left_iris_point).union(right_iris_point)

keypoints = left_eye_point.union(right_eye_point).union(face_oval_point)

keypoints = sorted(list(keypoints))

# In[5]:


added_points = [151, 9, 8, 168, 6, 197, 195, 5, 4]
for point in added_points:
    keypoints.append(point)

# In[6]:


train_x = []

for subject in json_data:
  subject_data = json_data[subject];
  
  for video in subject_data:
    for all_features in video['features']:
      train_x.append([all_features[i] for i in keypoints])
        

# ### Set validation data/validation labels

# In[7]:


#random.shuffle(train_x)

val_x = train_x[-6623:len(train_x)]
train_x = train_x[0:len(train_x) - 6623]

# In[8]:


len(train_x)

# ### No Data Augmentation VAE Setup/Performance

# In[9]:


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

# In[10]:


latent_dim = 8
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

# In[11]:


z = Sampling(name="vae_sampling")([z_mean, z_log_var])

vae_decoder_dense_1 = tf.keras.layers.Dense(50, activation="relu", name="vae_decoder_dense_1")(z)
vae_decoder_dense_2 = tf.keras.layers.Dense(100, activation="relu", name="vae_decoder_dense_2")(vae_decoder_dense_1)
vae_decoder_dense_3 = tf.keras.layers.Dense(200, activation="relu", name="vae_decoder_dense_3")(vae_decoder_dense_2)
vae_decoder_dense_4 = tf.keras.layers.Dense(features*3, activation=None, name="vae_decoder_dense_4")(vae_decoder_dense_3)
vae_decoder_outputs = tf.keras.layers.Reshape((features,3), name="vae_decoder_reshape")(vae_decoder_dense_4)

vae_decoder = tf.keras.Model(inputs=z, outputs=vae_decoder_outputs)

vae_decoder.summary()

# In[12]:


vae = tf.keras.Model(inputs=vae_encoder_inputs, outputs=vae_decoder_outputs, name="VAE")

vae.summary()

# In[13]:


vae.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])

# In[14]:


vae.fit(x=train_x, y=train_x, epochs=50)

# In[15]:


vae_encoder_fixed = tf.keras.Model(inputs=vae_encoder_inputs, outputs=z_mean)

vae_fixed = tf.keras.Sequential([vae_encoder_fixed, vae_decoder])

vae_fixed.summary()

# In[16]:


validation_metrics = []
validation_metrics.append(vae.evaluate(val_x, val_x))

# In[17]:


j = 0
reconstructions_arr = []
while j < (len(val_x)):
    example = np.array(val_x[j])
    reconstructions = vae_fixed.predict(np.reshape(example, (1, -1, 3)))
    reconstructions_arr.append(reconstructions[0])
    j += 1

i = 0
k = 0
x = []
y = []

while k < len(val_x[i]):
    x.append(val_x[i][k][0])
    y.append(val_x[i][k][1])
    k +=1

fig = plt.figure(figsize=(8,8))
ax = plt.axes(xlim=(0,1),ylim=(1,0))
scatter = ax.scatter(x, y, c='red')
scatter2 = ax.scatter(x, y, c='blue')

def update(n):
    x = []
    y = []
    k = 0
    global i
    coords = []
    coords2 = []

    while k < len(val_x[i]):
        x.append(val_x[i][k][0])
        y.append(val_x[i][k][1])
        coords.append([reconstructions_arr[i][k][0], reconstructions_arr[i][k][1]])
        coords2.append([val_x[i][k][0], val_x[i][k][1]])

        k +=1

    
    scatter.set_offsets(coords)
    scatter2.set_offsets(coords2)
    i += 1
    return scatter, scatter2

anim = FuncAnimation(fig, update, frames=(int((len(val_x)) / 4)), interval=60)
HTML(anim.to_html5_video())

# ### Explore latent representation of data

# In[79]:


outputs = vae_decoder.predict([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
predictions = outputs[0]

x = []
y = []

for prediction in predictions:
    x.append(prediction[0])
    y.append(prediction[1])

fig = plt.figure(figsize=(8,8))
ax = plt.axes(xlim=(0,1),ylim=(1,0))
scatter = ax.scatter(x, y, c='red')

# ### Test against prolific dataset samples

# In[26]:


json_path = '/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/prolific_eye_data_experiment/vae_test_json/'
all_json_files = os.listdir(json_path)

json_data = {}
for filename in all_json_files:
  with open(json_path + filename, 'r') as file:
    s_data = json.load(file)
    json_data = {**json_data, **s_data}

# In[27]:


left_eye_point = set(sum(mp_face_mesh.FACEMESH_LEFT_EYE, ()))
right_eye_point = set(sum(mp_face_mesh.FACEMESH_RIGHT_EYE, ()))
left_iris_point = set(sum(mp_face_mesh.FACEMESH_LEFT_IRIS, ()))
right_iris_point = set(sum(mp_face_mesh.FACEMESH_RIGHT_IRIS, ()))

face_oval_point = set(sum(mp_face_mesh.FACEMESH_FACE_OVAL, ()))

#keypoints = left_eye_point.union(right_eye_point).union(left_iris_point).union(right_iris_point)

keypoints = left_eye_point.union(right_eye_point).union(face_oval_point)

keypoints = sorted(list(keypoints))

# In[28]:


added_points = [151, 9, 8, 168, 6, 197, 195, 5, 4]
for point in added_points:
    keypoints.append(point)

# In[29]:


prolific_val_x = []

for subject in json_data:
  subject_data = json_data[subject];
  
  for video in subject_data:
    for all_features in video['features']:
      prolific_val_x.append([all_features[i] for i in keypoints])

# In[25]:




# In[30]:


j = 0
reconstructions_arr = []
while j < (len(prolific_val_x)):
    example = np.array(prolific_val_x[j])
    reconstructions = vae_fixed.predict(np.reshape(example, (1, -1, 3)))
    reconstructions_arr.append(reconstructions[0])
    j += 1

i = 0
k = 0
x = []
y = []

while k < len(prolific_val_x[i]):
    x.append(prolific_val_x[i][k][0])
    y.append(prolific_val_x[i][k][1])
    k +=1

fig = plt.figure(figsize=(8,8))
ax = plt.axes(xlim=(0,1),ylim=(1,0))
scatter = ax.scatter(x, y, c='red')
scatter2 = ax.scatter(x, y, c='blue')

def update(n):
    x = []
    y = []
    k = 0
    global i
    coords = []
    coords2 = []

    while k < len(prolific_val_x[i]):
        x.append(prolific_val_x[i][k][0])
        y.append(prolific_val_x[i][k][1])
        coords.append([reconstructions_arr[i][k][0], reconstructions_arr[i][k][1]])
        coords2.append([prolific_val_x[i][k][0], prolific_val_x[i][k][1]])

        k +=1

    
    scatter.set_offsets(coords)
    scatter2.set_offsets(coords2)
    i += 1
    return scatter, scatter2

anim = FuncAnimation(fig, update, frames=(int((len(prolific_val_x)) - 1)), interval=60)
HTML(anim.to_html5_video())

# ### Transformation Functions for Data Augmentation

# In[ ]:


def translate(arr, x_translation, y_translation):
    translated_arr = []
    for element in arr:
        x = element[0] + x_translation
        y = element[1] + y_translation
        z = element[2]
        translated_arr.append([x, y, z])
    return translated_arr

# In[ ]:


def rotate(arr, angle):
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


def rotate_around_point(arr, angle, center_point):
    i = 0
    rotated_arr = []
    for element in arr:
        x = (((element[0] - center_point[0]) * np.cos(angle * (np.pi / 180)) 
        - (element[1] - center_point[1]) * np.sin(angle * (np.pi / 180))) 
        + center_point[0])

        y = (((element[0] - center_point[0]) * np.sin(angle * (np.pi / 180)) 
        + (element[1] - center_point[1]) * np.cos(angle * (np.pi / 180)))
        + center_point[1])

        z = element[2]
        rotated_arr.append([x, y, z])
        #print(rotated_arr)
        i += 1
    return rotated_arr

# In[ ]:


def mirror(arr):
    i = 0
    mirrored_arr = []
    for element in arr:
        x = 0.5 - (element[0] - 0.5)
        y = element[1]
        z = element[2]
        mirrored_arr.append([x, y, z])
        i += 1
    return mirrored_arr

# ### Augment Dataset

# In[ ]:


length = len(train_x)
i = 0
while(i < length):
    train_x.append(rotate_around_point(train_x[i], 25, [0.5, 0.5]))
    #train_x.append(rotate_around_point(train_x[i], 50, [0.5, 0.5]))
    train_x.append(rotate_around_point(train_x[i], 75, [0.5, 0.5]))
    #train_x.append(rotate_around_point(train_x[i], 90, [0.5, 0.5]))
    #train_x.append(rotate_around_point(train_x[i], 120, [0.5, 0.5]))
    #train_x.append(rotate_around_point(train_x[i], 145, [0.5, 0.5]))
    train_x.append(mirror(train_x[i]))
    train_x.append(translate(train_x[i], 0.03, 0.03))
    #train_x.append(rotate_around_point(mirror(train_x[i]), 25, [0.5, 0.5]))
    i += 1
print(np.shape(train_x))

# ### Visualize Augmented Dataset

# In[ ]:


j = 0
visualization_set = train_x[-6623:(len(train_x) - 1)]

i = 0
k = 0
x = []
y = []

while k < len(visualization_set[i]):
    x.append(visualization_set[i][k][0])
    y.append(visualization_set[i][k][1])
    k +=1

fig = plt.figure(figsize=(8,8))
ax = plt.axes(xlim=(0,1),ylim=(1,0))
scatter = ax.scatter(x, y, c='red')

def update(n):
    x = []
    y = []
    k = 0
    global i
    coords = []

    while k < len(visualization_set[i]):
        x.append(visualization_set[i][k][0])
        y.append(visualization_set[i][k][1])
        coords.append([visualization_set[i][k][0], visualization_set[i][k][1]])

        k +=1

    
    scatter.set_offsets(coords)
    i += 1
    return scatter

anim = FuncAnimation(fig, update, frames=(int(len(visualization_set)) - 1), interval=60)
HTML(anim.to_html5_video())

# ### Data Augmentation VAE Setup/Performance

# In[ ]:


vae.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])

# In[ ]:


vae.fit(x=train_x, y=train_x, epochs=50)

# In[ ]:


validation_metrics.append(vae.evaluate(val_x, val_x))

# In[ ]:


j = 0
reconstructions_arr = []
while j < (len(val_x)):
    example = np.array(val_x[j])
    reconstructions = vae_fixed.predict(np.reshape(example, (1, -1, 3)))
    reconstructions_arr.append(reconstructions[0])
    j += 1

i = 0
k = 0
x = []
y = []

while k < len(val_x[i]):
    x.append(val_x[i][k][0])
    y.append(val_x[i][k][1])
    k +=1

fig = plt.figure(figsize=(8,8))
ax = plt.axes(xlim=(0,1),ylim=(1,0))
scatter = ax.scatter(x, y, c='red')
scatter2 = ax.scatter(x, y, c='blue')

def update(n):
    x = []
    y = []
    k = 0
    global i
    coords = []
    coords2 = []

    while k < len(val_x[i]):
        x.append(val_x[i][k][0])
        y.append(val_x[i][k][1])
        coords.append([reconstructions_arr[i][k][0], reconstructions_arr[i][k][1]])
        coords2.append([val_x[i][k][0], val_x[i][k][1]])

        k +=1

    
    scatter.set_offsets(coords)
    scatter2.set_offsets(coords2)
    i += 1
    return scatter, scatter2

anim = FuncAnimation(fig, update, frames=(int(len(val_x)) - 1), interval=60)
HTML(anim.to_html5_video())

# ### Model Performance Comparison
# 

# Latent Dim Changes Performance Comparison 

# In[ ]:


validation_metrics = []
x = 16

while x >= 1:
    latent_dim = x 
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

    z = Sampling(name="vae_sampling")([z_mean, z_log_var])

    vae_decoder_dense_1 = tf.keras.layers.Dense(50, activation="relu", name="vae_decoder_dense_1")(z)
    vae_decoder_dense_2 = tf.keras.layers.Dense(100, activation="relu", name="vae_decoder_dense_2")(vae_decoder_dense_1)
    vae_decoder_dense_3 = tf.keras.layers.Dense(200, activation="relu", name="vae_decoder_dense_3")(vae_decoder_dense_2)
    vae_decoder_dense_4 = tf.keras.layers.Dense(features*3, activation=None, name="vae_decoder_dense_4")(vae_decoder_dense_3)
    vae_decoder_outputs = tf.keras.layers.Reshape((features,3), name="vae_decoder_reshape")(vae_decoder_dense_4)

    vae_decoder = tf.keras.Model(inputs=z, outputs=vae_decoder_outputs)

    vae_decoder.summary()

    vae = tf.keras.Model(inputs=vae_encoder_inputs, outputs=vae_decoder_outputs, name="VAE")

    vae.summary()

    vae.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])

    vae.fit(x=train_x, y=train_x, epochs=50)

    vae_encoder_fixed = tf.keras.Model(inputs=vae_encoder_inputs, outputs=z_mean)

    vae_fixed = tf.keras.Sequential([vae_encoder_fixed, vae_decoder])

    vae_fixed.summary()
    validation_metrics.append(vae.evaluate(val_x, val_x))

    x = x / 2


# In[ ]:


latent_dim = 16
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

z = Sampling(name="vae_sampling")([z_mean, z_log_var])

vae_decoder_dense_1 = tf.keras.layers.Dense(50, activation="relu", name="vae_decoder_dense_1")(z)
vae_decoder_dense_2 = tf.keras.layers.Dense(100, activation="relu", name="vae_decoder_dense_2")(vae_decoder_dense_1)
vae_decoder_dense_3 = tf.keras.layers.Dense(200, activation="relu", name="vae_decoder_dense_3")(vae_decoder_dense_2)
vae_decoder_dense_4 = tf.keras.layers.Dense(features*3, activation=None, name="vae_decoder_dense_4")(vae_decoder_dense_3)
vae_decoder_outputs = tf.keras.layers.Reshape((features,3), name="vae_decoder_reshape")(vae_decoder_dense_4)

vae_decoder = tf.keras.Model(inputs=z, outputs=vae_decoder_outputs)

vae_decoder.summary()

vae = tf.keras.Model(inputs=vae_encoder_inputs, outputs=vae_decoder_outputs, name="VAE")

vae.summary()

vae.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])

vae.fit(x=train_x, y=train_x, epochs=50)

vae_encoder_fixed = tf.keras.Model(inputs=vae_encoder_inputs, outputs=z_mean)

vae_fixed = tf.keras.Sequential([vae_encoder_fixed, vae_decoder])

vae_fixed.summary()
validation_metrics.append(vae.evaluate(val_x, val_x))

# In[ ]:


print(validation_metrics)
loss = []
mse = []
for metric in validation_metrics:
    loss.append(metric[0])
    mse.append(metric[1])
plt.plot(loss)
plt.plot(mse)

# ### Augment Validation data

# In[ ]:


length = len(val_x)
i = 0
while(i < length):
    val_x.append(rotate_around_point(val_x[i], 25, [0.5, 0.5]))
    i += 1
print(np.shape(val_x))

# In[ ]:


len(val_x) / 2
len(val_x)
augmented_val_x = val_x[int(len(val_x) / 2):len(val_x)]
len(augmented_val_x)

