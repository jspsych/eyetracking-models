#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')
!pip install mediapipe
import mediapipe as mp
import os
import json

# # Load Trained VAE
# 
# 
# 
# 

# In[ ]:


vae_path = '/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/vae_encoder/vae_2022-07-11_16:53:53'
vae_encoder = tf.keras.models.load_model(vae_path)

vae_encoder.summary()

# # Load MediaPipe model to get the set of mesh points

# In[ ]:


mp_face_mesh = mp.solutions.face_mesh

left_eye_point = set(sum(mp_face_mesh.FACEMESH_LEFT_EYE, ()))
right_eye_point = set(sum(mp_face_mesh.FACEMESH_RIGHT_EYE, ()))
left_iris_point = set(sum(mp_face_mesh.FACEMESH_LEFT_IRIS, ()))
right_iris_point = set(sum(mp_face_mesh.FACEMESH_RIGHT_IRIS, ()))

face_oval_point = set(sum(mp_face_mesh.FACEMESH_FACE_OVAL, ()))

#keypoints = left_eye_point.union(right_eye_point).union(left_iris_point).union(right_iris_point)

keypoints = left_eye_point.union(right_eye_point).union(face_oval_point)

keypoints = sorted(list(keypoints))

# # Load in JSON from gaze tracking experiment

# In[ ]:


json_path = '/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/json/'
all_json_files = os.listdir(json_path)

json_data = {}
for filename in all_json_files:
  with open('/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/json/'+filename, 'r') as file:
    s_data = json.load(file)
    json_data = {**json_data, **s_data}

# # Create a function to structure input
# 
# Input to the model is `[(14, 4), (14, 8, 3)]` where first chunk is latent VAE features and second chunk is iris locations. Indexes 0-12 are calibration points, and index 13 is the unknown point. 0-12 have to have the following structure:
# 
# 0. `x=10, y=10`
# 1. `x=10, y=50`
# 2.
# 12.
# 
# 
# 

# In[ ]:


frames = []

for subject in json_data:
  subject_data = json_data[subject];
  
  for video in subject_data:
    for all_features in video['features']:
      frames.append([all_features[i] for i in keypoints])

# # Use trained model to get latent features

# In[ ]:


print(tf.shape(frames))

latent_features = vae_encoder.predict(frames)

# In[ ]:


len(latent_features)
