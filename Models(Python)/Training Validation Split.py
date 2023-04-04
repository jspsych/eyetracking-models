#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')
!pip install mediapipe
import mediapipe as mp
import os
import json
import random
import math

# # Get JSON input files in Randomized Order

# In[2]:


input_json_path = '/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/json_inputs/'

all_input_json_files = os.listdir(input_json_path)
random.shuffle(all_input_json_files)

last_training_index = len(all_input_json_files) - (math.floor(len(all_input_json_files) / 10))
training_subjects = all_input_json_files[0:last_training_index]
validation_subjects = all_input_json_files[last_training_index:len(all_input_json_files)]

input_json_data = {}
for filename in training_subjects:
  with open(input_json_path + filename, 'r') as file:
    s_data = json.load(file)
    input_json_data = {**input_json_data, **s_data}

# # Set Training Labels and Training Data

# In[3]:


calibration_points = [[10, 50], [10, 10], [90, 10], [50, 90],
                   [30, 70], [50, 50], [50, 10], [90, 90],
                   [70, 70], [70, 30], [10, 90], [90, 50],
                   [30, 30]]

# In[4]:


train_y = []

for subject in input_json_data:
    for y in input_json_data[subject]['y']:
        train_y.append(y)
        for element in calibration_points:
            train_y.append(element)

print(len(train_y))
#print(train_y)

# In[5]:


train_x = []

for subject in input_json_data:
    for sample in input_json_data[subject]['x']:
        calibration_arr = []
        for point in sample:
            total_arr = []
            head_pose = np.array(point[0])
            iris_points = np.array(point[1])
            head_pose = head_pose.flatten()
            iris_points = iris_points.flatten()
            for element in head_pose:
                total_arr.append(element)
            for element in iris_points:
                total_arr.append(element)
            calibration_arr.append(total_arr)
        #train_x.append(calibration_arr[0])
        for element in calibration_arr:
            train_x.append(element)

# In[6]:


print(np.shape(train_x))
print(np.shape(train_y))

# ## Normalize and Shuffle Training Data

# In[7]:


train_x = np.array(train_x)
mean = train_x.mean(axis=0)
train_x -= mean
std = train_x.std(axis=0)
train_x /= std
train_y = np.array(train_y)

# In[8]:


from sklearn.utils import shuffle

train_x, train_y = shuffle(train_x, train_y)

# In[9]:


print(np.shape(train_x))
print(np.shape(train_y))

# # Set Validation Labels and Validation Data

# In[10]:


validation_input_json_data = {}
for filename in validation_subjects:
  with open(input_json_path + filename, 'r') as file:
    s_data = json.load(file)
    validation_input_json_data = {**input_json_data, **s_data}

# In[11]:


val_x = []
val_y = []

for subject in validation_input_json_data:
    for y in validation_input_json_data[subject]['y']:
        val_y.append(y)

for subject in validation_input_json_data:
    for sample in validation_input_json_data[subject]['x']:
        calibration_arr = []
        for point in sample:
            total_arr = []
            head_pose = np.array(point[0])
            iris_points = np.array(point[1])
            head_pose = head_pose.flatten()
            iris_points = iris_points.flatten()
            for element in head_pose:
                total_arr.append(element)
            for element in iris_points:
                total_arr.append(element)
            calibration_arr.append(total_arr)
        val_x.append(calibration_arr[0])

# In[12]:


val_x = np.array(val_x)
mean = val_x.mean(axis=0)
val_x -= mean
std = val_x.std(axis=0)
val_x /= std
val_y = np.array(val_y)

# In[13]:


print(np.shape(val_x))
print(np.shape(val_y))

# In[14]:


train_x_list = train_x.tolist()
train_y_list = train_y.tolist()
val_x_list = val_x.tolist()
val_y_list = val_y.tolist()

training_data = {
    'train_x': train_x_list,
    'train_y': train_y_list
}

validation_data = {
    'val_x': val_x_list,
    'val_y': val_y_list
}

with open('/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/model_data/training_data.json', 'w') as file:
    json.dump(training_data, file)

with open('/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/model_data/validation_data.json', 'w') as file:
    json.dump(validation_data, file)

