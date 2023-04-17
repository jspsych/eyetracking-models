#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
 import cv2
import os
import fnmatch
import json
import tensorflow as tf
import random
from google.colab.patches import cv2_imshow
!pip install mediapipe
import mediapipe as mp
from google.colab import drive
drive.mount('/content/drive')

# # Configure MediaPipe FaceMesh
# 

# In[ ]:


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

# # Process Video Pipeline

# In[ ]:


def extract_mesh_from_video(path):
  # open a video file for video capturing
  cap = cv2.VideoCapture(path)
  out = []
  
  # to see if video capturing has been initialized
  while(cap.isOpened()):
    # return (1) if any frames grabbed (2) grabbed image (empty if ret is false)
    ret, frame = cap.read()
    # Q: why frame could be none?
    if frame is not None:
      # process an RGB image and returns the face landmarks on each detected face
      results = face_mesh.process(frame)
      # check if any faces detected
      if not results.multi_face_landmarks:
        continue
      landmarks = results.multi_face_landmarks[0].landmark
      # store landmarks as an array of arrays
      lm = [[a.x, a.y, a.z] for a in landmarks]
      # 3D tensor that stores landmarks frame by frame
      out.append(lm)
    else:
      break

  if len(out) > 0:
    out = np.reshape(np.array(out), (len(out), -1, 3)).tolist()
  return out

# In[ ]:


example_landmarks_data = extract_mesh_from_video("/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/webm/v2sfzuft_2_test_95_8.webm")
tf.shape(example_landmarks_data)

# each video has a shape of 32x478x3
# 32 frames, 478 landmarks, coordinates (x, y, z)

# In[ ]:


unique_subjects

# In[ ]:


# get the path of the webm file
path = '/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/webm/'

# store all the file directories
all_files = os.listdir(path)
print(len(all_files))

# get unique subjects
unique_subjects = set([filepath.split('_')[0] for filepath in os.listdir(path)])
print(unique_subjects)

for subject in unique_subjects:
  all_data = {}

  print(subject)

  # check if the subject json file already exists. if so, skip the remainning body
  if os.path.isfile('/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/json/'+subject+'.json'):
    continue
   
  subject_data = []
  # go through all file directories in the webm file, find those that start with the subject name
  subject_files = fnmatch.filter(all_files, subject+'*')
  # manage every single file directory that starts with the subject name
  for filename in subject_files:
    # transform file name into an array
    fileinfo = filename.replace('.','_').split('_')
    # store relevant values
    subject = fileinfo[0]
    block = fileinfo[1]
    phase = fileinfo[2]
    x = fileinfo[3]
    y = fileinfo[4]
    meshfeatures = extract_mesh_from_video(path + filename)
    # create and append a dictionary to the exisiting array
    subject_data.append({
        'block': block,
        'phase': phase,
        'x': x,
        'y': y,
        'features': meshfeatures 
    })
  # once the last for loop is over, assign the subject_data array as the value to the subject key
  all_data[subject] = subject_data

  # export the json file for the subject to the drive
  with open('/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/json/'+subject+'.json', 'w') as file:
    json.dump(all_data, file)

# # Load JSON

# In[ ]:


json_path = '/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/json/'
all_json_files = os.listdir(json_path)

json_data = {}
for filename in all_json_files:
  with open('/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/json/'+filename, 'r') as file:
    s_data = json.load(file)
    json_data = {**json_data, **s_data}

# every element in json_data is the info of a video
# {
#     "lyln56b2": [
#                  {
#                     "block": ...,
#                     "phase": ...,
#                     "x": ...,
#                     "y": ...,
#                     "features": ...,
#                  },
#                  {...},
#                  {...},
#                  ...
#     ]
# }

# # Ridge Regression Model

# In[ ]:


from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import statistics
import math

def is_calibration_data(dict):
  if dict["phase"] == "calibration":
    return True
  else: return False

def is_test_data(dict):
  if dict["phase"] == "test":
    return True
  else: return False

# def get_tc_data(dict):
#   all_videos_data = []
#   for key, value in dict.items():
#     for per_video_data in value:
#       all_videos_data.append(per_video_data)
#   calibration_data = list(filter(is_calibration_data, all_videos_data))
#   test_data = list(filter(is_test_data, all_videos_data))
#   return calibration_data, test_data

def split_and_flatten_data(v_list):
  predictor_data = []
  output_data = []
  for video in v_list:
    coor = [int(video["x"]), int(video["y"])]
    features = video["features"]
    for feature in features:
      predictor_data.append(feature)
      output_data.append(coor)
  predictor_data = np.reshape(np.array(predictor_data), (-1, 478*3)).tolist()
  return predictor_data, output_data

def split_xy(lst_of_coor):
  x = []
  y = []
  for coor in lst_of_coor:
    x.append(coor[0])
    y.append(coor[1])
  x = np.array(x)
  y = np.array(y)
  return x, y

def draw_one_subject(lst_of_coor):
  x, y = split_xy(lst_of_coor)
  plt.scatter(x, y)

def is_block_n_data(dict, block_name):
  if dict["block"] == block_name:
    return True
  else: return False


# In[ ]:


from sympy import Point3D, Line3D

def get_centroid(lst):
  pt1 = lst[0]
  pt2 = lst[1]
  pt3 = lst[2]
  pt4 = lst[3]
  x1, y1, z1 = pt1
  x2, y2, z2 = pt2
  x3, y3, z3 = pt3
  x4, y4, z4 = pt4
  p1, p2 = Point3D(x1, y1, z1), Point3D(x2, y2, z2)
  p3, p4 = Point3D(x3, y3, z3), Point3D(x4, y4, z4)
  l1 = Line3D(p1, p2)
  l2 = Line3D(p3, p4)
  c = intersection(l1, l2)[0]
  x, y, z = c.x, c.y, c.z
  return [x, y, z]


# get_centroid([lst[i] for i in [226, 244, 223, 230]]),
#              get_centroid([lst[i] for i in [359, 463, 257, 253]]),

def landmark_filter(lst):
  lm_lst =  [lst[i] for i in [13, 19, 234, 454, 10, 152]]
  centroid_lst = [
             get_centroid([lst[i] for i in [474, 475, 476, 477]]),
             get_centroid([lst[i] for i in [469, 470, 471, 472]])]
  return lm_lst.extend(centroid_lst)

def split_and_flatten_data(v_list):
  predictor_data = []
  output_data = []
  for video in v_list:
    coor = [int(video["x"]), int(video["y"])]
    features = video["features"]
    for feature in features:
      feature = landmark_filter(feature)
      predictor_data.append(feature)
      output_data.append(coor)
  predictor_data = np.reshape(np.array(predictor_data), (-1, len(feature)*3)).tolist()
  return predictor_data, output_data

model_data = {}

for key, value in json_data.items():
  calibration_data = list(filter(is_calibration_data, value))
  test_data = list(filter(is_test_data, value))
  c_training_data, c_target_data = split_and_flatten_data(calibration_data)
  t_predictor_data, t_target_data = split_and_flatten_data(test_data)

  ridge_model = Ridge()
  ridge_model.fit(c_training_data, c_target_data)
  predictions = ridge_model.predict(t_predictor_data)

  residual_distance = []
  tx, ty = split_xy(t_target_data)
  px, py = split_xy(predictions)
  residual_x = px - tx
  residual_y = py - ty
  for i in range(len(predictions)):
    d = math.sqrt(residual_x[i]**2 + residual_y[i]**2)
    residual_distance.append(d)
  mean_d = statistics.mean(residual_distance)

  model_data[key] = {
      "calibration data": calibration_data,
      "test data": t_target_data,
      "predicted data": predictions,
      "residual d": residual_distance,
      "mean d": mean_d,
      "residual x": residual_x,
      "residual y": residual_y 
  }


# In[ ]:


# for key, value in model_data.items():
#     print(value["mean d"])

# test_data = []
# predictions = []
# residual_x = []
# residual_y = []
# residual_d = []

# for key, value in model_data.items():
#   test_data.append(value["test data"])
#   predictions.append(value["predicted data"])
#   residual_x.append(value["residual x"])
#   residual_y.append(value["residual y"])
#   residual_d.append(value["residual d"])

# test_data = np.concatenate(test_data).tolist()
# predictions = np.concatenate(predictions).tolist()

# draw_one_subject(predictions)
# draw_one_subject(test_data)
# plt.show()

# residual_x = np.concatenate(residual_x).tolist()
# plt.hist(residual_x)
# plt.show()

# residual_y = np.concatenate(residual_y).tolist()
# plt.hist(residual_y)
# plt.show()

# residual_d = np.concatenate(residual_d).tolist()
# plt.hist(residual_d)
# plt.show()

# # Simple Keras Model on One Subject

# In[ ]:


left_eye_point = set(sum(mp_face_mesh.FACEMESH_LEFT_EYE, ()))
right_eye_point = set(sum(mp_face_mesh.FACEMESH_RIGHT_EYE, ()))
left_iris_point = set(sum(mp_face_mesh.FACEMESH_LEFT_IRIS, ()))
right_iris_point = set(sum(mp_face_mesh.FACEMESH_RIGHT_IRIS, ()))

keypoints = left_eye_point.union(right_eye_point).union(left_iris_point).union(right_iris_point)

keypoints = sorted(list(keypoints))

left_iris_point
right_iris_point 

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

# In[ ]:


train_y[0]

# In[ ]:


# ideas
# - try sharing the weights between the eyes to improve training?
# - error in y seems to be larger than error in x; is there a way to scale y values? normalization?
# - add calibration data as model input
# - try learning a generative model of face location by compressing eye (but NOT iris) data through something like a VAE.
#   then learn generative model of iris within? or maybe just learn together and hope VAE can separate dimensions out?
#   could even train the generative model on webcam face datasets. don't need this specific data.
# - check literature for solutions

model_inputs = tf.keras.Input(shape=(40,3))
model_flatten = tf.keras.layers.Flatten()(model_inputs)
model_dense_1 = tf.keras.layers.Dense(units=400, activation="relu")(model_flatten)
model_res_1 = tf.keras.layers.Concatenate()([model_flatten, model_dense_1])
model_dropout_1 = tf.keras.layers.Dropout(0.25)(model_res_1)
model_dense_2 = tf.keras.layers.Dense(units=250, activation="relu")(model_dropout_1)
model_res_2 = tf.keras.layers.Concatenate()([model_flatten, model_dense_1, model_dense_2])
model_dropout_2 = tf.keras.layers.Dropout(0.25)(model_res_2)
model_outputs = tf.keras.layers.Dense(units=2, activation=None)(model_dropout_2)

model = tf.keras.Model(inputs=model_inputs, outputs=model_outputs)

model.summary()

# In[ ]:


model.compile(optimizer='nadam', loss='mean_absolute_error', metrics=[tf.keras.metrics.mean_absolute_error])

# In[ ]:


model.fit(x=train_x, y=train_y, epochs=100, validation_data = (val_x, val_y))

# # a VAE for compressing the head/eye position

# In[ ]:


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a mesh."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# In[ ]:


latent_dim = 2

vae_encoder_inputs = tf.keras.Input(shape=(40,3), name="vae_encoder_input")
vae_encoder_flatten = tf.keras.layers.Flatten(name="vae_flatten")(vae_encoder_inputs)
vae_encoder_dense_1 = tf.keras.layers.Dense(units=200, activation="relu", name="vae_dense_1")(vae_encoder_flatten)
vae_encoder_dense_2 = tf.keras.layers.Dense(units=50, activation="relu", name="vae_dense_2")(vae_encoder_dense_1)
z_mean = tf.keras.layers.Dense(units=latent_dim, name="z_mean")(vae_encoder_dense_2)
z_log_var = tf.keras.layers.Dense(units=latent_dim, name="z_log_var")(vae_encoder_dense_2)
z = Sampling(name="vae_sampling")([z_mean, z_log_var])

# In[ ]:


vae_decoder_dense_1 = tf.keras.layers.Dense(50, activation="relu", name="vae_decoder_dense_1")(z)
vae_decoder_dense_2 = tf.keras.layers.Dense(200, activation="relu", name="vae_decoder_dense_2")(vae_decoder_dense_1)
vae_decoder_dense_3 = tf.keras.layers.Dense(120, activation="sigmoid", name="vae_decoder_dense_3")(vae_decoder_dense_2)
vae_decoder_outputs = tf.keras.layers.Reshape((40,3), name="vae_decoder_reshape")(vae_decoder_dense_3)

# In[ ]:


vae = tf.keras.Model(inputs=vae_encoder_inputs, outputs=vae_decoder_outputs, name="VAE")

#kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
#vae.add_loss(kl_loss)

# In[ ]:


vae.compile(optimizer="adam", loss="mean_squared_error")

# In[ ]:


vae.losses

# In[ ]:


vae.fit(x=train_x, y=train_x, epochs=25)
