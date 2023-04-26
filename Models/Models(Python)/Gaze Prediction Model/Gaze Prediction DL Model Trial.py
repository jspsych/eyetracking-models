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

# 
# 
# # Load Trained VAE Model

# In[ ]:


vae_path = '/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/vae_encoder/vae_2022-07-21_17:15:34'
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
keypoints = keypoints.union([151, 9, 8, 168, 6, 197, 195, 5, 4])

keypoints = sorted(list(keypoints))


# # Helper functions for the processing function
# 

# In[ ]:


import statistics
import math
import random

# Order in which calibration data is to be sorted
calibration_pts = [['10', '50'], ['10', '10'], ['90', '10'], ['50', '90'],
                   ['30', '70'], ['50', '50'], ['50', '10'], ['90', '90'],
                   ['70', '70'], ['70', '30'], ['10', '90'], ['90', '50'],
                   ['30', '30']]

# Define indices of corresponding features in the 478 features list
irises = [469, 470, 471, 472, 474, 475, 476, 477]
face_cross = [226, 446, 9, 195]

# Return coefficients a, b that represent the straight line 
# constructed by the given points pt1, pt2 (y = ax + b)
def get_line(pt1, pt2):
  x1, y1 = pt1
  x2, y2 = pt2
  a = (y2 - y1) / (x2 - x1)
  b = y1 - (a * x1)
  return [a, b]

# Return the coordinate of intersection of two straight lines
# l1, l2 in terms of [x, y]
def get_intersection(l1, l2):
  a1, b1 = l1
  a2, b2 = l2
  x = (b2 - b1) / (a1 - a2)
  y = a1 * x + b1
  return [x, y]

# Return the angle that a vector needs to rotate counter-clockwisely
# in order to point at the same direction as the x-axis
def get_ccw_angle(vector):
  x, y = vector
  tan = y / x
  r = math.atan(tan)
  if x >= 0 and y > 0:
    pass
  elif x < 0 and y >= 0:
    r = r + math.pi
  elif x <= 0 and y < 0:
    r = r + math.pi
  elif x > 0 and y <= 0:
    r = r + 2 * math.pi
  else:
    r = 0
  return r

# Given a list of 4 landmarks on the face mesh, construct a new coordinate
# system relative to the face, where the first two points from the list
# determine the x-axis and its intersection with the line constructed by
# the last two points is the origin of the new coordinate system. Return
# origin, rad. rad represents the angle the x-axis of the new coordinate
# system needs to rotate counter-clockwisely in order to point at the same
# direction as the x-axis of the coordinate system of the entire screen.
# The 2 return values serve to calculate the normalized iris features
def get_face_plane(points3d):
  points2d = []
  for point3d in points3d:
    x, y, z = point3d
    points2d.append([x, y])
  pt1, pt2, pt3, pt4 = points2d
  xaxis = get_line(pt1, pt2)
  yaxis = get_line(pt3, pt4)
  origin = get_intersection(xaxis, yaxis)
  v = []
  for a, b in zip(pt1, pt2):
    v.append(b - a)
  rad = 2 * math.pi - get_ccw_angle(v)
  return origin, rad

# Return a set of normalized iris features relative to the face coordinate
# system given the original iris features, origin and the counter-clockwise
# angle of the face coordinate system relative to the entire screen
def rotate(origin, points, angle):
  ox, oy = origin
  normalized_points = []
  for point in points:
    px, py, pz = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    nx = qx - ox
    ny = qy - oy
    normalized_points.append([nx, ny])
  return normalized_points

# Return the 6-dimensional face representation and the normalized iris features
# given a list of subject videos
def predict_and_normalize(videos):
  face_frames = []
  normalized_irises = []
  for video in videos:
    frames = video["features"]
    for frame in frames:
      face_frame = [frame[i] for i in keypoints]
      face_frames.append(face_frame)
      irises_data = [frame[i] for i in irises]
      o, r = get_face_plane([frame[i] for i in face_cross])
      normalized_data = rotate(o, irises_data, r)
      normalized_irises.append(normalized_data)
  # The latent features are eventually converted because vae_encoder.predict()
  # only supports a 3D tensor as the input
  latent_features = vae_encoder.predict(face_frames)
  latent_features = latent_features.tolist()
  return latent_features, normalized_irises

# Return a list of videos with the given 'block' attribute
def get_block_data(block_num, subject_data):
  block_data = []
  for video in subject_data:
    if video['block'] == block_num:
      block_data.append(video)
  return block_data

# Return 2 lists of videos for calibration and test, respectively,
# given a list of videos share the same 'block' attribute
def get_ct_data(vlst):
  calibration_data = []
  test_data = []
  for video in vlst:
    if video['phase'] == 'calibration':
      calibration_data.append(video)
    else:
      test_data.append(video)
  return calibration_data, test_data

# Sort calibration data respective to the reference order
def sort_calibration(c_data):
  sorted_data = []
  for pt in calibration_pts:
    x = pt[0]
    y = pt[1]
    for video in c_data:
      if video['x'] == x and video['y'] == y:
        sorted_data.append(video)
  return sorted_data

# # The main processing function

# In[ ]:


# This giant function takes the json file of one individual subject as input
# and processes it. Eventually, it returns a list of inputs and its 
# corresponding list of target gaze points.

def get_inputs_targets(subject_name, subject_data):

  print('Begin to process subject ' + subject_name + '...')

  latent_features, normalized_irises = predict_and_normalize(subject_data)

  # latent_features (3605 x 6) and normalized_irises (3605 x 8 x 2) are two 
  # separate lists. We may want to merge them together for convenience so that
  # every element in aggregate_features contains all crucial information of the
  # face in one frame

  aggregate_features = []

  for a, b in zip(latent_features, normalized_irises):
    aggregate_features.append([a, b])

  print('Face representation and normalized iris features sorted...')

  # We want to make a copy of subject_data and replace the 'features' content
  # of every video with information of the latent features and normalized irises.
  # We want to put these info back to the dictionary because eventually we need
  # to sort the input for the incoming deep learning model according to the 
  # 'phase' and the 'block' attributes

  # This counter records number of frames processed.
  # It updates per video processed
  frames_counter = 0

  subject_data_copy = subject_data.copy()

  print('Made a copy of the subject ' + subject_name + '...')

  # Loop through videos
  for video in subject_data_copy:
    # Check number of frames of the video
    frames_num = len(video['features'])
    # Index of the first element we want from aggregate_features
    head = frames_counter
    # Index of the first element we want from aggregate_features FOR THE NEXT VIDEO
    tail = head + frames_num
    # Rewrite the 'features' attribute
    video['features'] = [aggregate_features[i] for i in range(head, tail)]
    # Update counter
    frames_counter = tail

  print('\"features\" attribute rewritten...')

  # Inputs list for the deep learning model
  inputs = []
  # Targets list corresponding to the inputs list
  targets = []

  # Loop through blocks
  for i in ['0', '1', '2']:
    block_data = get_block_data(i, subject_data_copy)
    c_data, t_data = get_ct_data(block_data)
    # Check if there are 13 calibration videos for this block
    if len(c_data) < 13:
      # If not, print error messages and return
      print('ERROR MESSAGE: ' + subject_name + ' has insufficient number of calibration data in block ' + i + ': ' + str(len(c_data)))
      return False
    c_data = sort_calibration(c_data)
    c_v_frames = []
    for c in c_data:
      c_v_frames.append(len(c['features']))
    # Loop through test videos first. The targets depend on the test videos NOT the calibration videos
    for t_video in t_data:
      # Note down the target gaze coordinate
      target = [int(t_video['x']), int(t_video['y'])]
      # Declare an individual input list. There should be eventually 14 elements in it: 1 test video, all 13 calibration videos
      input = []
      t_frames = t_video['features']
      try:
        # Try random frame selection
        input.append(random.choice(t_frames))
      except:
        # Catch the exception and go to the next test video
        x, y = target
        print('ERROR MESSAGE: ' + subject_name + ' has a test video with 0 frames')
        print('-----> block ' + i + ', (' + x + ', ' + y + ')')
        continue
      # For every test video, append all calibration videos info
      c_idx = 0
      for c_video in c_data:
        c_frames = c_video['features']
        try:
          # Try random frame selection
          input.append(random.choice(c_frames))
          c_idx += 1
        except:
          # Catch the exception and stop processing the entire subject
          x, y = calibration_pts[c_idx]
          print('ERROR MESSAGE: ' + subject_name + ' has a calibration video with 0 frames')
          print('-----> block ' + i + ', (' + x + ', ' + y + ')')
          return False
      inputs.append(input)
      targets.append(target)

  return inputs, targets

# # Load subject json files and export inputs json files

# In[ ]:


# At first load all subject json files from the json folder and
# store them into a giant dictionary called json_data

json_path = '/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/prolific_eye_data_experiment/json/'
all_json_files = os.listdir(json_path)

json_data = {}
i = 0

for filename in all_json_files:
    # *** TEMPORARY SOLUTION ***
    # Control amount of json files stored in json_data to avoid overloading RAM
    # Change conditions to get different set of files (limit appears to be ~30)
    if (i >= 120 and i <= 148): 
        with open(json_path + filename, 'r') as file:
            s_data = json.load(file)
            json_data = {**json_data, **s_data}
    i += 1
for k, v in json_data.items():
  print(k)

# In[ ]:


# Loop through every subject info in json_data and store inputs info of
# every subject as a dictionary, eventually export it in the json format
# to the folder called json_inputs

for k, v in json_data.items():
  try:
    i, t = get_inputs_targets(k, v)
    subject_json = {
        k: {
            'x': i,
            'y': t
        }
    }
    with open('/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/json_inputs/'+k+'_inputs.json', 'w') as file:
      json.dump(subject_json, file)
      print('Subject ' + k + ' data has been successfully exported!\n')
  except:
    print('NOTICE: Subject ' + k + ' is abandoned\n')
    continue
