#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import os
import fnmatch
import json
import math
!pip install mediapipe
import mediapipe as mp
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

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


def calculate_mesh_points_variance(path):
    deviations = []
    meshfeatures = extract_mesh_from_video(path)
    i = 0;
    while(i < len(meshfeatures[0])):
        point_x_across_frames = []
        for frame in meshfeatures:
            point_x_across_frames.append(frame[i])

        x_point = 0;
        y_point = 0;
        z_point = 0;
        for point in point_x_across_frames:
            x_point += point[0]
            y_point += point[1]
            z_point += point[2]
        x_mean = x_point / len(point_x_across_frames)
        y_mean = y_point / len(point_x_across_frames)
        z_mean = z_point / len(point_x_across_frames)

        sum_x_diffs = 0;
        sum_y_diffs = 0;
        sum_z_diffs = 0;

        for point in point_x_across_frames:
            sum_x_diffs += (point[0] - x_mean) * (point[0] - x_mean)
            sum_y_diffs += (point[1] - y_mean) * (point[1] - y_mean)
            sum_z_diffs += (point[2] - z_mean) * (point[2] - z_mean)

        standard_deviation_x = math.sqrt(sum_x_diffs / len(point_x_across_frames))
        standard_deviation_y = math.sqrt(sum_y_diffs / len(point_x_across_frames))
        standard_deviation_z = math.sqrt(sum_z_diffs / len(point_x_across_frames))

        deviations.append([standard_deviation_x, standard_deviation_y, standard_deviation_z])
        i += 1
    print(len(deviations))
    return deviations

# In[ ]:


def calculate_overall_mesh_variance(path):
    deviation_arr = calculate_mesh_points_variance(path)
    x_mean = 0;
    y_mean = 0;
    z_mean = 0;

    for arr in deviation_arr:
        x_mean += arr[0]
        y_mean += arr[1]
        z_mean += arr[2]
    
    x_mean = x_mean / len(deviation_arr)
    y_mean = y_mean / len(deviation_arr)
    z_mean = z_mean / len(deviation_arr)
    return [x_mean, y_mean, z_mean]

# In[ ]:


# get the path of the webm file
path = '/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/prolific_eye_data_experiment/153_sample_webm/'
all_files = os.listdir(path)

# In[ ]:


ten8qa9r = []
for filename in all_files:
    if "6zkff4vo" in filename:
        ten8qa9r.append(filename)

# In[ ]:


ax = plt.axes(xlim=(0,1),ylim=(1,0))

x = []
y = []

for filename in ten8qa9r:
    face_mesh_arr = ((extract_mesh_from_video(path + filename)))

    for face_mesh_points in face_mesh_arr:
        for point in face_mesh_points:
            x.append(point[0])
            y.append(point[1])
ax.scatter(x, y, c='red', s=1)

# In[ ]:


ax = plt.axes(xlim=(0,1),ylim=(1,0))
x = []
y = []

for filename in ten8qa9r:
    face_mesh_arr = ((extract_mesh_from_video(path + filename)))
    for face_mesh_points in face_mesh_arr:  
        x.append(face_mesh_points[10][0])
        y.append(face_mesh_points[10][1])
ax.scatter(x, y, c='red', s=1)

# In[ ]:


for filename in all_files:
    print(filename, calculate_overall_mesh_variance(path + filename))

# In[ ]:


import math

for filename in all_files:
    meshfeatures = extract_mesh_from_video(path + filename)

    point1_across_frames = []
    for frame in meshfeatures:
        point1_across_frames.append(frame[0])

    x_point = 0;
    y_point = 0;
    z_point = 0;
    for point in point1_across_frames:
        x_point += point[0]
        y_point += point[1]
        z_point += point[2]
    x_mean = x_point / len(point1_across_frames)
    y_mean = y_point / len(point1_across_frames)
    z_mean = z_point / len(point1_across_frames)

    sum_x_diffs = 0;
    sum_y_diffs = 0;
    sum_z_diffs = 0;

    for point in point1_across_frames:
        sum_x_diffs += (point[0] - x_mean) * (point[0] - x_mean)
        sum_y_diffs += (point[1] - y_mean) * (point[1] - y_mean)
        sum_z_diffs += (point[2] - z_mean) * (point[2] - z_mean)

    standard_deviation_x = math.sqrt(sum_x_diffs / len(point1_across_frames))
    standard_deviation_y = math.sqrt(sum_y_diffs / len(point1_across_frames))
    standard_deviation_z = math.sqrt(sum_z_diffs / len(point1_across_frames))

    print(filename, standard_deviation_x, standard_deviation_y, standard_deviation_z)

# In[ ]:


filename = '4ypfnlhr_2_test_35_48.webm'
meshfeatures = extract_mesh_from_video(path + filename)
print(meshfeatures)
