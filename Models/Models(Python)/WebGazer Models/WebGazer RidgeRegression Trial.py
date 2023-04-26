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


from PIL import Image
from google.colab.patches import cv2_imshow

leftEyeTopArcKeypoints = [25, 33, 246, 161, 160, 159, 158, 157, 173, 243]
leftEyeBottomArcKeypoints = [25, 110, 24, 23, 22, 26, 112, 243]
rightEyeTopArcKeypoints = [463, 398, 384, 385, 386, 387, 388, 466, 263, 255]
rightEyeBottomArcKeypoints = [463, 341, 256, 252, 253, 254, 339, 255]



def getBox(top, bottom, w, h):
  topLeftOrigin = {
      "x": round(min([a[0] for a in top]) * w),
      "y": round(min([a[1] for a in top]) * h)
  }
  bottomRightOrigin = {
      "x": round(max([a[0] for a in bottom]) * w),
      "y": round(max([a[1] for a in bottom]) * h)
  }
  return {
      "origin": topLeftOrigin,
      "width": bottomRightOrigin["x"] - topLeftOrigin["x"],
      "height": bottomRightOrigin["y"] - topLeftOrigin["y"]
  }



def resizeEye(eyeImage):
  return cv2.resize(eyeImage, (10, 6), interpolation = cv2.INTER_AREA)



def equalizeHistogram(eyeImage):
  gray = cv2.cvtColor(eyeImage, cv2.COLOR_BGR2GRAY)
  return cv2.equalizeHist(gray)



def getEyeFeats(leftEye, rightEye):
  resizeLeft = resizeEye(leftEye)
  # cv2_imshow(resizeLeft)
  resizeRight = resizeEye(rightEye)
  leftData = equalizeHistogram(resizeLeft)
  # cv2_imshow(leftData)
  rightData = equalizeHistogram(resizeRight)
  return np.concatenate((leftData.flatten(), rightData.flatten())).tolist()



def getEyeData(path):
  cap = cv2.VideoCapture(path)
  out = []

  while(cap.isOpened()):
    width = cap.get(3)
    height = cap.get(4)

    ret, frame = cap.read()
    if frame is not None:
      results = face_mesh.process(frame)
      if not results.multi_face_landmarks:
        continue
      landmarks = results.multi_face_landmarks[0].landmark
      lm = [[a.x, a.y] for a in landmarks]

      leftTop = [lm[i] for i in leftEyeTopArcKeypoints]
      leftBottom = [lm[i] for i in leftEyeBottomArcKeypoints]
      rightTop = [lm[i] for i in rightEyeTopArcKeypoints]
      rightBottom = [lm[i] for i in rightEyeBottomArcKeypoints]

      leftBox = getBox(leftTop, leftBottom, width, height)
      rightBox = getBox(rightTop, rightBottom, width, height)

      leftOriginX = leftBox["origin"]["x"]
      leftOriginY = leftBox["origin"]["y"]
      leftWidth = leftBox["width"]
      leftHeight = leftBox["height"]
      rightOriginX = rightBox["origin"]["x"]
      rightOriginY = rightBox["origin"]["y"]
      rightWidth = rightBox["width"]
      rightHeight = rightBox["height"]

      if (leftWidth == 0 or leftHeight == 0 or rightWidth == 0 or rightHeight == 0):
        continue
      leftPatch = frame[leftOriginY:leftOriginY + leftHeight, leftOriginX:leftOriginX + leftWidth]
      rightPatch = frame[rightOriginY:rightOriginY + rightHeight, rightOriginX:rightOriginX + rightWidth]

      pixels = getEyeFeats(leftPatch, rightPatch)
      out.append(pixels)

    else:
      break
  return out

# In[ ]:


path = '/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/webm/v2sfzuft_2_test_95_8.webm'
example = getEyeData(path)
np.shape(example)

# In[ ]:


path = '/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/prolific_eye_data_experiment/webm/'

all_files = os.listdir(path)
print(len(all_files))

unique_subjects = set([filepath.split('_')[0] for filepath in os.listdir(path)])
print(unique_subjects)

for subject in unique_subjects:
  all_data = {}

  print(subject)

  if os.path.isfile('/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/json_webgazer/'+subject+'.json'):
    continue
   
  subject_data = []
  subject_files = fnmatch.filter(all_files, subject+'*')
  for filename in subject_files:
    fileinfo = filename.replace('.','_').split('_')
    subject = fileinfo[0]
    block = fileinfo[1]
    phase = fileinfo[2]
    x = fileinfo[3]
    y = fileinfo[4]
    eyeData = getEyeData(path + filename)
    subject_data.append({
        'block': block,
        'phase': phase,
        'x': x,
        'y': y,
        'eyes': eyeData
    })
  all_data[subject] = subject_data

  with open('/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/json_webgazer/'+subject+'.json', 'w+') as file:
    json.dump(all_data, file)

# In[ ]:


json_path = '/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/json_webgazer/'
all_json_files = os.listdir(json_path)

json_data = {}
for filename in all_json_files:
  with open('/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/json_webgazer/'+filename, 'r') as file:
    s_data = json.load(file)
    json_data = {**json_data, **s_data}

# In[ ]:


def isBlockZero(video):
  return video["block"] == "0"


def isBlockOne(video):
  return video["block"] == "1"


def isBlockTwo(video):
  return video["block"] == "2" 


def isCalibration(video):
  return video["phase"] == "calibration"


def isTest(video):
  return video["phase"] == "test"


def meanAndSdEuclideanDistance(obs, pred):
  if len(obs) != len(pred): return None
  else:
    dist = []
    for i in range(len(pred)):
      d = np.linalg.norm(pred[i] - obs[i])
      dist.append(d)
    return np.mean(dist), np.std(dist)

# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import pandas as pd
import csv

df = pd.DataFrame(columns=['Subject Name', 'Coefficient of Determination', 'MAE', 'Mean Euclidean Distance', 'SD of Euclidean Distance'])
subject_names = []
cods = []
maes = []
meds = []
sds = []

# with open('/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/json_webgazer/1lzaw0tb.json', 'r') as file:
#     json_data = json.load(file)


for k, v in json_data.items():
  videos = v
  blockData = None

  for i in ["0", "1", "2"]:
    if i == "0": blockData = list(filter(isBlockZero, videos))
    elif i == "1": blockData = list(filter(isBlockOne, videos))
    else: blockData = list(filter(isBlockTwo, videos))

    calibrationData = list(filter(isCalibration, blockData))
    testData = list(filter(isTest, blockData))

    trainingFeatures = []
    trainingTargets = []
    testFeatures = []
    testTargets = []

    for data in calibrationData:
      targetData = [int(data["x"]), int(data["y"])]
      for eyeData in data["eyes"]:
        trainingFeatures.append(eyeData)
        trainingTargets.append(targetData)

    for data in testData:
      targetData = [int(data["x"]), int(data["y"])]
      for eyeData in data["eyes"]:
        testFeatures.append(eyeData)
        testTargets.append(targetData)

    ridge_model = Ridge()
    ridge_model.fit(trainingFeatures, trainingTargets, None)
    coefficientOfDetermination = ridge_model.score(testFeatures, testTargets)

    predictions = ridge_model.predict(testFeatures)

    MAE = mean_absolute_error(testTargets, predictions)
    med, sd = meanAndSdEuclideanDistance(testTargets, predictions)
    subject_names.append(k + str(i))
    cods.append(coefficientOfDetermination)
    maes.append(MAE)
    meds.append(med)
    sds.append(sd)

df['Subject Name'] = subject_names
df['Coefficient of Determination'] = cods
df['MAE'] = maes
df['Mean Euclidean Distance'] = meds
df['SD of Euclidean Distance'] = sds

# In[ ]:


df
