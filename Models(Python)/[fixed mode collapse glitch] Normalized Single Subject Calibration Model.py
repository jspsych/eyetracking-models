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
import keras

# In[ ]:


input_json_path = '/content/drive/Shareddrives/URSI 2022/Eye Tracking ML/json_inputs/'

all_input_json_files = os.listdir(input_json_path)

input_json_data = {}
with open(input_json_path + all_input_json_files[36], 'r') as file:
    s_data = json.load(file)
    input_json_data = {**input_json_data, **s_data}

# In[ ]:


calibration_points = [[10, 50], [10, 10], [90, 10], [50, 90],
                   [30, 70], [50, 50], [50, 10], [90, 90],
                   [70, 70], [70, 30], [10, 90], [90, 50],
                   [30, 30]]

# In[ ]:


train_y = []

for subject in input_json_data:
    for y in input_json_data[subject]['y']:
        for element in calibration_points:
            train_y.append(element)

print(len(train_y))
#print(train_y)

# In[ ]:


train_x = []

for subject in input_json_data:
    for sample in input_json_data[subject]['x']:
        calibration_arr = []
        i = 0
        while (i < 13):
            total_arr = []
            head_pose = np.array(sample[i][0])
            iris_points = np.array(sample[i][1])
            head_pose = head_pose.flatten()
            iris_points = iris_points.flatten()
            #for element in head_pose:
                #total_arr.append(element)
            for element in iris_points:
                total_arr.append(element)
            calibration_arr.append(total_arr)
            i += 1
        #train_x.append(calibration_arr[0])
        for element in calibration_arr:
            train_x.append(element)

# In[ ]:


print(np.shape(train_x))
print(np.shape(train_y))

# In[ ]:


val_y = []

for subject in input_json_data:
    for y in input_json_data[subject]['y']:
        val_y.append(y)

# In[ ]:


val_x = []

for subject in input_json_data:
    for sample in input_json_data[subject]['x']:
        calibration_arr = []
        i = 0
        while (i <= 13):
            if (i == 13):
                total_arr = []
                head_pose = np.array(sample[i][0])
                iris_points = np.array(sample[i][1])
                head_pose = head_pose.flatten()
                iris_points = iris_points.flatten()
                #for element in head_pose:
                    #total_arr.append(element)
                for element in iris_points:
                    total_arr.append(element)
                calibration_arr.append(total_arr)
            i += 1
        #train_x.append(calibration_arr[0])
        for element in calibration_arr:
            val_x.append(element)

# In[ ]:


print(np.shape(val_x))
print(np.shape(val_y))

# In[ ]:


#train_x = []
#train_y = []

# In[ ]:


append_train_x = val_x[:45]
for element in append_train_x:
    train_x.append(element)
val_x = val_x[45:]

append_train_y = (val_y[:45])
for element in append_train_y:
    train_y.append(element)
val_y = val_y[45:]

# In[ ]:


print(np.shape(train_x))
print(np.shape(train_y))
print(np.shape(val_x))
print(np.shape(val_y))

# In[ ]:


train_x = np.array(train_x)
mean = train_x.mean(axis=0)
train_x -= mean
std = train_x.std(axis=0)
train_x /= std
train_y = np.array(train_y)

val_x = np.array(val_x)
val_x -= mean
val_x /= std
val_y = np.array(val_y)

# In[ ]:


print(train_x[1])
print(val_x[1])

# In[ ]:


from sklearn.utils import shuffle

train_x, train_y = shuffle(train_x, train_y)

# In[ ]:


print(np.shape(train_x))
print(np.shape(train_y))

# In[ ]:


from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([layers.Dense(128, activation="relu"), layers.Dense(64, activation="relu"), layers.Dense(16, activation="relu"), layers.Dense(2, activation="relu")])
model.compile(optimizer=keras.optimizers.RMSprop(1e-2), loss="MeanSquaredError", metrics=["mae"])

history = model.fit(train_x, train_y, epochs=70, batch_size=40000, validation_data=(val_x, val_y))

# In[ ]:


plt.xlim(0, 100)
plt.ylim(100, 0)
x2 = []
y2 = []
sample_num = 0
while sample_num < 15:
    x = [train_y[sample_num][0]]
    x2 = [(train_y[sample_num][0])]
    x.append(model.predict(train_x)[sample_num][0])
    x3 = [model.predict(train_x)[sample_num][0]]
    y = [train_y[sample_num][1]]
    y2 = [(train_y[sample_num][1])]
    y.append(model.predict(train_x)[sample_num][1])
    y3 = [model.predict(train_x)[sample_num][1]]
    plt.plot(x, y, color="black", marker="o", markersize=5)
    plt.plot(x3, y3, color="red", marker="o", markersize=5)
    plt.plot(x2, y2, color="blue", marker="o", markersize=5)
    sample_num += 1

# In[ ]:


plt.xlim(0, 100)
plt.ylim(100, 0)
x2 = []
y2 = []
sample_num = 0
while sample_num < 15:
    x = [val_y[sample_num][0]]
    x2 = [(val_y[sample_num][0])]
    x.append(model.predict(val_x)[sample_num][0])
    x3 = [model.predict(val_x)[sample_num][0]]
    y = [val_y[sample_num][1]]
    y2 = [(val_y[sample_num][1])]
    y.append(model.predict(val_x)[sample_num][1])
    y3 = [model.predict(val_x)[sample_num][1]]
    plt.plot(x, y, color="black", marker="o", markersize=5)
    plt.plot(x3, y3, color="red", marker="o", markersize=5)
    plt.plot(x2, y2, color="blue", marker="o", markersize=5)
    sample_num += 1



# In[ ]:


val_x[1]

# In[ ]:


import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "r", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# In[ ]:


plt.clf()
acc = history_dict["mae"]
val_acc = history_dict["val_mae"]
plt.plot(epochs, acc, "r", label="Training mae")
plt.plot(epochs, val_acc, "b", label="Validation mae")
plt.title("Training and validation mae")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()
plt.show()
