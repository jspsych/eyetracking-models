#!/usr/bin/env python
# coding: utf-8

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

def get_tc_data(dict):
  all_videos_data = []
  for key, value in dict.items():
    for per_video_data in value:
      all_videos_data.append(per_video_data)
  calibration_data = list(filter(is_calibration_data, all_videos_data))
  test_data = list(filter(is_test_data, all_videos_data))
  return calibration_data, test_data

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



# model_data = {}

# for key, value in json_data.items():
#   block_counter = 0
#   subject_data = []

#   while block_counter < 3:
#     block_data = []
#     block_name = str(block_counter)

#     for video in value:
#       if is_block_n_data(video, block_name):
#         block_data.append(video)

#     calibration_data = list(filter(is_calibration_data, block_data))
#     test_data = list(filter(is_test_data, block_data))
#     c_training_data, c_target_data = split_and_flatten_data(calibration_data)
#     t_predictor_data, t_target_data = split_and_flatten_data(test_data)

#     ridge_model = Ridge()
#     ridge_model.fit(c_training_data, c_target_data, None)
#     predictions = ridge_model.predict(t_predictor_data).tolist()

#     residual_distance = []
#     tx, ty = split_xy(t_target_data)
#     px, py = split_xy(predictions)
#     residual_x = px - tx
#     residual_y = py - ty
#     for i in range(len(predictions)):
#       d = math.sqrt(residual_x[i]**2 + residual_y[i]**2)
#       residual_distance.append(d)
#     mean_d = statistics.mean(residual_distance)

#     subject_data.append({
#         "block": block_name,
#         "calibration data": calibration_data,
#         "test data": t_target_data,
#         "predicted data": predictions,
#         "residual d": residual_distance,
#         "mean d": mean_d,
#         "residual x": residual_x,
#         "residual y": residual_y 
#     })

#     block_counter+=1

#   model_data[key] = subject_data


# for key, value in model_data.items():
#   for v in value:
#     print(v["mean d"])

# test_data = []
# predictions = []
# residual_x = []
# residual_y = []
# residual_d = []

# for key, value in model_data.items():
#   for v in value:
#     test_data.append(v["test data"])
#     predictions.append(v["predicted data"])
#     residual_x.append(v["residual x"])
#     residual_y.append(v["residual y"])
#     residual_d.append(v["residual d"])

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

# In[ ]:


# c_data, t_data = get_tc_data(json_data)
# c_training_data, c_target_data = split_and_flatten_data(c_data)
# t_predictor_data, t_output_data = split_and_flatten_data(t_data)

# ridge_model = Ridge()
# ridge_model.fit(c_training_data, c_target_data, None)

# predictions = ridge_model.predict(t_predictor_data)
# print(tf.shape(predictions))

# draw_one_subject(predictions)
# draw_one_subject(t_output_data)
# plt.show()

# t_x, t_y = split_xy(t_output_data)
# p_x, p_y = split_xy(predictions)

# x_residuals = p_x - t_x
# y_residuals = p_y - t_y
# plt.hist(x_residuals, bins = 100)
# plt.show()
# plt.hist(y_residuals, bins = 100)
# plt.show()

# In[ ]:


# model_data = {}

# for key, value in json_data.items():
#   calibration_data = list(filter(is_calibration_data, value))
#   test_data = list(filter(is_test_data, value))
#   c_training_data, c_target_data = split_and_flatten_data(calibration_data)
#   t_predictor_data, t_target_data = split_and_flatten_data(test_data)

#   ridge_model = Ridge()
#   ridge_model.fit(c_training_data, c_target_data)
#   predictions = ridge_model.predict(t_predictor_data)

#   residual_distance = []
#   tx, ty = split_xy(t_target_data)
#   px, py = split_xy(predictions)
#   residual_x = px - tx
#   residual_y = py - ty
#   for i in range(len(predictions)):
#     d = math.sqrt(residual_x[i]**2 + residual_y[i]**2)
#     residual_distance.append(d)
#   mean_d = statistics.mean(residual_distance)

#   model_data[key] = {
#       "calibration data": calibration_data,
#       "test data": t_target_data,
#       "predicted data": predictions,
#       "residual d": residual_distance,
#       "mean d": mean_d,
#       "residual x": residual_x,
#       "residual y": residual_y 
#   }

# for key, value in model_data.items():
#   print(value["mean d"])

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

# In[ ]:


# def landmark_filter(lst):
#   return [lst[i] for i in [469, 470, 471, 472, 474, 475, 476, 477, 35, 159, 133, 145, 263, 386, 362, 274, 2, 1, 5, 218, 438, 61, 291, 13, 152, 234, 454, 10]]

# def split_and_flatten_data(v_list):
#   predictor_data = []
#   output_data = []
#   for video in v_list:
#     coor = [int(video["x"]), int(video["y"])]
#     features = video["features"]
#     for feature in features:
#       feature = landmark_filter(feature)
#       predictor_data.append(feature)
#       output_data.append(coor)
#   predictor_data = np.reshape(np.array(predictor_data), (-1, len(feature)*3)).tolist()
#   return predictor_data, output_data

# model_data = {}

# for key, value in json_data.items():
#   calibration_data = list(filter(is_calibration_data, value))
#   test_data = list(filter(is_test_data, value))
#   c_training_data, c_target_data = split_and_flatten_data(calibration_data)
#   t_predictor_data, t_target_data = split_and_flatten_data(test_data)

#   ridge_model = Ridge()
#   ridge_model.fit(c_training_data, c_target_data)
#   predictions = ridge_model.predict(t_predictor_data)

#   residual_distance = []
#   tx, ty = split_xy(t_target_data)
#   px, py = split_xy(predictions)
#   residual_x = px - tx
#   residual_y = py - ty
#   for i in range(len(predictions)):
#     d = math.sqrt(residual_x[i]**2 + residual_y[i]**2)
#     residual_distance.append(d)
#   mean_d = statistics.mean(residual_distance)

#   model_data[key] = {
#       "calibration data": calibration_data,
#       "test data": t_target_data,
#       "predicted data": predictions,
#       "residual d": residual_distance,
#       "mean d": mean_d,
#       "residual x": residual_x,
#       "residual y": residual_y 
#   }

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
