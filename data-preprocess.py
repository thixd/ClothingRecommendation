import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2
import shutil
import tqdm
import glob
from ultralytics import YOLO
import faiss
import pickle

myyolo = YOLO('./best.pt')
faiss_img_vector_list = []
img_list=[]

class_name_label = ['sunglass','hat','jacket','shirt','pants','shorts','skirt','dress','bag','shoe']
interest_label = ['jacket','dress','bag','shoe']

def get_crop_clothing_pieces_image(img_path, gd_no):
  # print("get_crop_clothing_pieces_image")
  cur_img = cv2.imread(img_path)
  cur_img = cv2.resize(cur_img,(100,100),interpolation=cv2.INTER_AREA)
  cur_img = cv2.cvtColor(cur_img,cv2.COLOR_BGR2RGB)
  img_list.append(cur_img)
  #convert to float32 and normalize it by /255
  float32_img = np.array(cur_img).astype(np.float32) / 255.0
  faiss_img_vector_list.append(float32_img)
  # result = myyolo.predict(img_path)
  # boxes = result[0].boxes
  # num_of_detected_pieces = len(boxes.cls)    
  # for i in range(num_of_detected_pieces):
  #     label_index = boxes.cls[i]
  #     label = class_name_label[int(label_index)]

  #     if result[0] is not None:
  #         x1, y1, w, h = boxes.xywh[i]
  #         x1, y1, w, h = int(x1), int(y1), int(w), int(h)
  #         x_start = round(x1 - (w/2))
  #         y_start = round(y1 - (h/2))
  #         x_end = round(x_start + w)
  #         y_end = round(y_start + h)
  #         cur_img = img[y_start:y_end, x_start:x_end]
  #         cur_img = cv2.resize(cur_img,(100,100),interpolation=cv2.INTER_AREA)
  #         cur_img = cv2.cvtColor(cur_img,cv2.COLOR_BGR2RGB)
  #         img_list.append(cur_img)
  #         #convert to float32 and normalize it by /255
  #         float32_img = np.array(cur_img).astype(np.float32) / 255.0
  #         faiss_img_vector_list.append(float32_img)
  #         #faiss_img_vector = np.append(faiss_img_vector, float32_img)
    

def get_pickel(folder_name):
  faiss_img_vector = np.array(faiss_img_vector_list, dtype=np.float32)
  flattened_images = faiss_img_vector.reshape(faiss_img_vector.shape[0], -1)

  # Then, create the Faiss index
  index = faiss.IndexFlatL2(flattened_images.shape[1])

  # Add the flattened vectors to the index
  index.add(flattened_images)

  chunk = faiss.serialize_index(index)
  #index3 = faiss.deserialize_index(np.load("index.npy"))   # identical to index

  filename = f"faiss-{folder_name}.pkl"

  with open(filename, "wb") as f:
      pickle.dump(chunk, f)


for label in interest_label:
  folder_path = label
  files = os.listdir(folder_path)

  faiss_img_vector_list = []
  gd_no_list = []
  # Loop through each file in the folder
  for file in files:
    file_path = os.path.join(folder_path, file)
    gd_no, file_extension = os.path.splitext(file)
    get_crop_clothing_pieces_image(file_path, gd_no)
    gd_no_list.append(gd_no)
  
  get_pickel(label)

  gd_no_name = f"gdnolist_{folder_path}.txt"
  with open(gd_no_name, "w") as file:
    for gd_no in gd_no_list:
        file.write(f"{gd_no}\n")