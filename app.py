import numpy as np
from flask import Flask, request
from flask_restful import Resource, Api
import pickle
from ultralytics import YOLO
import cv2
import faiss
import os
from werkzeug.utils import secure_filename
from PIL import Image
import io

MYYOLO = YOLO('./faiss.pt')
class_name_label = ['sunglass','hat','jacket','shirt','pants','shorts','skirt','dress','bag','shoe']
interest_label = ['jacket','dress','bag','shoe']

UPLOAD_FOLDER = './upload'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
api = Api(app)

FAISS_INDEX = {}  # Create a dictionary to store the indexes
GD_NO_LIST_DICT = {}

for label in interest_label:
  subfolder = 'faiss_data'
  file_path = os.path.join(subfolder, f"faiss-{label}.pkl")
  with open(file_path, "rb") as f:
    FAISS_INDEX[label] = faiss.deserialize_index(pickle.load(f))

for label in interest_label:
  subfolder = 'faiss_data'
  file_path = os.path.join(subfolder, f"gdnolist-{label}.txt")
  # file_path = f"gdnolist-{label}.pkl" 
  with open(file_path, "r") as file:
    GD_NO_LIST_DICT[label] = [line.strip() for line in file]

def get_crop_clothing_pieces_image(img_path):
  # print(img_path)
  faiss_img_vector_list = []
  img_list = []
  
  img = cv2.imread(img_path)
  result = MYYOLO.predict(img_path)

  boxes = result[0].boxes
  # print(boxes)

  num_of_detected_pieces = len(boxes.cls)    
  for i in range(num_of_detected_pieces):
    label_index = boxes.cls[i]
    label = class_name_label[int(label_index)]
    if label in interest_label:
      if result[0] is not None:
        x1, y1, w, h = boxes.xywh[i]
        x1, y1, w, h = int(x1), int(y1), int(w), int(h)
        x_start = round(x1 - (w/2))
        y_start = round(y1 - (h/2))
        x_end = round(x_start + w)
        y_end = round(y_start + h)
        cur_img = img[y_start:y_end, x_start:x_end]
        cur_img = cv2.resize(cur_img,(100,100),interpolation=cv2.INTER_AREA)
        cur_img = cv2.cvtColor(cur_img,cv2.COLOR_BGR2RGB)
        img_list.append(cur_img)
        #convert to float32 and normalize it by /255
        float32_img = np.array(cur_img).astype(np.float32) / 255.0
        faiss_img_vector_list.append([label, float32_img])
  return faiss_img_vector_list

def format_img_vector(input_image_vector):
  query_vector = np.array([input_image_vector], dtype=np.float32)
  flattened_query_vector = query_vector.reshape(query_vector.shape[0], -1)
  return flattened_query_vector

class RecommendSimilarProduct(Resource):
  def post(self):
    if not os.path.exists(UPLOAD_FOLDER):
      os.makedirs(UPLOAD_FOLDER)

    file = request.files['file']
    filename = secure_filename(file.filename)
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(img_path)

    #detect clothing pieces
    faiss_img_vector_list = get_crop_clothing_pieces_image(img_path)
    result = []

    for i in range(len(faiss_img_vector_list)):
      label = faiss_img_vector_list[i][0]
      input_image_vector = faiss_img_vector_list[i][1]
      flattened_query_vector = format_img_vector(input_image_vector)
      #find k=3 similar product 
      _, I = FAISS_INDEX[label].search(flattened_query_vector, 3)
      dict_result = {}
      dict_result['label'] = label
      #dict_result['gd_nos'] = I[0].tolist()
      gd_nos = []
      list_index_gd = I[0].tolist()
      for i in list_index_gd:
        gd_nos.append(GD_NO_LIST_DICT[label][i])
      dict_result['gd_nos'] = gd_nos
      print(gd_nos)
      result.append(dict_result)

    result_json = {
      "result": result
    }
    return result_json
  
api.add_resource(RecommendSimilarProduct, "/api/recommend")

if __name__ == "__main__":
    app.run(debug=True)
