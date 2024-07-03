# Clothing Recommendation

## Project Goal
The goal of this project is to recommend similar pieces of clothing based on a given outfit. By leveraging image recognition and a clothing database from ecommerce, the system identifies clothing items such as t-shirts, shoes, dresses, sunglasses, etc., in an outfit image and suggests similar items to the user.

## Technical Overview
- **Object Detection**: YOLO is used to detect various parts of an outfit in an image, such as t-shirts, shoes, dresses, sunglasses, etc.
  ![Object detection](https://github.com/thixd/ClothingRecommendation/assets/77187869/ec4fe3f1-9eb7-4e0c-8550-b8e47befad59)
  Training result
  ![Training result](https://github.com/thixd/ClothingRecommendation/assets/77187869/f29180ce-fc05-4c1c-ba65-93f84a096860)
  ![image](https://github.com/thixd/ClothingRecommendation/assets/77187869/f79d9524-8448-49ef-a70c-07c6c2cf7a36)
- **Similarity Search**: FAISS is employed to find similar clothing items in the database and recommend them to the user.

## Getting Started
To begin with this project:
1. Install Python 3.10.5
2. Install dependencies by running: pip install -r requirements.txt
3. To start the AI server in the current folder, run: python app.py (Flask server)
