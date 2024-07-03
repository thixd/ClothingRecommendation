import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os

# Replace 'your_file.csv' with the path to your CSV file
csv_file = 'data.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Iterate through the DataFrame and download images
for index, row in df.iterrows():
    gd_no = row['GD_NO']
    go_to_link = row['GO_TO_LINK']
    category_cd = row['CATEGORY_CD']

    # Make a request to the image URL
    response = requests.get(go_to_link)

    if response.status_code == 200:
        # Get the content of the response
        image_data = response.content

        # Save the image with GD_NO as the filename
        image = Image.open(BytesIO(image_data))
        image = image.convert('RGB')
        # image.save(f'{gd_no}.jpg')
        image.save(os.path.join(category_cd, f'{gd_no}.jpg'))

        print(f'Saved image for GD_NO {gd_no}')
    else:
        print(f'Failed to download image for GD_NO {gd_no}')