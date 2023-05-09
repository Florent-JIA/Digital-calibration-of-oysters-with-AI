import os
import glob
from PIL import Image

input_dir = r'photos'
output_dir = r'photos1'

for folder_name in os.listdir(input_dir):
    input_folder_path = os.path.join(input_dir, folder_name)
    output_folder_path = os.path.join(output_dir, folder_name)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    img_paths = glob.glob(os.path.join(input_folder_path, '*.jpg'))

    for img_path in img_paths:
        output_path = os.path.splitext(img_path.replace(input_dir, output_dir))[0] + '.png'

        with Image.open(img_path) as im:
            im.save(output_path)