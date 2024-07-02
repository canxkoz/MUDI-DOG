import os
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt


image_paths = glob.glob("YOUR-PATH/augmented_dataset/*/*/*.png")
new_folder = "YOUR-PATH/augmented_dataset"

for image_path in image_paths:
  image = cv2.imread(image_path)
  image_path = image_path.split("augmented_dataset")
  new_path = new_folder + image_path[1]
  cv2.imwrite(new_path, image)