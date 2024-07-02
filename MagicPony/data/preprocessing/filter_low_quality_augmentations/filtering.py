import os
import glob

import torch
from ultralytics import YOLO

device=torch.device("cuda:0") # GPU device

# get the list of image folders
image_folders = glob.glob("/YOUR-PATH/train/*")
image_folders.sort()

# load the YOLO-World model
model = YOLO('yolov8x-worldv2.pt')  # or choose yolov8m/l-world.pt
classes = ["others", "bird"] # the class that we are interested in, in this case it is birds
model.set_classes(classes)

removed_view_folder_count = 0
for i in range(len(image_folders)):
  count = 0
  image_paths = glob.glob(image_folders[i] + "/*view*rgb.png")
  for image_path in image_paths:
    results = model.predict(image_path, verbose=False, device=device)
    bird_counts = 0
    for j, c in enumerate(results[0].boxes.cls):
      if c.cpu().numpy() == 1.0 and results[0].boxes.conf.cpu().numpy()[j] > 0.7:
        bird_counts += 1
    if bird_counts == 0:
      break
    count += 1
  
  if count == 5:
    # copy folder to filtered_dataset
    if not os.path.exists("/YOUR-PATH/train"):
      os.mkdir("/YOUR-PATH/train")
    os.system(f"cp -r {image_folders[i]} /YOUR-PATH/train")
  else:
    removed_view_folder_count += 1

  print(removed_view_folder_count)
