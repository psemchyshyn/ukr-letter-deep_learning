import shutil
import os

datafolder = "letters"
for folder in os.listdir(datafolder):
    path = os.path.join(datafolder, folder)
    if not path.endswith("_1"):
        shutil.rmtree(path)
        continue
    for f in os.listdir(path):
      if not f.endswith(".jpg"):
        p = os.path.join(path, f)
        shutil.rmtree(p)
    