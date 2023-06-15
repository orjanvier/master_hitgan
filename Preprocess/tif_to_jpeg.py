import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
os.environ["TF_CPP_MIN_LOG_LEVEL"]="4"
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers




import os
from PIL import Image

yourpath = "/bhome/ovier/master/2021_v2/Code/output_data/Exp11/2023-06-02_3D_Najmv5_Slice512_123/01"
for root, dirs, files in os.walk(yourpath, topdown=False):
    for name in files:
        print(os.path.join(root, name))
        if os.path.splitext(os.path.join(root, name))[1].lower() == ".tiff":
            if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".jpg"):
                print("A jpeg file already exists for %s" % name)
            # If a jpeg is *NOT* present, create one from the tiff.
            else:
                outfile = os.path.splitext(os.path.join(root, name))[0] + ".jpg"
                try:
                    im = Image.open(os.path.join(root, name))
                    print(outfile)
                    print("Generating jpeg for %s" % name)
                    im.thumbnail(im.size)
                    im.save(outfile, "JPEG", quality=100)
                except Exception:
                    print("e")

