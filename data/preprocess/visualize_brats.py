import glob, os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

t1 = glob.glob("/home/lsl/Research/asyndgan/datasets/brats18/HGG/Brats18_2013_2_1/Brats18_2013_2_1_t1.nii.gz")
t2 = glob.glob("/home/lsl/Research/asyndgan/datasets/brats18/HGG/Brats18_2013_2_1/Brats18_2013_2_1_t2.nii.gz")
t1ce = glob.glob("/home/lsl/Research/asyndgan/datasets/brats18/HGG/Brats18_2013_2_1/Brats18_2013_2_1_t1ce.nii.gz")
seg = glob.glob("/home/lsl/Research/asyndgan/datasets/brats18/HGG/Brats18_2013_2_1/Brats18_2013_2_1_seg.nii.gz")
flair = glob.glob("/home/lsl/Research/asyndgan/datasets/brats18/HGG/Brats18_2013_2_1/Brats18_2013_2_1_flair.nii.gz")

gen_path = "/home/lsl/Research/asyndgan/data/preprocess/img_view"
if not os.path.exists(gen_path):
    os.mkdir(gen_path)

def readImg(im_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(im_path))

# ims = readImg(t1)

slice_layer = 100
types = {'t1':t1,'t2':t2,'t1ce':t1ce,'seg':seg,'flair':flair}
# 将五类图片生成至gen_path
for t in types:
    TwoDimg = readImg(types[t][0])
    # import pdb;pdb.set_trace()
    img = TwoDimg[slice_layer].astype(np.uint8)
    img = Image.fromarray(img).convert("RGB")
    if t == 'seg':
        img = ImageEnhance.Contrast(img).enhance(60)
    print(np.unique(img))
    img.save(f"{gen_path}/{t}_layer{slice_layer}.png")
