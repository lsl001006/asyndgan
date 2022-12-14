import glob
import os
from os.path import join
import cv2

train = True
TARGET_SIZE = 286 # Size of the generated patch
STRIDE = TARGET_SIZE//2
ROOT = '/home/lsl/Research/asyndgan/datasets'
TRAIN_IMG = '/home/lsl/Research/asyndgan/datasets/MoNuSegTrainingData'
TEST_IMG = join(ROOT,'MoNuSegTestData')
if train:
    IMGS_DIR = join(TRAIN_IMG,'Tissue Images')
    MASKS_DIR = join(TRAIN_IMG,'Binary_masks')
    OUTPUT_DIR = join(TRAIN_IMG,'Output_'+str(TARGET_SIZE)+'_'+str(STRIDE))
    LIST_FILE = join(TRAIN_IMG,'list_'+str(TARGET_SIZE)+'_'+str(STRIDE)+'.txt') # list of all generated patch
else:
    IMGS_DIR = join(TEST_IMG,'Tissue_Images')
    MASKS_DIR = join(TEST_IMG,'Binary_masks')
    OUTPUT_DIR = join(TEST_IMG,'Output_'+str(TARGET_SIZE)+'_'+str(STRIDE))
    LIST_FILE = join(TEST_IMG,'list_'+str(TARGET_SIZE)+'_'+str(STRIDE)+'.txt') # list of all generated patch


img_paths = glob.glob(os.path.join(IMGS_DIR, "*.tif"))
mask_paths = glob.glob(os.path.join(MASKS_DIR, "*.png"))
print(os.path.exists(IMGS_DIR))
print(img_paths)
img_paths.sort()
mask_paths.sort()
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
    img_filename = os.path.splitext(os.path.basename(img_path))[0]
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
    print(img_filename)
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    assert img_filename == mask_filename and img.shape[:2] == mask.shape[:2]
    k = 0
    for y in range(0, img.shape[0], STRIDE):
        for x in range(0, img.shape[1], STRIDE):
            img_tile = img[y:y + TARGET_SIZE, x:x + TARGET_SIZE]
            mask_tile = mask[y:y + TARGET_SIZE, x:x + TARGET_SIZE]

            if img_tile.shape[0] == TARGET_SIZE and img_tile.shape[1] == TARGET_SIZE:
                
                out_img_path = os.path.join(OUTPUT_DIR, "{}_{}.jpg".format(img_filename, k))
                cv2.imwrite(out_img_path, img_tile)
                f = open(LIST_FILE, "a")
                f.write(img_filename+'_'+str(k)+'\n')
                f.close()
                out_mask_path = os.path.join(OUTPUT_DIR, "{}_{}_m.png".format(mask_filename, k))
                cv2.imwrite(out_mask_path, mask_tile)

            k += 1

    print("Processed {} {}/{}".format(img_filename, i + 1, len(img_paths)))