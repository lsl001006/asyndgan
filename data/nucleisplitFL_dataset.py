import imp
import os.path

import h5py
from PIL import Image
import torch
import torch.utils.data as data
import numpy as np
from torchvision.utils import save_image

import data.preprocess.mytransforms as my_transforms
from skimage import morphology
from data.base_dataset import BaseDataset, get_params, get_transform
import pdb

def get_train_imgs_list(h5_filepath):
    img_list = []
    
    with h5py.File(h5_filepath, 'r') as h5_file:
        img_filenames = list(h5_file['images'].keys())
        conditon_filenames = list(h5_file['labels'].keys())
        label_filenames = list(h5_file['labels_ternary'].keys())
        weight_filenames = list(h5_file['weight_maps'].keys())

        for img_name in img_filenames:
            if img_name in label_filenames and img_name in conditon_filenames and img_name in weight_filenames:
                item = ('images/{:s}'.format(img_name),
                        'labels/{:s}'.format(img_name),
                        'labels_ternary/{:s}'.format(img_name),
                        'weight_maps/{:s}'.format(img_name))
                img_list.append(tuple(item))

    return img_list




class LabelEncoding(object):
    """
    Encoding the label, computes boundary individually
    """
    def __init__(self, radius=1):
        self.radius = radius

    def __call__(self, label):
        if not isinstance(label, np.ndarray):
            label = np.array(label)

        # ternary label: one channel (0: background, 1: inside, 2: boundary) #
        new_label = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
        new_label[label[:, :, 0] > 255*0.5] = 1  # inside
        boun = morphology.dilation(new_label) & (~morphology.erosion(new_label, morphology.disk(self.radius)))
        new_label[boun > 0] = 2  # boundary

        label = Image.fromarray(new_label.astype(np.uint8))

        return label

class NucleiSegTrainDataset_FL(BaseDataset):
    def __init__(self, opt, idx=None, img_size=256):    
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if idx is None:
            h5_name = "train_all.h5"
        else:
            if idx ==0:
                h5_name = "train_breast.h5"
            elif idx ==1:
                h5_name = "train_kidney.h5"
            elif idx ==2:
                h5_name = "train_liver.h5"
            elif idx ==3:
                h5_name = "train_prostate.h5"

        print(f"Load: {h5_name}")
        super().__init__(opt)
        self.h5_filepath = os.path.join(opt.dataroot, h5_name)

        self.transform = my_transforms.Compose([
                # my_transforms.RandomResize(0.8, 1.25),
                my_transforms.RandomHorizontalFlip(),
                my_transforms.RandomVerticalFlip(),
                # my_transforms.RandomAffine(0.3),
                # my_transforms.RandomRotation(90),
                my_transforms.RandomCrop(img_size),
                my_transforms.LabelEncoding(),
                my_transforms.ToTensor(2),
                my_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 2)]
            )

        self.img_list = get_train_imgs_list(self.h5_filepath)
        
        if len(self.img_list) == 0:
            raise(RuntimeError('Found 0 image pairs in given directories.'))



    def __getitem__(self, index):
        with h5py.File(self.h5_filepath, 'r') as h5_file:
            img_path, condition_path,  label_path, weight_map_path = self.img_list[index]
            
            img, condition, label, weight_map = h5_file[img_path][()], h5_file[condition_path][()], h5_file[label_path][()], h5_file[weight_map_path][()]
            
            if np.max(label) == 1:
                label = (label * 255).astype(np.uint8)
            
            # img.type is uint8
            img = Image.fromarray(img, 'RGB')
            condition = Image.fromarray(condition).convert('RGB')
            label = Image.fromarray(label)
            weight_map = Image.fromarray(weight_map)
            
            # visualize
            # img.save("visual_load/img_{}.png".format(index))
            # label.save("visual_load/label_{}.png".format(index))
            # condition.save("visual_load/condition_{}.png".format(index))
            # weight_map.save("visual_load/weightmap_{}.png".format(index))
            
            img, condition, weight_map, label = self.transform((img, condition, weight_map, label))
            
            # # visualize
            # save_image(self.denormalize(img), "visual_load/img_trsfed_{}.png".format(index))
            # save_image(self.denormalize(condition), "visual_load/condition_trsfed_{}.png".format(index))
            # save_image(weight_map/255., "visual_load/weight_map_trsfed_{}.png".format(index))
            # save_image(self.convert_label(label), "visual_load/label_trsfed_{}.png".format(index))

        return {'A': condition, 'B': img, 'A_paths': str(index), 'B_paths': str(index),
                "label_ternary": label,
                "weight_map": weight_map}

    def __len__(self):
        return len(self.img_list)
    
    def denormalize(self, img):
        m, s = (0.7442, 0.5381, 0.6650), (0.1580, 0.1969, 0.1504)
        m,s = torch.tensor(m).view(-1,1,1), torch.tensor(s).view(-1,1,1)
        img = img*s+m
        return img
    
    def convert_label(self, label):
        _, h, w = label.shape
        new_label = torch.zeros((3, h, w), device=label.device)
        for i in range(3):
            new_label[i, :, :] = (label==i).int()
        return new_label

class NucleiSplitFLDataset(BaseDataset):

    def __init__(self, opt):

        self.split_db = []
        for i in range(4):
            self.split_db.append(NucleiSegTrainDataset_FL(opt, i))

    def __getitem__(self, index):

        result = {}
        for k, v in enumerate(self.split_db):
            database = v
            if index >= len(database):
                index = index % len(database)

            index_value = database[index]
            result['A_' + str(k)] = index_value['A']
            result['B_' + str(k)] = index_value['B']
            result['A_paths_' + str(k)] = index_value['A_paths']
            result['B_paths_' + str(k)] = index_value['B_paths']
            result['label_ternary_' + str(k)] = index_value['label_ternary']
            result['weight_map_' + str(k)] = index_value['weight_map']

        return result

    def __len__(self):
        """Return the total number of images in the dataset."""
        length = 0
        for i in self.split_db:
            if len(i) > length:
                length = len(i)

        return length





