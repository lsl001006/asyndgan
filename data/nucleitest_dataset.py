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


def get_test_imgs_list(h5_filepath):
    img_list = []

    with h5py.File(h5_filepath, 'r') as h5_file:
        img_filenames = list(h5_file['images'].keys())
        label_filenames = list(h5_file['labels_ternary'].keys())
        weight_filenames = list(h5_file['weight_maps'].keys())
        instance_label_filenames = list(h5_file['labels_instance'].keys())

        for img_name in img_filenames:
            if img_name in label_filenames and img_name in weight_filenames and img_name in instance_label_filenames:
                item = ('images/{:s}'.format(img_name),
                        'labels_ternary/{:s}'.format(img_name),
                        'weight_maps/{:s}'.format(img_name),
                        'labels_instance/{:s}'.format(img_name))
                img_list.append(tuple(item))

    return img_list

class NucleiTestDataset(BaseDataset):
    def __init__(self, opt):

        self.h5_filepath = os.path.join(opt.dataroot, "test.h5")
        self.transform = my_transforms.Compose([
                my_transforms.LabelEncoding(),
                my_transforms.ToTensor(1),
                my_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 1)]
            )
        
        self.img_list = get_test_imgs_list(self.h5_filepath)
        if len(self.img_list) == 0:
            raise(RuntimeError('Found 0 image pairs in given directories.'))

    def __getitem__(self, index):
        with h5py.File(self.h5_filepath, 'r') as h5_file:
            img_path, label_path, weight_map_path, instance_label_path = self.img_list[index]
            img, label, weight_map, instance_label = h5_file[img_path][()], h5_file[label_path][()], \
                                                     h5_file[weight_map_path][()], h5_file[instance_label_path][()]
            img = Image.fromarray(img, 'RGB')
            label = Image.fromarray(label)
            weight_map = Image.fromarray(weight_map)
            instance_label = torch.from_numpy(instance_label.astype(np.int16))
            
            # # visualize
            # img.save("visual_load/img_{}.png".format(index))
            # label.save("visual_load/label_{}.png".format(index))
            # weight_map.save("visual_load/weightmap_{}.png".format(index))

            img, weight_map, label = self.transform((img, weight_map, label))
            
            # visualize
            # save_image(self.denormalize(img), "visual_load/img_trsfed_{}.png".format(index))
            # save_image(weight_map/255., "visual_load/weight_map_trsfed_{}.png".format(index))
            # save_image(self.convert_label(label), "visual_load/label_trsfed_{}.png".format(index))

        return {'B': img, 'A_paths': str(index), 'B_paths': str(index),
                "label_ternary": label,
                "weight_map": weight_map,
                "instance_label": instance_label}

    def __len__(self):
        return len(self.img_list)
    
    def denormalize(self, img):
        m, s = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        m,s = torch.tensor(m).view(-1,1,1), torch.tensor(s).view(-1,1,1)
        img = img*s+m
        return img
    
    def convert_label(self, label):
        _, h, w = label.shape
        new_label = torch.zeros((3, h, w), device=label.device)
        for i in range(3):
            new_label[i, :, :] = (label==i).int()
        return new_label